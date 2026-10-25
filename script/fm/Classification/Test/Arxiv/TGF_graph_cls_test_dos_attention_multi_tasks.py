"""
Assumption:
    Train a temporal graph classification foundation model and a decoder for each dataset.
    The foundation model is trained to predict the graph-level properties.
    There are multiple properties to predict, and the model is trained to predict all of them.
=======
Date:
    2024-11-25

"""

import os
import math
import sys
import timeit
from collections import deque

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_percentage_error
import random
import wandb
import yaml
import logging



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from script.nn.models.Decoder import Decoder, MultiHeadDecoder
from script.nn.models.AttentionDOS import AttentionDOS
from script.nn.models.GraphDecoder import MultiPropertyMLP, MLP, MLP_hardcoded
from script.nn.models.TGCN import RecurrentGCN_TGCN, RecurrentGCN_TGCN_one_layer, RecurrentGCN_TGCN_one_layer_ssm
from script.utils.tg_utils import readout_function
from script.utils.tgf_param import *
from script.utils.config import args
from script.utils.util import set_random, logger, init_logger, save_or_append_dict_to_csv
from script.utils.early_stopping import EarlyStopping
from script.utils.data_util import load_multiple_datasets, prepare, multi_datasets_attributes_loading, \
    extra_dataset_attributes_loading, load_multiple_datasets_modified
from script.configs.data_spec import dataset_pack_dict, DATA_PATH, SAVED_MODEL_PATH
from script.utils.data_util import load_DOS_snapshot_feature


def tg_test(args, encoder, attention_dos, decoder,
            data_list, test_snapshots, dataset_idx, t_node_feats, edge_attr, t_graph_labels, t_graph_feats,
            hidden_state, dos_memory):
    """
    Test the temporal graph classification foundation model.
    """

    tg_labels, tg_preds = [], []

    encoder.eval()
    decoder.eval()
    attention_dos.eval()

    for snapshot_idx in test_snapshots[dataset_idx]:
        with torch.no_grad():
            # x = torch.from_numpy(t_node_feats[snapshot_idx]).to(torch.float32).to(args.device)
            x = torch.from_numpy(np.array(t_node_feats[snapshot_idx], dtype=np.float32)).to(args.device)

            edge_index = prepare(data_list[dataset_idx], snapshot_idx, args)[:3]
            embeddings, hidden_state = encoder(x, edge_index, edge_attr, hidden_state)
            hidden_state = hidden_state.detach()

            # Convert memory buffer to tensor (batch_size = 1)
            dos_memory_tensor = torch.stack(list(dos_memory)).unsqueeze(0)  # Shape: (1, memory_size, feature_dim)

            # Get the current DOS vector
            current_dos = torch.tensor(t_graph_feats[snapshot_idx], dtype=torch.float32).to(args.device).unsqueeze(
                0)  # Shape: (1, feature_dim)

            # Compute Attention-Based DOS
            tg_dos_feat = attention_dos(dos_memory_tensor, current_dos).squeeze(0)  # Shape: (feature_dim,)

            # Update memory (deque ensures fixed size)
            dos_memory.append(current_dos.squeeze(0))

            # graph-level embeddings
            if args.set_green_readout_TGCN:
                non_zero_rows = (x.abs().sum(dim=1) != 0).sum()
                tg_embeddings = readout_function(embeddings[0:non_zero_rows], args.graph_agg)
            else:
                tg_embeddings = readout_function(embeddings, args.graph_agg)

            if args.set_green_hidden_layer_TGCN:
                non_zero_rows = (x.abs().sum(dim=1) != 0).sum()
                # Create a mask
                mask = torch.zeros_like(hidden_state)
                mask[:non_zero_rows, :] = hidden_state[:non_zero_rows, :]

                # Assign masked tensor back
                hidden_state = mask

            if args.concat_att_dos_to_original:
                tg_agg_feat = torch.cat((tg_embeddings, tg_dos_feat, current_dos.squeeze(0)), dim=0)
            else:
                tg_agg_feat = torch.cat((tg_embeddings, tg_dos_feat), dim=0)

            # graph prediction
            tg_label = t_graph_labels[snapshot_idx, :].float()
            tg_pred = decoder(tg_agg_feat.view(1, tg_agg_feat.size()[0])).squeeze(0)

            # save labels and predictions per snapshot
            tg_labels.append(tg_label)
            tg_preds.append(tg_pred)

    pdps_labels = [tensor.detach().cpu().numpy() for tensor in tg_labels]  # detach, move to CPU and convert to NumPy
    pdps_preds = [tensor.detach().cpu().numpy() for tensor in tg_preds]  # detach, move to CPU and convert to NumPy

    flat_per_data_per_snap_labels = np.array(pdps_labels)
    flat_per_data_per_snap_preds = np.array(pdps_preds)

    test_auc = roc_auc_score(flat_per_data_per_snap_labels, flat_per_data_per_snap_preds)
    test_ap = average_precision_score(flat_per_data_per_snap_labels, flat_per_data_per_snap_preds)

    test_auc_per_task = roc_auc_score(flat_per_data_per_snap_labels, flat_per_data_per_snap_preds, average=None)
    test_ap_per_task = average_precision_score(flat_per_data_per_snap_labels, flat_per_data_per_snap_preds,
                                               average=None)

    return test_auc, test_ap, hidden_state, test_auc_per_task, test_ap_per_task


def test_a_trained_model(args, encoder, attention_dos, decoder, logger,
                         dataset_names, data_list, train_snapshots, val_snapshots, test_snapshots,
                         num_datasets, dos_memory,
                         ckpt_path=None, results_filename=f"{DATA_PATH}/output/results/tgf_Multitask_results_TEST.csv"):
    """
    Train the temporal graph classification foundation model.
    """
    start_all_dataset = timeit.default_timer()

    logger.info(f"load model from: {ckpt_path}")

    pretrained_model_dict = torch.load(ckpt_path)
    # Filter out 'internal_memory' from the saved state_dict
    encoder_state_dict = pretrained_model_dict["encoder"]
    filtered_encoder_state_dict = {k: v for k, v in encoder_state_dict.items() if "internal_memory" not in k}
    encoder.load_state_dict(filtered_encoder_state_dict, strict=False)

    # encoder.load_state_dict(pretrained_model_dict["encoder"])
    decoder.load_state_dict(pretrained_model_dict["decoder"])
    attention_dos.load_state_dict(pretrained_model_dict["attention_dos"])

    encoder = encoder.to(args.device)
    decoder = decoder.to(args.device)
    attention_dos = attention_dos.to(args.device)

    dataset_performance_all = []

    for dataset_idx in range(num_datasets):
        start_dataset_time = timeit.default_timer()

        encoder.eval()
        decoder.eval()
        attention_dos.eval()

        # to make sure that the node features are also loaded
        assert args.node_feats, "Node features should be loaded for each dataset!"

        # TODO: "t_graph_feats" & "t_node_feats" are currently the same features!!!
        t_graph_labels, t_graph_feats, t_node_feats = extra_dataset_attributes_loading(args, dataset_names[dataset_idx],  multi_property=True)

        # load DOS features for graph snapshots
        if args.use_dos:
            tg_dos_feat = load_DOS_snapshot_feature(dataset_names[dataset_idx], args=args)
            t_graph_feats = tg_dos_feat  # to use DOS features for graph snapshot features

        # to save labels and predictions per snapshot
        hidden_state, edge_attr = None, None

        # ======================================== # Training set
        for snapshot_idx in train_snapshots[dataset_idx]:
            # ======================================== # level: snapshot
            with torch.no_grad():
                # prepare edge_index and node features
                edge_index = prepare(data_list[dataset_idx], snapshot_idx, args)
                x = torch.from_numpy(t_node_feats[snapshot_idx]).to(torch.float32).to(args.device)

                # forward pass
                embeddings, hidden_state = encoder(x, edge_index, edge_attr, hidden_state)
                hidden_state = hidden_state.detach()
                # Convert memory buffer to tensor (batch_size = 1)
                dos_memory_tensor = torch.stack(list(dos_memory)).unsqueeze(0)  # Shape: (1, memory_size, feature_dim)

                # Get the current DOS vector
                current_dos = torch.tensor(t_graph_feats[snapshot_idx], dtype=torch.float32).to(args.device).unsqueeze(
                    0)  # Shape: (1, feature_dim)
                # Compute Attention-Based DOS
                tg_dos_feat = attention_dos(dos_memory_tensor, current_dos).squeeze(0)  # Shape: (feature_dim,)

                # Update memory (deque ensures fixed size)
                dos_memory.append(current_dos.squeeze(0))

        # ======================================== # Validation set
        for snapshot_idx in val_snapshots[dataset_idx]:
            with torch.no_grad():
                edge_index = prepare(data_list[dataset_idx], snapshot_idx, args)
                x = torch.from_numpy(t_node_feats[snapshot_idx]).to(torch.float32).to(args.device)

                embeddings, hidden_state = encoder(x, edge_index, edge_attr, hidden_state)
                hidden_state = hidden_state.detach()

                dos_memory_tensor = torch.stack(list(dos_memory)).unsqueeze(0)  # Shape: (1, memory_size, feature_dim)

                # Get the current DOS vector
                current_dos = torch.tensor(t_graph_feats[snapshot_idx], dtype=torch.float32).to(args.device).unsqueeze(
                    0)  # Shape: (1, feature_dim)
                # Compute Attention-Based DOS
                tg_dos_feat = attention_dos(dos_memory_tensor, current_dos).squeeze(0)  # Shape: (feature_dim,)

                # Update memory (deque ensures fixed size)
                dos_memory.append(current_dos.squeeze(0))

        # ======================================== # level: dataset
        # ======================================== # Test set
        test_start_time = timeit.default_timer()
        test_auc, test_ap, test_hiddenstate,  test_auc_per_task, test_ap_per_task = tg_test(args, encoder, attention_dos, decoder, data_list, test_snapshots,
                                                      dataset_idx,
                                                      t_node_feats, edge_attr, t_graph_labels, t_graph_feats,
                                                      hidden_state, dos_memory)
        test_time = timeit.default_timer() - test_start_time
        full_forward_pass_time = timeit.default_timer() - start_dataset_time

        # Log overall test metrics
        logger.info(
            f"{dataset_idx}. Dataset: {dataset_names[dataset_idx]}, Test AUC: {test_auc}, Test AP: {test_ap}, Time: {test_time}, Full Dataset Pass Time: {full_forward_pass_time}"
        )

        # Log per-task test metrics
        num_tasks = len(supported_properties)
        for task_idx in range(num_tasks):
            task_name = list(supported_properties.keys())[task_idx]  # Get task name dynamically

            logger.info(
                f"\tTask {task_name} - Test AUC: {test_auc_per_task[task_idx]}, Test AP: {test_ap_per_task[task_idx]}"
            )

        # Log to WandB
        if args.wandb:
            wandb_log_dict = {
                "dataset_test": dataset_names[dataset_idx],
                "test_auc": test_auc,
                "test_ap": test_ap,
                "test_time": test_time,
                "full_dataset_pass_time": full_forward_pass_time,
            }

            # Add per-task test metrics to WandB log
            for task_idx in range(num_tasks):
                task_name = list(supported_properties.keys())[task_idx]
                wandb_log_dict[f"test_auc_{task_name}"] = test_auc_per_task[task_idx]
                wandb_log_dict[f"test_ap_{task_name}"] = test_ap_per_task[task_idx]

            wandb.log(wandb_log_dict)

        # Store dataset performance including per-task metrics
        dataset_performance = {
            "seed": args.seed,
            "model": args.model,
            "DOS": args.use_dos,
            "data_pack_train": args.data_pack_train,
            "data_pack_test": args.data_pack_test,
            "dataset": dataset_names[dataset_idx],
            "test_auc": test_auc,
            "test_ap": test_ap,
            "test_time": test_time,
            "full_forward_pass_time": full_forward_pass_time,
        }

        # Add per-task metrics to dataset_performance
        for task_idx in range(num_tasks):
            task_name = list(supported_properties.keys())[task_idx]
            dataset_performance[f"test_auc_{task_name}"] = test_auc_per_task[task_idx]
            dataset_performance[f"test_ap_{task_name}"] = test_ap_per_task[task_idx]
        dataset_performance_all.append(dataset_performance)

        save_or_append_dict_to_csv(dataset_performance, results_filename)

    total_all_dataset_time = timeit.default_timer() - start_all_dataset
    logger.info(f"Total processing time for all dataset: {total_all_dataset_time}")

    return dataset_performance_all


def main():
    """
    Main flow...
    """
    if args.use_dos:
        print("Using DOS as graph-level features!")
        if args.concat_att_dos_to_original:
            num_graph_level_feats = 40
        else:
            num_graph_level_feats = 20

    else:
        print("Using engineered features as graph-level features!")
        num_graph_level_feats = 4

    seed = args.seed
    model_name = args.model
    val_ratio = args.val_ratio
    test_ratio = args.test_ratio
    wandb_activate = args.wandb
    lr = args.lr
    nfeat = args.nfeat
    model_parameter_scaling_factor = args.model_parameter_scaling_factor
    nhid = args.nhid * model_parameter_scaling_factor
    nout = args.nout * model_parameter_scaling_factor  # NOTE: output dimension of the encoder
    data_pack_train = args.data_pack_train
    data_pack_test = args.data_pack_test  # should be "pack_test"

    # select and load a data package
    selected_data_pack = data_pack_test

    if selected_data_pack != "pack_1":  # TODO: currently, we only have 1 variation for each data pack containing more than 1 datasets
        args.data_pack_idx_start = 0
        args.data_pack_idx_end = 1

    data_pack_list_of_list = dataset_pack_dict[selected_data_pack]

    for data_pack_list in data_pack_list_of_list[args.data_pack_idx_start: args.data_pack_idx_end]:

        # if selected_data_pack == "pack_1":
        #     run_name = f"TEST_{model_name}_{selected_data_pack}:[{','.join(data_pack_list)}]_{data_pack_train}_{lr}_{args.max_epoch}_{seed}"
        #     run_name_train = f"{model_name}_{data_pack_train}:[{','.join(data_pack_list)}]_{lr}_{args.max_epoch}_{seed}"
        # else:
        #     print(f"selected_data_pack contains {len(data_pack_list)} datasets, not including the names in the run_name!")
        #     run_name = f"TEST_{model_name}_{selected_data_pack}_{data_pack_train}_{lr}_{args.max_epoch}_{seed}"
        #     run_name_train = f"{model_name}_{data_pack_train}_{lr}_{args.max_epoch}_{seed}"

        run_name = f"TEST_{model_name}_{selected_data_pack}_{data_pack_train}_{lr}_{args.max_epoch}_{seed}"
        if args.use_dos:
            run_name_train = f"{model_name}_{data_pack_train}_{lr}_{args.max_epoch}_{seed}"
        else:
            run_name_train = f"{model_name}_{data_pack_train}_{lr}_{args.max_epoch}_{seed}_no_DOS"

        logger.info(f"TEST: Run name train: {run_name_train}")
        logger.info(f"TEST: Run name: {run_name}")

        # set random seed
        set_random(seed)

        # set logger
        log_filename = f"{DATA_PATH}/output/log/{run_name}_log.txt"
        init_logger(log_filename)
        logger.info(f"Starting {run_name}")

        # set results filename
        results_filename = f"{DATA_PATH}/output/results/tgf_Multitask_results_TEST.csv"

        dataset_names, data_list = load_multiple_datasets_modified(data_pack_list)
        num_datasets = len(data_list)

        # data preparation and split
        num_snapshots = [data_list[i]["time_length"] for i in range(num_datasets)]
        val_splits = [math.ceil(num_snapshots[i] * val_ratio) for i in range(num_datasets)]
        test_splits = [math.ceil(num_snapshots[i] * test_ratio) for i in range(num_datasets)]

        train_snapshots = [list(range(0, num_snapshots[i] - val_splits[i] - test_splits[i])) for i in
                           range(num_datasets)]
        val_snapshots = [
            list(range(num_snapshots[i] - val_splits[i] - test_splits[i], num_snapshots[i] - test_splits[i])) for i in
            range(num_datasets)]
        test_snapshots = [list(range(num_snapshots[i] - test_splits[i], num_snapshots[i])) for i in range(num_datasets)]

        # define model
        encoder = RecurrentGCN_TGCN_one_layer_ssm(
            node_features=nfeat,
            hid_dim=nhid,
            out_dim=nout,
        )

        # decoder = Decoder(in_channels=nout + num_graph_level_feats,
        #                   drop_out=[args.dropout], out_channels=len(supported_properties))

        if not args.use_multi_head:
            print("Using a multi-head decoder for all tasks")

            decoder = Decoder(in_channels=nout + num_graph_level_feats,
                              hids_size=[nhid],
                              drop_out=[args.dropout], out_channels=len(supported_properties))
        else:
            print("Using multi decoders for each task")
            decoder = MultiHeadDecoder(
                in_channels=nout + num_graph_level_feats,
                hids_size=[nhid],
                drop_out=[args.dropout], out_channels=1,
                heads_num=len(supported_properties)
            )

        # Initialize AttentionDOS
        attention_dos = AttentionDOS(input_dim=20, memory_size=args.dos_memory_size).to(args.device)

        # Initialize a fixed-size memory buffer
        dos_memory = deque([torch.zeros(20).to(args.device) for _ in range(args.dos_memory_size)],
                           maxlen=args.dos_memory_size)

        # start the main process
        # ================================
        # ==== Test a pretrain model =====
        # ================================

        if wandb_activate:
            wandb.init(
                entity=args.entity,
                settings=wandb.Settings(start_method='fork'),
                project=args.project,
                name=f"{run_name}",
                tags=[
                    # str(seed),
                    "TEST",
                    str(model_name),
                    str(data_pack_train),
                    ",".join(data_pack_test),
                ],
                config=args,
            )

        numel_encoder = sum([p.numel() for p in encoder.parameters()])
        numel_decoder = sum([p.numel() for p in decoder.parameters()])
        numel_attention_dos = sum([p.numel() for p in attention_dos.parameters()])
        logger.info(f"Number of parameters in encoder: {numel_encoder}")
        logger.info(f"Number of parameters in decoder: {numel_decoder}")
        logger.info(f"Number of parameters in attention_dos: {numel_attention_dos}")

        # the trained model should be loaded from a checkpoint
        ckpt_path = f"{DATA_PATH}/output/ckpts/{run_name_train}.pth"
        dataset_performance_all = test_a_trained_model(args, encoder, attention_dos, decoder, logger,
                                                       dataset_names, data_list, train_snapshots, val_snapshots,
                                                       test_snapshots,
                                                       num_datasets,
                                                       dos_memory,
                                                       ckpt_path=ckpt_path, results_filename=results_filename)

        # logger.info(f"Best epoch performance: {dataset_performance_all}")

        if wandb_activate:
            wandb.finish()


if __name__ == '__main__':
    main()
