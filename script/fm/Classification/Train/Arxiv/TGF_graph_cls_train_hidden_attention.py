"""
Assumption:
    Train a temporal graph classification foundation model and a decoder for each dataset.
    The foundation model is trained to predict the graph-level properties.
    There are multiple properties to predict, and the model is trained to predict all of them.
=======
Date:
    2024-11-07

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
from script.nn.models.AttentionHidden import AttentionHidden
from script.nn.models.AttentionDOS import AttentionDOS
from script.nn.models.GraphDecoder import MLP_hardcoded
from script.nn.models.TGCN import RecurrentGCN_TGCN, RecurrentGCN_TGCN_one_layer
from script.utils.tg_utils import readout_function
from script.utils.tgf_param import *
from script.utils.config import args
from script.utils.util import set_random, logger, init_logger, save_or_append_dict_to_csv
from script.utils.early_stopping import EarlyStopping
from script.utils.data_util import load_multiple_datasets, prepare, multi_datasets_attributes_loading, \
    extra_dataset_attributes_loading, load_multiple_datasets_modified
from script.configs.data_spec import dataset_pack_dict, DATA_PATH, SAVED_MODEL_PATH
from script.utils.data_util import load_DOS_snapshot_feature


def tg_test(args, encoder, attention_hidden, decoder,
            data_list, test_snapshots, dataset_idx, t_node_feats, edge_attr, t_graph_labels, t_graph_feats,
            hidden_state, hidden_memory):
    """
    Test the temporal graph classification foundation model.
    """

    tg_labels, tg_preds = [], []

    encoder.eval()
    decoder.eval()
    attention_hidden.eval()

    for snapshot_idx in test_snapshots[dataset_idx]:
        with torch.no_grad():
            # prepare edge_index and node features
            edge_index = prepare(data_list[dataset_idx], snapshot_idx, args)
            # x = torch.from_numpy(t_node_feats[snapshot_idx]).to(torch.float32).to(args.device)
            x = torch.from_numpy(np.array(t_node_feats[snapshot_idx], dtype=np.float32)).to(args.device)

            # # Check which rows are not all zeros
            # non_zero_rows = (x.abs().sum(dim=1) != 0).sum()
            # print("Number of non zeros: {}".format(non_zero_rows))

            # forward pass
            embeddings, hidden_state = encoder(x, edge_index, edge_attr, hidden_state)
            hidden_state = hidden_state.detach()
            # Get the current DOS vector
            current_dos = torch.tensor(t_graph_feats[snapshot_idx], dtype=torch.float32).to(args.device).unsqueeze(
                0)  # Shape: (1, feature_dim)


            # Get the current DOS vector

            hidden_memory_tensor = torch.stack(list(hidden_memory)).unsqueeze(
                0)  # Shape: (1, memory_size, 33190, 16)

            tg_dos_feat = current_dos.squeeze(0)
            # Update memory (deque ensures fixed size)

            updated_hidden_state = attention_hidden(hidden_memory_tensor, hidden_state)  # Get updated state
            hidden_state = updated_hidden_state.clone()  # Avoid modifying in-place
            hidden_memory.append(hidden_state.detach())  # Store in memory safely


            # non_zero_rows = (hidden_state.abs().sum(dim=1) != 0).sum()
            # print("Number of non zeros HIDDEN: {}".format(non_zero_rows))

            # graph-level embeddings
            # USING JUST GREEN PART OF ACTIVATED nodes
            if args.set_green_readout_TGCN:
                non_zero_rows = (x.abs().sum(dim=1) != 0).sum()
                tg_embeddings = readout_function(embeddings[0:non_zero_rows], args.graph_agg)
            else:
                tg_embeddings = readout_function(embeddings, args.graph_agg)

            if args.set_green_hidden_layer_TGCN:
                # Create a mask
                non_zero_rows = (x.abs().sum(dim=1) != 0).sum()
                mask = torch.zeros_like(hidden_state)
                mask[:non_zero_rows, :] = hidden_state[:non_zero_rows, :]

                # Assign masked tensor back
                hidden_state = mask

            tg_agg_feat = torch.cat((tg_embeddings, tg_dos_feat), dim=0)

            # graph prediction
            tg_label = t_graph_labels[snapshot_idx].float().view(1, )
            tg_pred = decoder(tg_agg_feat.view(1, tg_agg_feat.size()[0]))

            # save labels and predictions per snapshot
            tg_labels.append(tg_label)
            tg_preds.append(tg_pred)

    pdps_labels = [tensor.detach().cpu().numpy() for tensor in tg_labels]  # detach, move to CPU and convert to NumPy
    pdps_preds = [tensor.detach().cpu().numpy() for tensor in tg_preds]  # detach, move to CPU and convert to NumPy

    # Concatenate the lists
    flat_per_data_per_snap_labels = np.concatenate(pdps_labels)
    flat_per_data_per_snap_preds = np.concatenate(pdps_preds)

    test_auc = roc_auc_score(flat_per_data_per_snap_labels, flat_per_data_per_snap_preds)
    test_ap = average_precision_score(flat_per_data_per_snap_labels, flat_per_data_per_snap_preds)

    return test_auc, test_ap, hidden_state


def train_val_test(args, encoder, attention_hidden, decoder, criterion, optimizer, logger,
                   dataset_names, data_list, train_snapshots, val_snapshots, test_snapshots,
                   num_datasets,
                   hidden_memory,
                   early_stopping=None,
                   results_filename=f"{DATA_PATH}/output/results/tgf_results.csv"):
    """
    Train the temporal graph classification foundation model.
    """
    encoder = encoder.to(args.device)
    decoder = decoder.to(args.device)
    attention_hidden = attention_hidden.to(args.device)

    start_train = timeit.default_timer()

    best_epoch = 0
    # Set memory size
    epoch_losses, epoch_times = [], []
    epoch_train_auc, epoch_train_ap = [], []
    epoch_val_auc, epoch_val_ap, epoch_val_time = [], [], []
    epoch_test_auc, epoch_test_ap, epoch_test_time = [], [], []

    for epoch in range(args.max_epoch):
        logger.info(f"Epoch: {epoch}")
        start_epoch = timeit.default_timer()

        encoder.train()
        decoder.train()
        attention_hidden.train()

        dataset_losses, dataset_times = [], []
        dataset_train_aucs, dataset_train_aps = [], []
        dataset_val_aucs, dataset_val_aps = [], []
        dataset_val_times = []
        dataset_test_aucs, dataset_test_aps = [], []
        dataset_test_times = []

        # shuffle the datasets
        dataset_rnd_order = random.sample(range(num_datasets), num_datasets)
        for dataset_idx in dataset_rnd_order:
            start_dataset_time = timeit.default_timer()
            logger.info(f"Epoch: {epoch}, Dataset: {dataset_idx}")

            encoder.train()
            decoder.train()
            attention_hidden.train()

            # load dataset features
            # t_graph_labels, t_graph_feat = extra_dataset_attributes_loading(args, dataset_names[dataset_idx])  # TODO: we should not load dataset each time!!!

            # to make sure that the node features are also loaded
            assert args.node_feats, "Node features should be loaded for each dataset!"

            # TODO: "t_graph_feats" & "t_node_feats" are currently the same features!!!
            t_graph_labels, t_graph_feats, t_node_feats = extra_dataset_attributes_loading(args,
                                                                                           dataset_names[dataset_idx])

            # load DOS features for graph snapshots
            # TODO: here we are replacing the features not concat to the previous features
            if args.use_dos:
                tg_dos_feat = load_DOS_snapshot_feature(dataset_names[dataset_idx], args=args)
                t_graph_feats = tg_dos_feat  # to use DOS features for graph snapshot features

            # to save labels and predictions per snapshot
            per_dataset_per_snap_labels, per_dataset_per_snap_preds = [], []
            per_dataset_per_snap_losses = []
            hidden_state, edge_attr = None, None

            for snapshot_idx in train_snapshots[dataset_idx]:
                # ======================================== # level: snapshot
                optimizer.zero_grad()

                # prepare edge_index and node features
                edge_index = prepare(data_list[dataset_idx], snapshot_idx, args)
                # x = torch.from_numpy(t_node_feats[snapshot_idx]).to(torch.float32).to(args.device)
                x = torch.from_numpy(np.array(t_node_feats[snapshot_idx], dtype=np.float32)).to(args.device)

                # # Check which rows are not all zeros
                # non_zero_rows = (x.abs().sum(dim=1) != 0).sum()
                # print("Number of non zeros: {}".format(non_zero_rows))

                # forward pass
                embeddings, hidden_state = encoder(x, edge_index, edge_attr, hidden_state)
                hidden_state = hidden_state.detach()



                current_dos = torch.tensor(t_graph_feats[snapshot_idx], dtype=torch.float32).to(args.device).unsqueeze(
                    0)  # Shape: (1, feature_dim)

                # Convert memory buffer to tensor (batch_size = 1)
                hidden_memory_tensor = torch.stack(list(hidden_memory)).unsqueeze(
                    0)  # Shape: (1, memory_size, 33190, 16)


                tg_dos_feat = current_dos.squeeze(0)
                # Update memory (deque ensures fixed size)

                updated_hidden_state = attention_hidden(hidden_memory_tensor, hidden_state)  # Get updated state
                hidden_state = updated_hidden_state.detach().clone().requires_grad_()  # Maintain gradient tracking
                hidden_memory.append(hidden_state.detach())  # Store in memory safely

                # non_zero_rows = (hidden_state.abs().sum(dim=1) != 0).sum()
                # print("Number of non zeros HIDDEN: {}".format(non_zero_rows))

                # graph-level embeddings
                # USING JUST GREEN PART OF ACTIVATED nodes
                if args.set_green_readout_TGCN:
                    non_zero_rows = (x.abs().sum(dim=1) != 0).sum()
                    tg_embeddings = readout_function(embeddings[0:non_zero_rows], args.graph_agg)
                else:
                    tg_embeddings = readout_function(embeddings, args.graph_agg)

                if args.set_green_hidden_layer_TGCN:
                    # Create a mask
                    non_zero_rows = (x.abs().sum(dim=1) != 0).sum()
                    mask = torch.zeros_like(hidden_state)
                    mask[:non_zero_rows, :] = hidden_state[:non_zero_rows, :]

                    # Assign masked tensor back
                    hidden_state = mask


                tg_agg_feat = torch.cat((tg_embeddings, tg_dos_feat), dim=0)

                # graph prediction
                tg_label = t_graph_labels[snapshot_idx].float().view(1, )
                tg_pred = decoder(tg_agg_feat.view(1, tg_agg_feat.size()[0]))

                # loss
                loss = criterion(tg_pred, tg_label)
                loss.backward()
                optimizer.step()
                per_dataset_per_snap_losses.append(loss.item())

                # for later evaluation
                per_dataset_per_snap_labels.append(tg_label)
                per_dataset_per_snap_preds.append(tg_pred)

            # ======================================== # level: dataset
            # all snapshots in one dataset are done
            dataset_losses.append(np.mean(per_dataset_per_snap_losses))

            pdps_labels = [tensor.detach().cpu().numpy() for tensor in
                           per_dataset_per_snap_labels]  # detach, move to CPU and convert to NumPy
            pdps_preds = [tensor.detach().cpu().numpy() for tensor in
                          per_dataset_per_snap_preds]  # detach, move to CPU and convert to NumPy

            # Concatenate the lists
            flat_per_data_per_snap_labels = np.concatenate(pdps_labels)
            flat_per_data_per_snap_preds = np.concatenate(pdps_preds)

            train_auc = roc_auc_score(flat_per_data_per_snap_labels, flat_per_data_per_snap_preds)
            train_ap = average_precision_score(flat_per_data_per_snap_labels, flat_per_data_per_snap_preds)
            dataset_train_aucs.append(train_auc)
            dataset_train_aps.append(train_ap)

            # validation
            val_start_time = timeit.default_timer()
            val_auc, val_ap, val_hiddenstate = tg_test(args, encoder, attention_hidden, decoder, data_list, val_snapshots,
                                                       dataset_idx,
                                                       t_node_feats, edge_attr, t_graph_labels, t_graph_feats,
                                                       hidden_state, hidden_memory)
            dataset_val_times.append(timeit.default_timer() - val_start_time)
            dataset_val_aucs.append(val_auc)
            dataset_val_aps.append(val_ap)

            # test
            test_start__time = timeit.default_timer()
            test_auc, test_ap, test_hiddenstate = tg_test(args, encoder, attention_hidden, decoder, data_list,
                                                          test_snapshots,
                                                          dataset_idx,
                                                          t_node_feats, edge_attr, t_graph_labels, t_graph_feats,
                                                          val_hiddenstate, hidden_memory)
            dataset_test_times.append(timeit.default_timer() - test_start__time)
            dataset_test_aucs.append(test_auc)
            dataset_test_aps.append(test_ap)

            dataset_times.append(timeit.default_timer() - start_dataset_time)

        # ========================================  # level: epoch
        # average the results for all datasets for one epoch

        avg_epoch_loss = np.mean(dataset_losses)
        epoch_losses.append(avg_epoch_loss)

        avg_all_dataset_train_auc, avg_all_dataset_train_ap = np.mean(dataset_train_aucs), np.mean(dataset_train_aps)
        epoch_train_auc.append(avg_all_dataset_train_auc)
        epoch_train_ap.append(avg_all_dataset_train_ap)

        avg_all_dataset_val_auc, avg_all_dataset_val_ap = np.mean(dataset_val_aucs), np.mean(dataset_val_aps)
        avg_all_dataset_val_time = np.mean(dataset_val_times)
        epoch_val_auc.append(avg_all_dataset_val_auc)
        epoch_val_ap.append(avg_all_dataset_val_ap)
        epoch_val_time.append(avg_all_dataset_val_time)

        avg_all_dataset_test_auc, avg_all_dataset_test_ap = np.mean(dataset_test_aucs), np.mean(dataset_test_aps)
        avg_all_dataset_test_time = np.mean(dataset_test_times)
        epoch_test_auc.append(avg_all_dataset_test_auc)
        epoch_test_ap.append(avg_all_dataset_test_ap)
        epoch_test_time.append(avg_all_dataset_test_time)

        epoch_times.append(timeit.default_timer() - start_epoch)
        logger.info(
            f"Epoch: {epoch}, Train loss: {avg_epoch_loss}, Train AUC: {avg_all_dataset_train_auc}, Train AP: {avg_all_dataset_train_ap}, Time: {np.mean(epoch_times)}")
        logger.info(f"\t\tVal AUC: {avg_all_dataset_val_auc}, Val AP: {avg_all_dataset_val_ap}")
        logger.info(f"\t\tTest AUC: {avg_all_dataset_test_auc}, Test AP: {avg_all_dataset_test_ap}")

        if args.wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_epoch_loss,
                "train_auc": avg_all_dataset_train_auc,
                "train_ap": avg_all_dataset_train_ap,
                "val_auc": avg_all_dataset_val_auc,
                "val_ap": avg_all_dataset_val_ap,
                "test_auc": avg_all_dataset_test_auc,
                "test_ap": avg_all_dataset_test_ap,
            })

        # early stopping
        val_metric_indicator = [
            ("val_auc", avg_all_dataset_val_auc, True),  # higher is better
            # ("val_ap", avg_all_dataset_val_ap, True),  # we can also use AP (or both) for early stopping
        ]
        early_stop, best_epoch = early_stopping.step(val_metric_indicator, epoch,
                                                     {"encoder": encoder.state_dict(), "decoder": decoder.state_dict(),
                                                      "attention_hidden": attention_hidden.state_dict()})

        if early_stop:
            break

    best_epoch_performance = {
        "seed": args.seed,
        "model": args.model,
        "data_pack": args.data_pack_train,
        "dataset": ",".join(dataset_names),
        "best_epoch": best_epoch,
        "train_auc": epoch_train_auc[best_epoch],
        "train_ap": epoch_train_ap[best_epoch],
        "val_auc": epoch_val_auc[best_epoch],
        "val_ap": epoch_val_ap[best_epoch],
        "test_auc": epoch_test_auc[best_epoch],
        "test_ap": epoch_test_ap[best_epoch],
        "avg_epoch_time": np.mean(epoch_times),
        "best_epoch_time": epoch_times[best_epoch],  # time for the best epoch
        "best_epoch_val_time": epoch_val_time[best_epoch],  # validation time for the best epoch
        "best_epoch_test_time": epoch_test_time[best_epoch],  # test time for the best epoch
    }

    save_or_append_dict_to_csv(best_epoch_performance, results_filename)

    total_train_time = timeit.default_timer() - start_train
    logger.info(f"Total Time: {total_train_time}")

    return best_epoch_performance


def main():
    """
    Main flow...
    """
    if args.use_dos:
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
    nhid = args.nhid
    nout = args.nout  # NOTE: output dimension of the encoder

    # # Load the datasets listed on the .txt file
    # selected_data_pack = "dataset_package_1.txt"  # "dataset_package_2.txt"
    # logger.info(f"Selected data package: {selected_data_pack}")
    # selected_data_path = "../../data/input/data_list/{}".format(selected_data_pack)

    # select and load a data package
    selected_data_pack = args.data_pack_train

    if selected_data_pack != "pack_1":  # TODO: currently, we only have 1 variation for each data pack containing more than 1 datasets
        args.data_pack_idx_start = 0
        args.data_pack_idx_end = 1

    data_pack_list_of_list = dataset_pack_dict[selected_data_pack]

    for data_pack_list in data_pack_list_of_list[args.data_pack_idx_start: args.data_pack_idx_end]:

        if selected_data_pack == "pack_1":
            run_name = f"{model_name}_{selected_data_pack}:[{','.join(data_pack_list)}]_{lr}_{args.max_epoch}_{seed}"
        else:
            print(
                f"selected_data_pack contains {len(data_pack_list)} datasets, not including the names in the run_name!")
            if args.use_dos:
                run_name = f"{model_name}_{selected_data_pack}_{lr}_{args.max_epoch}_{seed}"
            else:
                run_name = f"{model_name}_{selected_data_pack}_{lr}_{args.max_epoch}_{seed}_no_DOS"

        # set random seed
        set_random(seed)

        # set logger
        log_filename = f"{DATA_PATH}/output/log/{run_name}_log.txt"
        init_logger(log_filename)
        logger.info(f"Starting {run_name}")

        # set results & checkpoint paths
        results_filename = f"{DATA_PATH}/output/results/tgf_results.csv"
        ckpt_path = f"{DATA_PATH}/output/ckpts/{run_name}.pth"

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
        encoder = RecurrentGCN_TGCN_one_layer(
            node_features=nfeat,
            hid_dim=nhid,
            out_dim=nout,
        )
        # decoder = MultiPropertyMLP(
        #     in_dim=params['nout'] + num_graph_level_feats,
        #     hid_dim=params['nhid'],
        #     out_dim=1,
        #     num_properties=params['num_properties'],
        #     drop=params['dropout'],
        # )

        # # let's first try to predict only one property
        # decoder = MLP(
        #     in_dim=nout + num_graph_level_feats,
        #     hid_dim=nhid,
        #     out_dim=1,
        #     drop=dropout,
        # )
        decoder = MLP_hardcoded(
            in_dim=nout + num_graph_level_feats,
            hidden_dim_1=args.nout + num_graph_level_feats,
            hidden_dim_2=args.nout + num_graph_level_feats,
            drop=args.dropout,
        )

        # Initialize AttentionDOS
        # Ensure input_dim matches hidden_state's feature dimension (16)
        attention_hidden = AttentionHidden(input_dim=args.nhid, memory_size=args.dos_memory_size).to(args.device)


        # Initialize a fixed-size memory buffer
        # Initialize hidden memory with shape (memory_size, 33190, 16)
        hidden_memory = deque([torch.zeros(args.num_nodes, args.nhid).to(args.device) for _ in range(args.dos_memory_size)],
                              maxlen=args.dos_memory_size)

        # Add AttentionDOS to the optimizer
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()) + list(attention_hidden.parameters()),
            lr=lr,
        )

        # define losss
        criterion = torch.nn.BCEWithLogitsLoss()

        # define optimizer
        # optimizer = torch.optim.Adam(
        #     list(encoder.parameters()) + list(decoder.parameters()),
        #     lr=lr,
        # )

        early_stopping = EarlyStopping(patience=args.patience, logger=logger, save_model_path=ckpt_path)

        # start the main process
        # ===========================
        # ==== Train, val, test =====
        # ===========================
        if wandb_activate:
            wandb.init(
                entity=args.entity,
                settings=wandb.Settings(start_method='spawn'),
                project=args.project,
                name=f"{run_name}",
                tags=[
                    # str(seed),
                    str(lr),
                    str(args.max_epoch),
                    str(model_name),
                    str(selected_data_pack),
                    ",".join(data_pack_list),
                ],
                config=args,
            )

        numel_encoder = sum([p.numel() for p in encoder.parameters()])
        numel_decoder = sum([p.numel() for p in decoder.parameters()])
        numel_attention_dos = sum([p.numel() for p in attention_hidden.parameters()])

        logger.info(f"Number of parameters in encoder: {numel_encoder}")
        logger.info(f"Number of parameters in decoder: {numel_decoder}")
        logger.info(f"Number of parameters in attention_dos: {numel_attention_dos}")

        # main training procedure
        best_epoch_performance = train_val_test(args, encoder, attention_hidden, decoder, criterion, optimizer, logger,
                                                dataset_names, data_list, train_snapshots, val_snapshots,
                                                test_snapshots,
                                                num_datasets,
                                                hidden_memory,
                                                early_stopping=early_stopping, results_filename=results_filename)
        logger.info(f"Best epoch performance: {best_epoch_performance}")

        if wandb_activate:
            wandb.finish()


if __name__ == '__main__':
    main()

"""
Command to run:
(NOTE: make sure to be in `script/fm/` directory)

    python train_TGF_graph_cls.py --model TGCN --device cuda:0 --seed 1024 --lr 2e-4 --wandb --node_feats
"""
