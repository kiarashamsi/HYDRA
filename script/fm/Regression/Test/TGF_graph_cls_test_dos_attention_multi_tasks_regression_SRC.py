"""
Assumption:
    Test a temporal graph regression foundation model (multi-task REG).
    Metrics: MAE, RMSE, R^2 (overall and per-task).
=======
Date:
    2025-08-20
"""

import os
import math
import sys
import timeit
from collections import deque

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random
import wandb
import yaml
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from script.nn.models.Decoder import Decoder, MultiHeadDecoder
from script.nn.models.AttentionDOS import AttentionDOS
from script.nn.models.TGCN import RecurrentGCN_TGCN_one_layer, RecurrentGCN_TGCN_SRC
from script.utils.tg_utils import readout_function
from script.utils.tgf_param import *
from script.utils.config import args
from script.utils.util import set_random, logger, init_logger, save_or_append_dict_to_csv
from script.utils.data_util import load_multiple_datasets, prepare, multi_datasets_attributes_loading, \
    extra_dataset_attributes_loading, load_multiple_datasets_modified
from script.configs.data_spec import dataset_pack_dict, DATA_PATH, SAVED_MODEL_PATH
from script.utils.data_util import load_DOS_snapshot_feature


def compute_regression_metrics_np(labels_2d, preds_2d):
    """
    labels_2d, preds_2d: shape (N, T)
    Returns: avg_mae, avg_rmse, avg_r2, per_task_mae, per_task_rmse, per_task_r2
    """
    T = labels_2d.shape[1]
    mae_list, rmse_list, r2_list = [], [], []
    if T == 1:
        preds_2d = preds_2d.reshape(-1, 1)
    for i in range(T):
        y_true = labels_2d[:, i]
        y_pred = preds_2d[:, i]
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        try:
            r2 = r2_score(y_true, y_pred)
        except Exception:
            r2 = np.nan
        mae_list.append(mae)
        rmse_list.append(rmse)
        r2_list.append(r2)
    per_task_mae = np.array(mae_list)
    per_task_rmse = np.array(rmse_list)
    per_task_r2 = np.array(r2_list)
    avg_mae = np.nanmean(per_task_mae)
    avg_rmse = np.nanmean(per_task_rmse)
    avg_r2 = np.nanmean(per_task_r2)
    return avg_mae, avg_rmse, avg_r2, per_task_mae, per_task_rmse, per_task_r2


def tg_test(args, encoder, attention_dos, decoder,
            data_list, test_snapshots, dataset_idx, t_node_feats, edge_attr, t_graph_labels, t_graph_feats,
            hidden_state, dos_memory):
    """
    Evaluate on the test split (regression). Returns overall + per-task metrics and final hidden_state.
    NOTE: Uses prepare(... )[:3] to match your train-side tg_test style.
    """
    encoder.eval()
    decoder.eval()
    attention_dos.eval()

    all_labels, all_preds = [], []

    for snapshot_idx in test_snapshots[dataset_idx]:
        with torch.no_grad():
            x = torch.from_numpy(np.array(t_node_feats[snapshot_idx], dtype=np.float32)).to(args.device)

            # match your train tg_test usage:
            edge_index = prepare(data_list[dataset_idx], snapshot_idx, args)[:3]

            tg_embeddings, hidden_state = encoder(x, edge_index, edge_attr, hidden_state)
            hidden_state = hidden_state.detach()

            # DOS memory
            dos_memory_tensor = torch.stack(list(dos_memory)).unsqueeze(0)
            current_dos = torch.tensor(t_graph_feats[snapshot_idx], dtype=torch.float32).to(args.device).unsqueeze(0)
            tg_dos_feat = attention_dos(dos_memory_tensor, current_dos).squeeze(0)
            dos_memory.append(current_dos.squeeze(0))

            # Graph readout
            # if args.set_green_readout_TGCN:
            #     non_zero_rows = (x.abs().sum(dim=1) != 0).sum()
            #     tg_embeddings = readout_function(embeddings[0:non_zero_rows], args.graph_agg)
            # else:
            #     tg_embeddings = readout_function(embeddings, args.graph_agg)

            if args.set_green_hidden_layer_TGCN:
                non_zero_rows = (x.abs().sum(dim=1) != 0).sum()
                mask = torch.zeros_like(hidden_state)
                mask[:non_zero_rows, :] = hidden_state[:non_zero_rows, :]
                hidden_state = mask

            if args.concat_att_dos_to_original:
                tg_agg_feat = torch.cat((tg_embeddings.squeeze(0), tg_dos_feat, current_dos.squeeze(0)), dim=0)
            else:
                tg_agg_feat = torch.cat((tg_embeddings.squeeze(0), tg_dos_feat), dim=0)

            # Prediction: outputs raw regression values (MSE-trained decoder)
            tg_label = t_graph_labels[snapshot_idx, :].float().to(args.device)   # [T]
            y_pred   = decoder(tg_agg_feat.view(1, tg_agg_feat.size(0))).squeeze(0)  # [T]

            all_labels.append(tg_label.detach().cpu().numpy())
            all_preds.append(y_pred.detach().cpu().numpy())

    labels_np = np.array(all_labels)  # (N, T)
    preds_np  = np.array(all_preds)   # (N, T)

    avg_mae, avg_rmse, avg_r2, per_task_mae, per_task_rmse, per_task_r2 = compute_regression_metrics_np(labels_np, preds_np)
    return (avg_mae, avg_rmse, avg_r2,
            per_task_mae, per_task_rmse, per_task_r2,
            hidden_state)


def test_a_trained_model(args, encoder, attention_dos, decoder, logger,
                         dataset_names, data_list, train_snapshots, val_snapshots, test_snapshots,
                         num_datasets, dos_memory,
                         ckpt_path=None, results_filename=f"{DATA_PATH}/output/results/tgf_Multitask_results_regression_TEST.csv"):
    """
    Test the temporal graph regression foundation model.
    Uses the same input signature as your BC tester; no new pipeline pieces.
    """
    start_all_dataset = timeit.default_timer()

    logger.info(f"load model from: {ckpt_path}")

    pretrained_model_dict = torch.load(ckpt_path, map_location=args.device)
    # Filter out 'internal_memory' from encoder state dict if present
    encoder_state_dict = pretrained_model_dict["encoder"]
    filtered_encoder_state_dict = {k: v for k, v in encoder_state_dict.items() if "internal_memory" not in k}
    encoder.load_state_dict(filtered_encoder_state_dict, strict=False)
    decoder.load_state_dict(pretrained_model_dict["decoder"])
    attention_dos.load_state_dict(pretrained_model_dict["attention_dos"])

    encoder = encoder.to(args.device)
    decoder = decoder.to(args.device)
    attention_dos = attention_dos.to(args.device)

    dataset_performance_all = []
    task_names = list(supported_properties.keys())
    num_tasks = len(task_names)

    for dataset_idx in range(num_datasets):
        start_dataset_time = timeit.default_timer()

        encoder.eval()
        decoder.eval()
        attention_dos.eval()

        assert args.node_feats, "Node features should be loaded for each dataset!"

        # Snapshot labels & features
        t_graph_labels, t_graph_feats, t_node_feats = extra_dataset_attributes_loading(
            args, dataset_names[dataset_idx], multi_property=True
        )

        # DOS features override, if used during training
        if args.use_dos:
            tg_dos_feat = load_DOS_snapshot_feature(dataset_names[dataset_idx], args=args)
            t_graph_feats = tg_dos_feat

        hidden_state, edge_attr = None, None

        # -------- Adaptation pass: TRAIN (no grad) --------
        for snapshot_idx in train_snapshots[dataset_idx]:
            with torch.no_grad():
                edge_index = prepare(data_list[dataset_idx], snapshot_idx, args)
                x = torch.from_numpy(np.array(t_node_feats[snapshot_idx], dtype=np.float32)).to(args.device)

                embeddings, hidden_state = encoder(x, edge_index, edge_attr, hidden_state)
                hidden_state = hidden_state.detach()

                dos_memory_tensor = torch.stack(list(dos_memory)).unsqueeze(0)
                current_dos = torch.tensor(t_graph_feats[snapshot_idx], dtype=torch.float32).to(args.device).unsqueeze(0)
                tg_dos_feat = attention_dos(dos_memory_tensor, current_dos).squeeze(0)
                dos_memory.append(current_dos.squeeze(0))

        # -------- Adaptation pass: VAL (no grad) ----------
        for snapshot_idx in val_snapshots[dataset_idx]:
            with torch.no_grad():
                edge_index = prepare(data_list[dataset_idx], snapshot_idx, args)
                x = torch.from_numpy(np.array(t_node_feats[snapshot_idx], dtype=np.float32)).to(args.device)

                embeddings, hidden_state = encoder(x, edge_index, edge_attr, hidden_state)
                hidden_state = hidden_state.detach()

                dos_memory_tensor = torch.stack(list(dos_memory)).unsqueeze(0)
                current_dos = torch.tensor(t_graph_feats[snapshot_idx], dtype=torch.float32).to(args.device).unsqueeze(0)
                tg_dos_feat = attention_dos(dos_memory_tensor, current_dos).squeeze(0)
                dos_memory.append(current_dos.squeeze(0))

        # -------- Test split --------
        test_start_time = timeit.default_timer()
        (test_mae, test_rmse, test_r2,
         test_mae_pt, test_rmse_pt, test_r2_pt,
         test_hiddenstate) = tg_test(args, encoder, attention_dos, decoder,
                                     data_list, test_snapshots, dataset_idx,
                                     t_node_feats, edge_attr,
                                     t_graph_labels, t_graph_feats,
                                     hidden_state, dos_memory)
        test_time = timeit.default_timer() - test_start_time
        full_forward_pass_time = timeit.default_timer() - start_dataset_time

        # Logs
        logger.info(
            f"{dataset_idx}. Dataset: {dataset_names[dataset_idx]}, "
            f"Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}, Test R2: {test_r2:.4f}, "
            f"Time: {test_time:.4f}, Full Dataset Pass Time: {full_forward_pass_time:.4f}"
        )
        for t_idx, t_name in enumerate(task_names):
            logger.info(
                f"\tTask {t_name} - Test MAE: {test_mae_pt[t_idx]:.4f}, "
                f"Test RMSE: {test_rmse_pt[t_idx]:.4f}, Test R2: {test_r2_pt[t_idx]:.4f}"
            )

        # WandB
        if args.wandb:
            wandb_log = {
                "dataset_test": dataset_names[dataset_idx],
                "test_mae": test_mae,
                "test_rmse": test_rmse,
                "test_r2": test_r2,
                "test_time": test_time,
                "full_dataset_pass_time": full_forward_pass_time,
            }
            for t_idx, t_name in enumerate(task_names):
                wandb_log[f"test_mae/{t_name}"] = float(test_mae_pt[t_idx])
                wandb_log[f"test_rmse/{t_name}"] = float(test_rmse_pt[t_idx])
                wandb_log[f"test_r2/{t_name}"] = float(test_r2_pt[t_idx])
            wandb.log(wandb_log)

        # CSV record
        dataset_performance = {
            "seed": args.seed,
            "model": args.model,
            "DOS": args.use_dos,
            "data_pack_train": args.data_pack_train,
            "data_pack_test": args.data_pack_test,
            "dataset": dataset_names[dataset_idx],
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
            "test_time": test_time,
            "full_dataset_pass_time": full_forward_pass_time,
        }
        for t_idx, t_name in enumerate(task_names):
            dataset_performance[f"test_mae_{t_name}"] = float(test_mae_pt[t_idx])
            dataset_performance[f"test_rmse_{t_name}"] = float(test_rmse_pt[t_idx])
            dataset_performance[f"test_r2_{t_name}"] = float(test_r2_pt[t_idx])

        dataset_performance_all.append(dataset_performance)
        save_or_append_dict_to_csv(dataset_performance, results_filename)

    total_all_dataset_time = timeit.default_timer() - start_all_dataset
    logger.info(f"Total processing time for all dataset: {total_all_dataset_time:.4f}")

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
    nout = args.nout * model_parameter_scaling_factor  # encoder output dim
    data_pack_train = args.data_pack_train
    data_pack_test = args.data_pack_test  # e.g., "pack_test"

    selected_data_pack = data_pack_test
    if selected_data_pack != "pack_1":
        args.data_pack_idx_start = 0
        args.data_pack_idx_end = 1
    data_pack_list_of_list = dataset_pack_dict[selected_data_pack]

    for data_pack_list in data_pack_list_of_list[args.data_pack_idx_start: args.data_pack_idx_end]:
        run_name = f"TEST_{model_name}_{selected_data_pack}_{data_pack_train}_{lr}_{args.max_epoch}_{seed}"
        run_name_train = f"{model_name}_{data_pack_train}_{lr}_{args.max_epoch}_{seed}" + ("" if args.use_dos else "_no_DOS")

        logger.info(f"TEST: Run name train: {run_name_train}")
        logger.info(f"TEST: Run name: {run_name}")

        set_random(seed)

        log_filename = f"{DATA_PATH}/output/log/{run_name}_log.txt"
        init_logger(log_filename)
        logger.info(f"Starting {run_name}")

        results_filename = f"{DATA_PATH}/output/results/tgf_Multitask_results_regression_TEST.csv"

        dataset_names, data_list = load_multiple_datasets_modified(data_pack_list)
        num_datasets = len(data_list)
        print(f"model depth is = {args.model_layers_scaling_factor}")
        # Splits
        num_snapshots = [data_list[i]["time_length"] for i in range(num_datasets)]
        val_splits = [math.ceil(num_snapshots[i] * val_ratio) for i in range(num_datasets)]
        test_splits = [math.ceil(num_snapshots[i] * test_ratio) for i in range(num_datasets)]
        train_snapshots = [list(range(0, num_snapshots[i] - val_splits[i] - test_splits[i])) for i in range(num_datasets)]
        val_snapshots = [list(range(num_snapshots[i] - val_splits[i] - test_splits[i], num_snapshots[i] - test_splits[i])) for i in range(num_datasets)]
        test_snapshots = [list(range(num_snapshots[i] - test_splits[i], num_snapshots[i])) for i in range(num_datasets)]

        # Models (match train)
        encoder = RecurrentGCN_TGCN_SRC(
            node_features=nfeat,
            hid_dim=nhid,
            out_dim=nout,
            num_layers = args.model_layers_scaling_factor,

        )

        if not args.use_multi_head:
            print("Using a single decoder with multi-output heads")
            decoder = Decoder(in_channels=nout + num_graph_level_feats,
                              hids_size=[nhid],
                              drop_out=[args.dropout],
                              out_channels=len(supported_properties),
                              nlayers=[args.model_layers_scaling_factor],
                              )
        else:
            print("Using MultiHeadDecoder (one head per task)")
            decoder = MultiHeadDecoder(in_channels=nout + num_graph_level_feats,
                                       hids_size=[nhid],
                                       drop_out=[args.dropout],
                                       out_channels=1,
                                       heads_num=len(supported_properties),
                                       nlayers=[args.model_layers_scaling_factor],
                                       )

        attention_dos = AttentionDOS(input_dim=20, memory_size=args.dos_memory_size).to(args.device)
        dos_memory = deque([torch.zeros(20).to(args.device) for _ in range(args.dos_memory_size)],
                           maxlen=args.dos_memory_size)

        if wandb_activate:
            wandb.init(
                entity=args.entity,
                settings=wandb.Settings(start_method='fork'),
                project=args.project,
                name=f"{run_name}",
                tags=["TEST", str(model_name), str(data_pack_train), str(selected_data_pack)],
                config=args,
            )

        # Load trained weights and test
        ckpt_path = f"{DATA_PATH}/output/ckpts/{run_name_train}.pth"
        dataset_performance_all = test_a_trained_model(
            args, encoder, attention_dos, decoder, logger,
            dataset_names, data_list, train_snapshots, val_snapshots, test_snapshots,
            num_datasets, dos_memory,
            ckpt_path=ckpt_path, results_filename=results_filename
        )

        if wandb_activate:
            wandb.finish()


if __name__ == '__main__':
    main()
