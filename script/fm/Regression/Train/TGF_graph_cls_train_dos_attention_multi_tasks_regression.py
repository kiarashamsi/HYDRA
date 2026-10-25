"""
Assumption:
    Train a temporal graph regression foundation model with multiple regression tasks.
    The foundation model is trained to predict graph-level properties (all REG).
=======
Date:
    2025-08-18

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
from script.nn.models.AttentionDOS import AttentionDOS
from script.nn.models.GraphDecoder import MLP_hardcoded
from script.nn.models.Decoder import Decoder, MultiHeadDecoder
from script.nn.models.TGCN import RecurrentGCN_TGCN_GRU_with_memory, RecurrentGCN_TGCN_one_layer
from script.utils.tg_utils import readout_function
from script.utils.tgf_param import *
from script.utils.config import args
from script.utils.util import set_random, logger, init_logger, save_or_append_dict_to_csv, mkdirs
from script.utils.early_stopping import EarlyStopping
from script.utils.data_util import load_multiple_datasets, prepare, multi_datasets_attributes_loading, \
    extra_dataset_attributes_loading, load_multiple_datasets_modified
from script.configs.data_spec import dataset_pack_dict, DATA_PATH, SAVED_MODEL_PATH
from script.utils.data_util import load_DOS_snapshot_feature
from scipy.spatial.distance import pdist, squareform

# =========================
# Multi-task loss settings
# =========================
loss_comb_method = args.loss_comb_method  # supports: MULTI_LOSS_WEIGHTED_SUM, MULTI_LOSS_DYNAMIC_BALANCING, MULTI_LOSS_UNCERTAINTY_BASED, MULTI_LOSS_SIMPLE_SUM
num_tasks = len(supported_properties)  # all tasks are REG now
loss_weights = torch.ones(num_tasks).to(args.device)

# Uncertainty-based weighting (Kendall et al.)
if loss_comb_method == MULTI_LOSS_UNCERTAINTY_BASED:
    uncertainties = torch.nn.Parameter(torch.zeros(num_tasks), requires_grad=True).to(args.device)

# =============== Helpers ===============

def positional_encoding(t, d_model=16):
    pe = torch.zeros(d_model)
    for i in range(0, d_model, 2):
        pe[i] = math.sin(t / (10000 ** ((2 * i) / d_model)))
        if i + 1 < d_model:
            pe[i + 1] = math.cos(t / (10000 ** ((2 * i) / d_model)))
    return pe

def compute_regression_metrics_np(labels_2d, preds_2d):
    """
    labels_2d, preds_2d: shape (N, T) for N samples and T tasks
    Returns:
        avg_mae, avg_rmse, avg_r2, per_task_mae (T,), per_task_rmse (T,), per_task_r2 (T,)
    """
    T = labels_2d.shape[1]
    mae_list, rmse_list, r2_list = [], [], []
    for i in range(T):
        y_true = labels_2d[:, i]
        y_pred = preds_2d[:, i]
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        # r2_score can be undefined if variance=0; guard it
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

def calculate_loss(log_rate_pred, target_counts, criterion_mse):
    """
    Per-task loss with optional weighting schemes.
    log_rate_pred: tensor [T]
    target_counts: tensor [T] (non-negative)
    criterion_mse: PoissonNLLLoss(reduction='none', log_input=True)
    """
    # per-task losses (shape [T])
    per_task_losses = criterion_mse(log_rate_pred, target_counts)

    if loss_comb_method == MULTI_LOSS_WEIGHTED_SUM:
        combined_loss = (loss_weights * per_task_losses).sum()

    elif loss_comb_method == MULTI_LOSS_DYNAMIC_BALANCING:
        # inverse of current loss magnitude (detach to avoid gradients through weights)
        difficulties = per_task_losses.detach() + 1e-6
        task_weights = 1.0 / difficulties
        task_weights = task_weights / task_weights.sum() * len(per_task_losses)  # normalize
        combined_loss = (task_weights * per_task_losses).sum()

    elif loss_comb_method == MULTI_LOSS_UNCERTAINTY_BASED:
        # (1 / (2*exp(sigma))) * L_i + sigma, summed over tasks
        combined_loss = sum((1.0 / (2.0 * torch.exp(uncertainties[i]))) * per_task_losses[i] + uncertainties[i]
                            for i in range(len(per_task_losses)))

    elif loss_comb_method == MULTI_LOSS_SIMPLE_SUM:
        combined_loss = per_task_losses.sum()

    else:
        # fallback: simple mean
        combined_loss = per_task_losses.mean()

    return combined_loss

def tg_test(args, encoder, attention_dos, decoder,
            data_list, test_snapshots, dataset_idx, t_node_feats, edge_attr, t_graph_labels, t_graph_feats,
            hidden_state, dos_memory):
    """
    Evaluate on a split. Returns averaged metrics + per-task metrics and the last hidden_state.
    """
    encoder.eval()
    decoder.eval()
    attention_dos.eval()

    all_labels, all_preds = [], []

    for snapshot_idx in test_snapshots[dataset_idx]:
        with torch.no_grad():
            x = torch.from_numpy(np.array(t_node_feats[snapshot_idx], dtype=np.float32)).to(args.device)
            edge_index = prepare(data_list[dataset_idx], snapshot_idx, args)[:3]

            embeddings, hidden_state = encoder(x, edge_index, edge_attr, hidden_state)
            hidden_state = hidden_state.detach()

            # DOS memory
            dos_memory_tensor = torch.stack(list(dos_memory)).unsqueeze(0)  # (1, mem, feat)
            current_dos = torch.tensor(t_graph_feats[snapshot_idx], dtype=torch.float32).to(args.device).unsqueeze(0)
            tg_dos_feat = attention_dos(dos_memory_tensor, current_dos).squeeze(0)
            dos_memory.append(current_dos.squeeze(0))

            # Graph readout
            if args.set_green_readout_TGCN:
                non_zero_rows = (x.abs().sum(dim=1) != 0).sum()
                tg_embeddings = readout_function(embeddings[0:non_zero_rows], args.graph_agg)
            else:
                tg_embeddings = readout_function(embeddings, args.graph_agg)

            if args.set_green_hidden_layer_TGCN:
                non_zero_rows = (x.abs().sum(dim=1) != 0).sum()
                mask = torch.zeros_like(hidden_state)
                mask[:non_zero_rows, :] = hidden_state[:non_zero_rows, :]
                hidden_state = mask

            if args.concat_att_dos_to_original:
                tg_agg_feat = torch.cat((tg_embeddings, tg_dos_feat, current_dos.squeeze(0)), dim=0)
            else:
                tg_agg_feat = torch.cat((tg_embeddings, tg_dos_feat), dim=0)

            # Prediction: decoder outputs log-rate for Poisson loss
            tg_label = t_graph_labels[snapshot_idx, :].float()                   # [T]
            log_rate = decoder(tg_agg_feat.view(1, tg_agg_feat.size(0))).squeeze(0)  # [T]
            rate_pred = torch.exp(log_rate)                                      # positive rate

            all_labels.append(tg_label.detach().cpu().numpy())
            all_preds.append(rate_pred.detach().cpu().numpy())

    labels_np = np.array(all_labels)  # (N, T)
    preds_np  = np.array(all_preds)   # (N, T)

    avg_mae, avg_rmse, avg_r2, per_task_mae, per_task_rmse, per_task_r2 = compute_regression_metrics_np(labels_np, preds_np)
    return (avg_mae, avg_rmse, avg_r2,
            per_task_mae, per_task_rmse, per_task_r2,
            hidden_state)

def train_val_test(args, encoder, attention_dos, decoder, criterion_mse, optimizer, logger,
                   dataset_names, data_list, train_snapshots, val_snapshots, test_snapshots,
                   num_datasets, dos_memory, num_snapshots,
                   early_stopping=None,
                   results_filename=f"{DATA_PATH}/output/results/tgf_results.csv"):
    """
    Train the temporal graph regression foundation model.
    """
    encoder = encoder.to(args.device)
    decoder = decoder.to(args.device)
    attention_dos = attention_dos.to(args.device)

    # Early stop metric selection
    early_stop_metric_name = getattr(args, "early_stop_metric", "rmse").lower()  # "mae" | "rmse" | "r2"
    def higher_is_better(name):
        return True if name == "r2" else False

    start_train = timeit.default_timer()

    best_epoch = 0
    epoch_times = []

    # History containers
    epoch_train_mae = []
    epoch_train_rmse = []
    epoch_train_r2 = []
    epoch_train_mae_per_task = []
    epoch_train_rmse_per_task = []
    epoch_train_r2_per_task = []

    epoch_val_mae = []
    epoch_val_rmse = []
    epoch_val_r2 = []
    epoch_val_mae_per_task = []
    epoch_val_rmse_per_task = []
    epoch_val_r2_per_task = []
    epoch_val_time = []

    epoch_test_mae = []
    epoch_test_rmse = []
    epoch_test_r2 = []
    epoch_test_mae_per_task = []
    epoch_test_rmse_per_task = []
    epoch_test_r2_per_task = []
    epoch_test_time = []

    epoch_losses = []

    for epoch in range(args.max_epoch):
        logger.info(f"Epoch: {epoch}")
        start_epoch = timeit.default_timer()

        encoder.train()
        decoder.train()
        attention_dos.train()

        dataset_losses = []
        dataset_times = []

        dataset_train_mae = []
        dataset_train_rmse = []
        dataset_train_r2 = []
        dataset_train_mae_pt = []
        dataset_train_rmse_pt = []
        dataset_train_r2_pt = []

        dataset_val_mae = []
        dataset_val_rmse = []
        dataset_val_r2 = []
        dataset_val_mae_pt = []
        dataset_val_rmse_pt = []
        dataset_val_r2_pt = []
        dataset_val_times = []

        dataset_test_mae = []
        dataset_test_rmse = []
        dataset_test_r2 = []
        dataset_test_mae_pt = []
        dataset_test_rmse_pt = []
        dataset_test_r2_pt = []
        dataset_test_times = []

        dataset_rnd_order = random.sample(range(num_datasets), num_datasets)

        for dataset_idx in dataset_rnd_order:
            encoder.train()
            decoder.train()
            attention_dos.train()

            start_dataset_time = timeit.default_timer()
            logger.info(f"Epoch: {epoch}, Dataset: {dataset_idx}")

            assert args.node_feats, "Node features should be loaded for each dataset!"

            # Load snapshot labels and features; multi_property=True returns [S, T]
            t_graph_labels, t_graph_feats, t_node_feats = extra_dataset_attributes_loading(
                args, dataset_names[dataset_idx], multi_property=True
            )

            if args.use_dos:
                tg_dos_feat = load_DOS_snapshot_feature(dataset_names[dataset_idx], args=args)
                t_graph_feats = tg_dos_feat  # use DOS features for snapshot features

            per_dataset_labels = []
            per_dataset_preds = []
            per_dataset_losses = []

            hidden_state, edge_attr = None, None

            for snapshot_idx in train_snapshots[dataset_idx]:
                optimizer.zero_grad()

                # Inputs
                edge_index = prepare(data_list[dataset_idx], snapshot_idx, args)
                x = torch.from_numpy(np.array(t_node_feats[snapshot_idx], dtype=np.float32)).to(args.device)

                embeddings, hidden_state = encoder(x, edge_index, edge_attr, hidden_state)
                hidden_state = hidden_state.detach()

                # DOS memory
                dos_memory_tensor = torch.stack(list(dos_memory)).unsqueeze(0)
                current_dos = torch.tensor(t_graph_feats[snapshot_idx], dtype=torch.float32).to(args.device).unsqueeze(0)
                tg_dos_feat = attention_dos(dos_memory_tensor, current_dos).squeeze(0)
                dos_memory.append(current_dos.squeeze(0))

                # Readout
                if args.set_green_readout_TGCN:
                    non_zero_rows = (x.abs().sum(dim=1) != 0).sum()
                    tg_embeddings = readout_function(embeddings[0:non_zero_rows], args.graph_agg)
                else:
                    tg_embeddings = readout_function(embeddings, args.graph_agg)

                if args.set_green_hidden_layer_TGCN:
                    non_zero_rows = (x.abs().sum(dim=1) != 0).sum()
                    mask = torch.zeros_like(hidden_state)
                    mask[:non_zero_rows, :] = hidden_state[:non_zero_rows, :]
                    hidden_state = mask

                if args.concat_att_dos_to_original:
                    tg_agg_feat = torch.cat((tg_embeddings, tg_dos_feat, current_dos.squeeze(0)), dim=0)
                else:
                    tg_agg_feat = torch.cat((tg_embeddings, tg_dos_feat), dim=0)

                # Targets and predictions
                tg_label = t_graph_labels[snapshot_idx, :].float().to(args.device)            # [T], non-negative
                log_rate = decoder(tg_agg_feat.view(1, tg_agg_feat.size(0))).squeeze(0)       # [T]
                rate_pred = torch.exp(log_rate)                                               # [T], positive

                # Loss
                loss = calculate_loss(log_rate, tg_label, criterion_mse)
                loss.backward()
                optimizer.step()
                per_dataset_losses.append(loss.item())

                # Collect for train metrics
                per_dataset_labels.append(tg_label.detach().cpu().numpy())
                per_dataset_preds.append(rate_pred.detach().cpu().numpy())

            # === Dataset-level train metrics ===
            labels_np = np.array(per_dataset_labels)  # (N, T)
            preds_np = np.array(per_dataset_preds)    # (N, T)
            tr_mae, tr_rmse, tr_r2, tr_mae_pt, tr_rmse_pt, tr_r2_pt = compute_regression_metrics_np(labels_np, preds_np)

            dataset_losses.append(np.mean(per_dataset_losses))
            dataset_train_mae.append(tr_mae)
            dataset_train_rmse.append(tr_rmse)
            dataset_train_r2.append(tr_r2)
            dataset_train_mae_pt.append(tr_mae_pt)
            dataset_train_rmse_pt.append(tr_rmse_pt)
            dataset_train_r2_pt.append(tr_r2_pt)

            # === Validation ===
            val_start = timeit.default_timer()
            (v_mae, v_rmse, v_r2,
             v_mae_pt, v_rmse_pt, v_r2_pt,
             val_hiddenstate) = tg_test(args, encoder, attention_dos, decoder,
                                        data_list, val_snapshots, dataset_idx, t_node_feats, edge_attr,
                                        t_graph_labels, t_graph_feats, hidden_state, dos_memory)
            dataset_val_times.append(timeit.default_timer() - val_start)
            dataset_val_mae.append(v_mae); dataset_val_rmse.append(v_rmse); dataset_val_r2.append(v_r2)
            dataset_val_mae_pt.append(v_mae_pt); dataset_val_rmse_pt.append(v_rmse_pt); dataset_val_r2_pt.append(v_r2_pt)

            # === Test ===
            test_start = timeit.default_timer()
            (te_mae, te_rmse, te_r2,
             te_mae_pt, te_rmse_pt, te_r2_pt,
             _) = tg_test(args, encoder, attention_dos, decoder,
                          data_list, test_snapshots, dataset_idx, t_node_feats, edge_attr,
                          t_graph_labels, t_graph_feats, val_hiddenstate, dos_memory)
            dataset_test_times.append(timeit.default_timer() - test_start)
            dataset_test_mae.append(te_mae); dataset_test_rmse.append(te_rmse); dataset_test_r2.append(te_r2)
            dataset_test_mae_pt.append(te_mae_pt); dataset_test_rmse_pt.append(te_rmse_pt); dataset_test_r2_pt.append(te_r2_pt)

            dataset_times.append(timeit.default_timer() - start_dataset_time)

        # ================= Epoch aggregation over datasets =================
        avg_epoch_loss = np.mean(dataset_losses)
        epoch_losses.append(avg_epoch_loss)

        # Train
        epoch_train_mae.append(np.mean(dataset_train_mae))
        epoch_train_rmse.append(np.mean(dataset_train_rmse))
        epoch_train_r2.append(np.mean(dataset_train_r2))
        epoch_train_mae_per_task.append(np.mean(np.stack(dataset_train_mae_pt, axis=0), axis=0))
        epoch_train_rmse_per_task.append(np.mean(np.stack(dataset_train_rmse_pt, axis=0), axis=0))
        epoch_train_r2_per_task.append(np.mean(np.stack(dataset_train_r2_pt, axis=0), axis=0))

        # Val
        epoch_val_mae.append(np.mean(dataset_val_mae))
        epoch_val_rmse.append(np.mean(dataset_val_rmse))
        epoch_val_r2.append(np.mean(dataset_val_r2))
        epoch_val_mae_per_task.append(np.mean(np.stack(dataset_val_mae_pt, axis=0), axis=0))
        epoch_val_rmse_per_task.append(np.mean(np.stack(dataset_val_rmse_pt, axis=0), axis=0))
        epoch_val_r2_per_task.append(np.mean(np.stack(dataset_val_r2_pt, axis=0), axis=0))
        epoch_val_time.append(np.mean(dataset_val_times))

        # Test
        epoch_test_mae.append(np.mean(dataset_test_mae))
        epoch_test_rmse.append(np.mean(dataset_test_rmse))
        epoch_test_r2.append(np.mean(dataset_test_r2))
        epoch_test_mae_per_task.append(np.mean(np.stack(dataset_test_mae_pt, axis=0), axis=0))
        epoch_test_rmse_per_task.append(np.mean(np.stack(dataset_test_rmse_pt, axis=0), axis=0))
        epoch_test_r2_per_task.append(np.mean(np.stack(dataset_test_r2_pt, axis=0), axis=0))
        epoch_test_time.append(np.mean(dataset_test_times))

        epoch_times.append(timeit.default_timer() - start_epoch)

        # ================= Logging =================
        logger.info(
            f"Epoch {epoch} | Loss {avg_epoch_loss:.6f} | Train MAE {epoch_train_mae[-1]:.4f} RMSE {epoch_train_rmse[-1]:.4f} R2 {epoch_train_r2[-1]:.4f} | "
            f"Val MAE {epoch_val_mae[-1]:.4f} RMSE {epoch_val_rmse[-1]:.4f} R2 {epoch_val_r2[-1]:.4f} | "
            f"Test MAE {epoch_test_mae[-1]:.4f} RMSE {epoch_test_rmse[-1]:.4f} R2 {epoch_test_r2[-1]:.4f}"
        )

        # Per-task logging
        task_names = list(supported_properties.keys())
        for t_idx, t_name in enumerate(task_names):
            logger.info(
                f"\tTask {t_name} | Train MAE {epoch_train_mae_per_task[-1][t_idx]:.4f} RMSE {epoch_train_rmse_per_task[-1][t_idx]:.4f} R2 {epoch_train_r2_per_task[-1][t_idx]:.4f} | "
                f"Val MAE {epoch_val_mae_per_task[-1][t_idx]:.4f} RMSE {epoch_val_rmse_per_task[-1][t_idx]:.4f} R2 {epoch_val_r2_per_task[-1][t_idx]:.4f} | "
                f"Test MAE {epoch_test_mae_per_task[-1][t_idx]:.4f} RMSE {epoch_test_rmse_per_task[-1][t_idx]:.4f} R2 {epoch_test_r2_per_task[-1][t_idx]:.4f}"
            )

        # WandB (optional)
        if args.wandb:
            wandb_log = {
                "epoch": epoch,
                "train_loss": avg_epoch_loss,
                "train_mae": epoch_train_mae[-1],
                "train_rmse": epoch_train_rmse[-1],
                "train_r2": epoch_train_r2[-1],
                "val_mae": epoch_val_mae[-1],
                "val_rmse": epoch_val_rmse[-1],
                "val_r2": epoch_val_r2[-1],
                "test_mae": epoch_test_mae[-1],
                "test_rmse": epoch_test_rmse[-1],
                "test_r2": epoch_test_r2[-1],
            }
            for t_idx, t_name in enumerate(task_names):
                wandb_log[f"train_mae/{t_name}"] = epoch_train_mae_per_task[-1][t_idx]
                wandb_log[f"train_rmse/{t_name}"] = epoch_train_rmse_per_task[-1][t_idx]
                wandb_log[f"train_r2/{t_name}"] = epoch_train_r2_per_task[-1][t_idx]
                wandb_log[f"val_mae/{t_name}"] = epoch_val_mae_per_task[-1][t_idx]
                wandb_log[f"val_rmse/{t_name}"] = epoch_val_rmse_per_task[-1][t_idx]
                wandb_log[f"val_r2/{t_name}"] = epoch_val_r2_per_task[-1][t_idx]
                wandb_log[f"test_mae/{t_name}"] = epoch_test_mae_per_task[-1][t_idx]
                wandb_log[f"test_rmse/{t_name}"] = epoch_test_rmse_per_task[-1][t_idx]
                wandb_log[f"test_r2/{t_name}"] = epoch_test_r2_per_task[-1][t_idx]
            wandb.log(wandb_log)

        # ================= Early stopping on chosen metric =================
        if early_stop_metric_name == "mae":
            val_metric_value = epoch_val_mae[-1]
        elif early_stop_metric_name == "r2":
            val_metric_value = epoch_val_r2[-1]
        else:  # "rmse" default
            val_metric_value = epoch_val_rmse[-1]

        early_stop, best_epoch = early_stopping.step(
            [(f"val_{early_stop_metric_name}", val_metric_value, higher_is_better(early_stop_metric_name))],
            epoch,
            {"encoder": encoder.state_dict(), "decoder": decoder.state_dict(), "attention_dos": attention_dos.state_dict()},
        )
        if early_stop:
            break

    # ================= Best-epoch summary =================
    best_epoch_performance = {
        "seed": args.seed,
        "model": args.model,
        "data_pack": args.data_pack_train,
        "dataset": ",".join(dataset_names),
        "best_epoch": best_epoch,
        "train_mae": epoch_train_mae[best_epoch],
        "train_rmse": epoch_train_rmse[best_epoch],
        "train_r2": epoch_train_r2[best_epoch],
        "val_mae": epoch_val_mae[best_epoch],
        "val_rmse": epoch_val_rmse[best_epoch],
        "val_r2": epoch_val_r2[best_epoch],
        "test_mae": epoch_test_mae[best_epoch],
        "test_rmse": epoch_test_rmse[best_epoch],
        "test_r2": epoch_test_r2[best_epoch],
        "avg_epoch_time": np.mean(epoch_times),
        "best_epoch_time": epoch_times[best_epoch],
        "best_epoch_val_time": epoch_val_time[best_epoch],
        "best_epoch_test_time": epoch_test_time[best_epoch],
    }

    task_names = list(supported_properties.keys())
    for t_idx, t_name in enumerate(task_names):
        best_epoch_performance[f"train_mae_{t_name}"] = epoch_train_mae_per_task[best_epoch][t_idx]
        best_epoch_performance[f"train_rmse_{t_name}"] = epoch_train_rmse_per_task[best_epoch][t_idx]
        best_epoch_performance[f"train_r2_{t_name}"] = epoch_train_r2_per_task[best_epoch][t_idx]
        best_epoch_performance[f"val_mae_{t_name}"] = epoch_val_mae_per_task[best_epoch][t_idx]
        best_epoch_performance[f"val_rmse_{t_name}"] = epoch_val_rmse_per_task[best_epoch][t_idx]
        best_epoch_performance[f"val_r2_{t_name}"] = epoch_val_r2_per_task[best_epoch][t_idx]
        best_epoch_performance[f"test_mae_{t_name}"] = epoch_test_mae_per_task[best_epoch][t_idx]
        best_epoch_performance[f"test_rmse_{t_name}"] = epoch_test_rmse_per_task[best_epoch][t_idx]
        best_epoch_performance[f"test_r2_{t_name}"] = epoch_test_r2_per_task[best_epoch][t_idx]

    save_or_append_dict_to_csv(best_epoch_performance, results_filename)

    total_train_time = timeit.default_timer() - start_train
    logger.info(f"Total Time: {total_train_time}")
    return best_epoch_performance


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

    selected_data_pack = args.data_pack_train
    if selected_data_pack != "pack_1":
        args.data_pack_idx_start = 0
        args.data_pack_idx_end = 1

    data_pack_list_of_list = dataset_pack_dict[selected_data_pack]

    for data_pack_list in data_pack_list_of_list[args.data_pack_idx_start: args.data_pack_idx_end]:

        if selected_data_pack == "pack_1":
            run_name = f"{model_name}_{selected_data_pack}:[{','.join(data_pack_list)}]_{lr}_{args.max_epoch}_{seed}"
        else:
            print(f"selected_data_pack contains {len(data_pack_list)} datasets, not including the names in the run_name!")
            run_name = f"{model_name}_{selected_data_pack}_{lr}_{args.max_epoch}_{seed}" + ("_no_DOS" if not args.use_dos else "")

        # Reproducibility
        set_random(seed)

        # Logger
        log_filename = f"{DATA_PATH}/output/log/{run_name}_log.txt"
        init_logger(log_filename)
        logger.info(f"Starting {run_name}")

        # IO paths
        mkdirs(f"{DATA_PATH}/output/ckpts")
        mkdirs(f"{DATA_PATH}/output/results")
        results_filename = f"{DATA_PATH}/output/results/tgf_results.csv"
        ckpt_path = f"{DATA_PATH}/output/ckpts/{run_name}.pth"

        dataset_names, data_list = load_multiple_datasets_modified(data_pack_list)
        num_datasets = len(data_list)

        # Splits
        num_snapshots = [data_list[i]["time_length"] for i in range(num_datasets)]
        val_splits = [math.ceil(num_snapshots[i] * val_ratio) for i in range(num_datasets)]
        test_splits = [math.ceil(num_snapshots[i] * test_ratio) for i in range(num_datasets)]

        train_snapshots = [list(range(0, num_snapshots[i] - val_splits[i] - test_splits[i])) for i in range(num_datasets)]
        val_snapshots = [list(range(num_snapshots[i] - val_splits[i] - test_splits[i], num_snapshots[i] - test_splits[i])) for i in range(num_datasets)]
        test_snapshots = [list(range(num_snapshots[i] - test_splits[i], num_snapshots[i])) for i in range(num_datasets)]

        # Encoder
        encoder = RecurrentGCN_TGCN_one_layer(
            node_features=nfeat,
            hid_dim=nhid,
            out_dim=nout,
        )

        # Decoder: output size = num_tasks; output is log-rate for Poisson loss
        if not args.use_multi_head:
            print("Using a single decoder with multi-output heads")
            decoder = Decoder(in_channels=nout + num_graph_level_feats,
                              hids_size=[nhid],
                              drop_out=[args.dropout],
                              out_channels=num_tasks)   # returns [B, T]
        else:
            print("Using MultiHeadDecoder (one head per task)")
            decoder = MultiHeadDecoder(in_channels=nout + num_graph_level_feats,
                                       hids_size=[nhid],
                                       drop_out=[args.dropout],
                                       out_channels=1,
                                       heads_num=num_tasks)      # returns [B, T]

        # Attention-DOS
        attention_dos = AttentionDOS(input_dim=20, memory_size=args.dos_memory_size).to(args.device)
        dos_memory = deque([torch.zeros(20).to(args.device) for _ in range(args.dos_memory_size)],
                           maxlen=args.dos_memory_size)

        # Optimizer (include uncertainty params if used)
        params = list(encoder.parameters()) + list(decoder.parameters()) + list(attention_dos.parameters())
        if loss_comb_method == MULTI_LOSS_UNCERTAINTY_BASED:
            params += [uncertainties]
        optimizer = torch.optim.Adam(params, lr=lr)

        # # === Poisson loss (log_input=True => decoder should output log-rate) ===
        # criterion_poisson = torch.nn.PoissonNLLLoss(log_input=True, full=False, reduction='none')

        # === MSE loss (regression) ===
        criterion_mse = torch.nn.MSELoss(reduction='none')

        early_stopping = EarlyStopping(patience=args.patience, logger=logger, save_model_path=ckpt_path)

        if wandb_activate:
            wandb.init(
                entity=args.entity,
                settings=wandb.Settings(start_method='spawn'),
                project=args.project,
                name=f"{run_name}",
                tags=[str(lr), str(args.max_epoch), str(model_name), str(selected_data_pack)],
                config=args,
            )

        # Model size logs
        numel_encoder = sum([p.numel() for p in encoder.parameters()])
        numel_decoder = sum([p.numel() for p in decoder.parameters()])
        numel_attention_dos = sum([p.numel() for p in attention_dos.parameters()])
        logger.info(f"Number of parameters in encoder: {numel_encoder}")
        logger.info(f"Number of parameters in decoder: {numel_decoder}")
        logger.info(f"Number of parameters in attention_dos: {numel_attention_dos}")
        logger.info(f"Total Model Parameters: {numel_encoder + numel_decoder + numel_attention_dos}")

        # Train / Val / Test
        best_epoch_performance = train_val_test(
            args, encoder, attention_dos, decoder, criterion_mse, optimizer, logger,
            dataset_names, data_list, train_snapshots, val_snapshots, test_snapshots,
            num_datasets, dos_memory, num_snapshots,
            early_stopping=early_stopping, results_filename=results_filename
        )
        logger.info(f"Best epoch performance: {best_epoch_performance}")

        if wandb_activate:
            wandb.finish()


if __name__ == '__main__':
    main()

"""
Command to run:
(NOTE: make sure to be in `script/fm/` directory)

    python train_TGF_graph_cls.py --model TGCN --device cuda:0 --seed 1024 --lr 2e-4 --wandb --node_feats

Optional:
    --early_stop_metric rmse   # or mae / r2
"""
