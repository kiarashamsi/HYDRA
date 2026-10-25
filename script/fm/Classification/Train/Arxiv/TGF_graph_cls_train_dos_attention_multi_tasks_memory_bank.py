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
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_percentage_error, mean_squared_error
import random
import wandb
import yaml
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from script.nn.models.AttentionDOS import AttentionDOS
from script.nn.models.GraphDecoder import MLP_hardcoded
from script.nn.models.Decoder import Decoder, MultiHeadDecoder
from script.nn.models.TGCN import RecurrentGCN_TGCN, RecurrentGCN_TGCN_one_layer, RecurrentGCN_TGCN_one_layer_ssm, \
    RecurrentGCN_TGCN_with_attention_conv, RecurrentGCN_TGCN_GRU_with_memory
from script.utils.tg_utils import readout_function
from script.utils.tgf_param import *
from script.utils.config import args
from script.utils.util import set_random, logger, init_logger, save_or_append_dict_to_csv, mkdirs
from script.utils.early_stopping import EarlyStopping
from script.utils.data_util import load_multiple_datasets, prepare, multi_datasets_attributes_loading, \
    extra_dataset_attributes_loading, load_multiple_datasets_modified
from script.configs.data_spec import dataset_pack_dict, DATA_PATH, SAVED_MODEL_PATH
from script.utils.data_util import load_DOS_snapshot_feature
# from gtda.homology import VietorisRipsPersistence
# from gtda.diagrams import PersistenceEntropy
from scipy.spatial.distance import pdist, squareform

# Load multi property loss schema
loss_comb_method = args.loss_comb_method  # New argument to choose the loss combination method
loss_weights = torch.ones(len(supported_properties)).to(args.device)  # Default weights (if needed)
# Uncertainty if method needs it
if loss_comb_method == MULTI_LOSS_UNCERTAINTY_BASED:
    uncertainties = torch.nn.Parameter(torch.zeros(len(supported_properties)), requires_grad=True).to(
        args.device)


def compute_performance_multi_task(tasks_labels, tasks_preds, metrics=[roc_auc_score, average_precision_score]):
    number_task = tasks_labels.shape[-1]

    all_results = []
    for metric_measurement in metrics:
        results_all_tasks = []
        for task_idx in range(number_task):
            results_all_tasks.append(metric_measurement(tasks_labels[:, task_idx], tasks_preds[:, task_idx]))
        all_results.append(results_all_tasks)
    return all_results


# === Time Positional Encoding ===
def positional_encoding(t, d_model=16):
    pe = torch.zeros(d_model)
    for i in range(0, d_model, 2):
        pe[i] = math.sin(t / (10000 ** ((2 * i) / d_model)))
        if i + 1 < d_model:
            pe[i + 1] = math.cos(t / (10000 ** ((2 * i) / d_model)))
    return pe


# === TDA Snapshot Feature Computation ===
# def compute_tda_features_from_adj_matrix(adj_matrix, max_dim=1):
#     dist_matrix = squareform(pdist(adj_matrix, metric='euclidean'))
#     VR = VietorisRipsPersistence(homology_dimensions=range(max_dim + 1))
#     diagrams = VR.fit_transform([dist_matrix])
#     PE = PersistenceEntropy()
#     entropy = PE.fit_transform(diagrams)
#     return entropy[0]  # 1D vector
#
#
# def fast_homology_feature_from_graph(edge_index, num_nodes, max_dim=0):
#     deg_vector = np.bincount(edge_index[0].cpu().numpy(), minlength=num_nodes)
#     point_cloud = deg_vector.reshape(-1, 1).astype(np.float64)
#
#     VR = VietorisRipsPersistence(homology_dimensions=[dim for dim in range(max_dim + 1)])
#     diagrams = VR.fit_transform([point_cloud])
#
#     PE = PersistenceEntropy()
#     entropy_vector = PE.fit_transform(diagrams)
#
#     return entropy_vector[0]  # e.g., shape (1,) or (2,) depending on max_dim


import numpy as np
# import networkx as nx
# import kmapper as km
# from sklearn.decomposition import PCA
# from sklearn.cluster import DBSCAN
# from sklearn.preprocessing import StandardScaler


def calculate_loss(tg_pred, tg_label, criterion_bc, criterion_reg):
    """Calculate the loss based on the combination method"""
    # Binary Cross Entropy per property (task)
    individual_losses = []
    if len(supported_properties) != len(supported_properties_type):
        raise ValueError(
            "The lengths of 'supported_properties' and 'supported_properties_type' do not match. Please check the input.")
    for i in range(len(supported_properties_type)):
        if supported_properties_type[i] == "BC":
            individual_losses.append(criterion_bc(tg_pred[i], tg_label[i]))
        elif supported_properties_type[i] == "REG":
            individual_losses.append(criterion_reg(tg_pred[i], tg_label[i]))

    if loss_comb_method == MULTI_LOSS_WEIGHTED_SUM:
        # Combine using weighted sum
        combined_loss = sum(loss_weights[i] * individual_losses[i] for i in range(len(individual_losses)))

    elif loss_comb_method == MULTI_LOSS_DYNAMIC_BALANCING:
        # Example: loss weights dynamically adjusted by task difficulty
        task_difficulties = [loss.item() for loss in individual_losses]
        task_weights = 1.0 / (torch.tensor(task_difficulties) + 1e-6)  # Normalize inverse difficulties
        combined_loss = sum(task_weights[i] * individual_losses[i] for i in range(len(individual_losses)))

    elif loss_comb_method == MULTI_LOSS_UNCERTAINTY_BASED:
        # Uncertainty-based weighting (as per Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses")
        combined_loss = sum(
            (1 / (2 * torch.exp(uncertainties[i]))) * individual_losses[i] + uncertainties[i]
            for i in range(len(individual_losses))
        )
    elif loss_comb_method == MULTI_LOSS_SIMPLE_SUM:
        # Default: simple sum
        combined_loss = sum(individual_losses)

    else:
        # Multi label pass to loss function

        combined_loss = criterion_bc(tg_pred, tg_label)

    return combined_loss


def tg_test(args, encoder, attention_dos, decoder,
            data_list, test_snapshots, dataset_idx, t_node_feats, edge_attr, t_graph_labels, t_graph_feats,
            hidden_state, memory_bank , dos_memory):
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
            embeddings, hidden_state, memory_bank = encoder(x, edge_index, edge_attr, hidden_state, memory_bank)

            hidden_state = hidden_state.detach()
            memory_bank = memory_bank.detach()

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

    bc_label_list = []
    bc_pred_list = []
    reg_label_list = []
    reg_pred_list = []

    for task_idx, task_type in enumerate(supported_properties_type):
        labels = flat_per_data_per_snap_labels[:, task_idx]
        preds = flat_per_data_per_snap_preds[:, task_idx]

        if task_type == "BC":
            bc_label_list.append(labels)
            bc_pred_list.append(preds)
        elif task_type == "REG":
            reg_label_list.append(labels)
            reg_pred_list.append(preds)

    # Stack to form arrays of shape [num_samples, num_bc_tasks] and [num_samples, num_reg_tasks]
    bc_labels = np.stack(bc_label_list, axis=1) if bc_label_list else None
    bc_preds = np.stack(bc_pred_list, axis=1) if bc_pred_list else None
    reg_labels = np.stack(reg_label_list, axis=1) if reg_label_list else np.array([[0]])
    reg_preds = np.stack(reg_pred_list, axis=1) if reg_pred_list else np.array([[0]])

    test_auc = roc_auc_score(bc_labels, bc_preds)
    test_ap = average_precision_score(bc_labels, bc_preds)

    test_mse = mean_squared_error(reg_labels, reg_preds)

    test_auc_per_task = roc_auc_score(bc_labels, bc_preds, average=None)
    test_ap_per_task = average_precision_score(bc_labels, bc_preds,
                                               average=None)

    test_mse_per_task = np.array([[mean_squared_error(reg_labels[:, i], reg_preds[:, i]) for i in
                                   range(reg_labels.shape[1])]])

    return test_auc, test_ap, hidden_state, test_auc_per_task, test_ap_per_task, test_mse, test_mse_per_task


def train_val_test(args, encoder, attention_dos, decoder, criterion_bc, criterion_reg, optimizer, logger,
                   dataset_names, data_list, train_snapshots, val_snapshots, test_snapshots,
                   num_datasets,
                   dos_memory,
                   num_snapshots,
                   early_stopping=None,
                   results_filename=f"{DATA_PATH}/output/results/tgf_results.csv"):
    """
    Train the temporal graph classification foundation model.
    """
    encoder = encoder.to(args.device)
    decoder = decoder.to(args.device)
    attention_dos = attention_dos.to(args.device)

    start_train = timeit.default_timer()

    best_epoch = 0
    # Set memory size
    epoch_losses, epoch_times = [], []
    epoch_train_auc, epoch_train_ap, epoch_train_mse, epoch_train_auc_per_task, epoch_train_mse_per_task, epoch_train_ap_per_task = [], [], [], [], [], []
    epoch_val_auc, epoch_val_ap, epoch_val_mse, epoch_val_auc_per_task, epoch_val_ap_per_task, epoch_val_mse_per_task, epoch_val_time = [], [], [], [], [], [], []
    epoch_test_auc, epoch_test_ap, epoch_test_mse, epoch_test_auc_per_task, epoch_test_ap_per_task, epoch_test_mse_per_task, epoch_test_time = [], [], [], [], [], [], []

    for epoch in range(args.max_epoch):
        logger.info(f"Epoch: {epoch}")
        start_epoch = timeit.default_timer()

        encoder.train()
        decoder.train()
        attention_dos.train()

        dataset_losses, dataset_times = [], []
        dataset_train_aucs, dataset_train_aps, dataset_train_mses, dataset_train_aucs_per_task, dataset_train_aps_per_task, dataset_train_mses_per_task = [], [], [], [], [], []
        dataset_val_aucs, dataset_val_aps, dataset_val_mses, dataset_val_aucs_per_task, dataset_val_aps_per_task, dataset_val_mses_per_task = [], [], [], [], [], []
        dataset_val_times = []
        dataset_test_aucs, dataset_test_aps, dataset_test_mses, dataset_test_aucs_per_task, dataset_test_aps_per_task, dataset_test_mses_per_task = [], [], [], [], [], []
        dataset_test_times = []

        # shuffle the datasets
        dataset_rnd_order = random.sample(range(num_datasets), num_datasets)
        for dataset_idx in dataset_rnd_order:
            encoder.train()
            decoder.train()
            attention_dos.train()

            start_dataset_time = timeit.default_timer()
            logger.info(f"Epoch: {epoch}, Dataset: {dataset_idx}")

            # load dataset features
            # t_graph_labels, t_graph_feat = extra_dataset_attributes_loading(args, dataset_names[dataset_idx])  # TODO: we should not load dataset each time!!!

            # to make sure that the node features are also loaded
            assert args.node_feats, "Node features should be loaded for each dataset!"

            # TODO: "t_graph_feats" & "t_node_feats" are currently the same features!!!
            t_graph_labels, t_graph_feats, t_node_feats = extra_dataset_attributes_loading(args,
                                                                                           dataset_names[dataset_idx],
                                                                                           multi_property=True)

            # load DOS features for graph snapshots
            # TODO: here we are replacing the features not concat to the previous features
            if args.use_dos:
                tg_dos_feat = load_DOS_snapshot_feature(dataset_names[dataset_idx], args=args)
                t_graph_feats = tg_dos_feat  # to use DOS features for graph snapshot features

            # Un comment for TDA representation
            # tg_tda_feat = load_TDA_snapshot_feature(dataset_names[dataset_idx], data_list, num_snapshots[dataset_idx], dataset_idx, args,
            #                                         t_node_feats)

            # to save labels and predictions per snapshot
            per_dataset_per_snap_labels, per_dataset_per_snap_preds = [], []
            per_dataset_per_snap_losses = []
            hidden_state, edge_attr, memory_bank = None, None, None

            for snapshot_idx in train_snapshots[dataset_idx]:
                # ======================================== # level: snapshot
                optimizer.zero_grad()

                # prepare edge_index and node features
                edge_index = prepare(data_list[dataset_idx], snapshot_idx, args)
                # x = torch.from_numpy(t_node_feats[snapshot_idx]).to(torch.float32).to(args.device)
                x = torch.from_numpy(np.array(t_node_feats[snapshot_idx], dtype=np.float32)).to(args.device)

                # Time Positional Embedding
                # print("Generating Time Embedding:")
                # time_pos_embed = positional_encoding(snapshot_idx, d_model=16).to(args.device)
                # print("Time Embedding : {}".format(time_pos_embed))

                # # Check which rows are not all zeros
                # non_zero_rows = (x.abs().sum(dim=1) != 0).sum()
                # print("Number of non zeros: {}".format(non_zero_rows))

                # forward pass


                # Convert memory buffer to tensor (batch_size = 1)
                dos_memory_tensor = torch.stack(list(dos_memory)).unsqueeze(0)  # Shape: (1, memory_size, feature_dim)

                # Get the current DOS vector
                current_dos = torch.tensor(t_graph_feats[snapshot_idx], dtype=torch.float32).to(args.device).unsqueeze(
                    0)  # Shape: (1, feature_dim)

                # Compute Attention-Based DOS
                tg_dos_feat = attention_dos(dos_memory_tensor, current_dos).squeeze(0)  # Shape: (feature_dim,)

                # Update memory (deque ensures fixed size)
                dos_memory.append(current_dos.squeeze(0))

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

                # tda_feat_tensor = torch.tensor(tg_tda_feat[snapshot_idx], dtype=torch.float32).to(args.device).unsqueeze(
                #     0)

                if args.concat_att_dos_to_original:
                    tg_agg_feat = torch.cat((tg_embeddings, tg_dos_feat, current_dos.squeeze(0)), dim=0)
                    # tg_agg_feat = torch.cat(
                    #     (tg_embeddings, tg_dos_feat, current_dos.squeeze(0), tda_feat_tensor.squeeze(0), time_pos_embed), dim=0)

                else:
                    tg_agg_feat = torch.cat((tg_embeddings, tg_dos_feat), dim=0)

                # graph prediction
                # tg_label = t_graph_labels[snapshot_idx].float().view(1, )
                tg_label = t_graph_labels[snapshot_idx, :].float()
                tg_pred = decoder(tg_agg_feat.view(1, tg_agg_feat.size()[0])).squeeze(0)

                # loss
                # Combine the losses based on method
                loss = calculate_loss(tg_pred, tg_label, criterion_bc, criterion_reg)

                # loss = criterion(tg_pred, tg_label)
                loss.backward()
                optimizer.step()
                per_dataset_per_snap_losses.append(loss.item())

                # for later evaluation
                per_dataset_per_snap_labels.append(tg_label)
                per_dataset_per_snap_preds.append(tg_pred)

            # ======================================== # level: dataset
            # all snapshots in one dataset are done
            dataset_losses.append(np.mean(per_dataset_per_snap_losses))

            pdps_labels = np.array([tensor.detach().cpu().numpy() for tensor in
                                    per_dataset_per_snap_labels])  # detach, move to CPU and convert to NumPy
            pdps_preds = np.array([tensor.detach().cpu().numpy() for tensor in
                                   per_dataset_per_snap_preds])  # detach, move to CPU and convert to NumPy

            # compute_performance_multi_task(pdps_labels, pdps_preds)  # Added=====
            # Concatenate the lists
            flat_per_data_per_snap_labels = np.array(pdps_labels)
            flat_per_data_per_snap_preds = np.array(pdps_preds)

            bc_label_list = []
            bc_pred_list = []
            reg_label_list = []
            reg_pred_list = []
            BC_task_available = False
            REG_task_avilable = False

            for task_idx, task_type in enumerate(supported_properties_type):
                labels = flat_per_data_per_snap_labels[:, task_idx]
                preds = flat_per_data_per_snap_preds[:, task_idx]

                if task_type == "BC":
                    bc_label_list.append(labels)
                    bc_pred_list.append(preds)
                    BC_task_available = True
                elif task_type == "REG":
                    reg_label_list.append(labels)
                    reg_pred_list.append(preds)
                    REG_task_avilable = True

            # Stack to form arrays of shape [num_samples, num_bc_tasks] and [num_samples, num_reg_tasks]
            bc_labels = np.stack(bc_label_list, axis=1) if bc_label_list else None
            bc_preds = np.stack(bc_pred_list, axis=1) if bc_pred_list else None
            reg_labels = np.stack(reg_label_list, axis=1) if reg_label_list else np.array([[0]])
            reg_preds = np.stack(reg_pred_list, axis=1) if reg_pred_list else np.array([[0]])

            # Average Metrics over all BC tasks
            train_auc = roc_auc_score(bc_labels, bc_preds)
            train_ap = average_precision_score(bc_labels, bc_preds)

            # Per BC task Metric
            train_auc_per_task = roc_auc_score(bc_labels, bc_preds,
                                               average=None)
            train_ap_per_task = average_precision_score(bc_labels, bc_preds,
                                                        average=None)

            dataset_train_aucs.append(train_auc)
            dataset_train_aucs_per_task.append(train_auc_per_task)
            dataset_train_aps.append(train_ap)
            dataset_train_aps_per_task.append(train_ap_per_task)

            # Average Metrics over all REG tasks
            train_mse = mean_squared_error(reg_labels, reg_preds)
            # Per REG task metric
            train_mse_per_task = np.array([[mean_squared_error(reg_labels[:, i], reg_preds[:, i]) for i in
                                            range(reg_labels.shape[1])]])

            dataset_train_mses.append(train_mse)
            dataset_train_mses_per_task.append(train_mse_per_task)

            # validation
            val_start_time = timeit.default_timer()
            val_auc, val_ap, val_hiddenstate, val_auc_per_task, val_ap_per_task, val_mse, val_mse_per_task = tg_test(
                args, encoder, attention_dos,
                decoder, data_list,
                val_snapshots,
                dataset_idx,
                t_node_feats, edge_attr,
                t_graph_labels, t_graph_feats,
                hidden_state, memory_bank, dos_memory)
            dataset_val_times.append(timeit.default_timer() - val_start_time)
            dataset_val_aucs.append(val_auc)
            dataset_val_aucs_per_task.append(val_auc_per_task)
            dataset_val_aps.append(val_ap)
            dataset_val_aps_per_task.append(val_ap_per_task)
            dataset_val_mses.append(val_mse)
            dataset_val_mses_per_task.append(val_mse_per_task)

            # test
            test_start__time = timeit.default_timer()
            test_auc, test_ap, test_hiddenstate, test_auc_per_task, test_ap_per_task, test_mse, test_mse_per_task = tg_test(
                args, encoder,
                attention_dos, decoder,
                data_list,
                test_snapshots,
                dataset_idx,
                t_node_feats, edge_attr,
                t_graph_labels,
                t_graph_feats,
                val_hiddenstate,
                memory_bank,
                dos_memory)
            dataset_test_times.append(timeit.default_timer() - test_start__time)
            dataset_test_aucs.append(test_auc)
            dataset_test_aucs_per_task.append(test_auc_per_task)
            dataset_test_aps.append(test_ap)
            dataset_test_aps_per_task.append(test_ap_per_task)
            dataset_test_mses.append(test_mse)
            dataset_test_mses_per_task.append(test_mse_per_task)

            dataset_times.append(timeit.default_timer() - start_dataset_time)

        # ========================================  # level: epoch
        # average the results for all datasets for one epoch

        avg_epoch_loss = np.mean(dataset_losses)
        epoch_losses.append(avg_epoch_loss)

        # Compute overall averages
        avg_all_dataset_train_auc = np.mean(dataset_train_aucs)
        avg_all_dataset_train_ap = np.mean(dataset_train_aps)
        avg_all_dataset_train_mse = np.mean(dataset_train_mses)
        avg_all_dataset_train_auc_per_task = np.mean(dataset_train_aucs_per_task, axis=0)
        avg_all_dataset_train_ap_per_task = np.mean(dataset_train_aps_per_task, axis=0)
        avg_all_dataset_train_mse_per_task = np.mean(dataset_train_mses_per_task, axis=0)

        epoch_train_auc.append(avg_all_dataset_train_auc)
        epoch_train_auc_per_task.append(avg_all_dataset_train_auc_per_task)
        epoch_train_ap.append(avg_all_dataset_train_ap)
        epoch_train_ap_per_task.append(avg_all_dataset_train_ap_per_task)
        epoch_train_mse.append(avg_all_dataset_train_mse)
        epoch_train_mse_per_task.append(avg_all_dataset_train_mse_per_task)

        avg_all_dataset_val_auc = np.mean(dataset_val_aucs)
        avg_all_dataset_val_ap = np.mean(dataset_val_aps)
        avg_all_dataset_val_mse = np.mean(dataset_val_mses)
        avg_all_dataset_val_auc_per_task = np.mean(dataset_val_aucs_per_task, axis=0)
        avg_all_dataset_val_ap_per_task = np.mean(dataset_val_aps_per_task, axis=0)
        avg_all_dataset_val_mse_per_task = np.mean(dataset_val_mses_per_task, axis=0)
        avg_all_dataset_val_time = np.mean(dataset_val_times)

        epoch_val_auc.append(avg_all_dataset_val_auc)
        epoch_val_auc_per_task.append(avg_all_dataset_val_auc_per_task)
        epoch_val_ap.append(avg_all_dataset_val_ap)
        epoch_val_ap_per_task.append(avg_all_dataset_val_ap_per_task)
        epoch_val_mse.append(avg_all_dataset_val_mse)
        epoch_val_mse_per_task.append(avg_all_dataset_val_mse_per_task)
        epoch_val_time.append(avg_all_dataset_val_time)

        avg_all_dataset_test_auc = np.mean(dataset_test_aucs)
        avg_all_dataset_test_ap = np.mean(dataset_test_aps)
        avg_all_dataset_test_mse = np.mean(dataset_test_mses)
        avg_all_dataset_test_auc_per_task = np.mean(dataset_test_aucs_per_task, axis=0)
        avg_all_dataset_test_ap_per_task = np.mean(dataset_test_aps_per_task, axis=0)
        avg_all_dataset_test_mse_per_task = np.mean(dataset_test_mses_per_task, axis=0)
        avg_all_dataset_test_time = np.mean(dataset_test_times)

        epoch_test_auc.append(avg_all_dataset_test_auc)
        epoch_test_auc_per_task.append(avg_all_dataset_test_auc_per_task)
        epoch_test_ap.append(avg_all_dataset_test_ap)
        epoch_test_ap_per_task.append(avg_all_dataset_test_ap_per_task)
        epoch_test_mse.append(avg_all_dataset_test_mse)
        epoch_test_mse_per_task.append(avg_all_dataset_test_mse_per_task)
        epoch_test_time.append(avg_all_dataset_test_time)

        epoch_times.append(timeit.default_timer() - start_epoch)

        # Log overall metrics
        logger.info(
            f"Epoch: {epoch}, Train loss: {avg_epoch_loss}, Train AUC: {avg_all_dataset_train_auc}, Train AP: {avg_all_dataset_train_ap}, Train MSE: {avg_all_dataset_train_mse}, Time: {np.mean(epoch_times)}")
        logger.info(
            f"\t\tVal AUC: {avg_all_dataset_val_auc}, Val AP: {avg_all_dataset_val_ap}, Val MSE: {avg_all_dataset_val_mse}")
        logger.info(
            f"\t\tTest AUC: {avg_all_dataset_test_auc}, Test AP: {avg_all_dataset_test_ap},  Test MSE: {avg_all_dataset_test_mse}")

        # Log per-task results
        # For BC
        logger.info(" BC Tasks")
        num_tasks = len(avg_all_dataset_train_auc_per_task)
        for task_idx in range(num_tasks):
            logger.info(
                f"\t\tTask {list(supported_properties.keys())[task_idx]} - Train AUC: {avg_all_dataset_train_auc_per_task[task_idx]}, Train AP: {avg_all_dataset_train_ap_per_task[task_idx]}")
            logger.info(
                f"\t\tTask {list(supported_properties.keys())[task_idx]} - Val AUC: {avg_all_dataset_val_auc_per_task[task_idx]}, Val AP: {avg_all_dataset_val_ap_per_task[task_idx]}")
            logger.info(
                f"\t\tTask {list(supported_properties.keys())[task_idx]} - Test AUC: {avg_all_dataset_test_auc_per_task[task_idx]}, Test AP: {avg_all_dataset_test_ap_per_task[task_idx]}")

        if REG_task_avilable:
            logger.info("Reg Tasks")
            num_tasks_reg = len(avg_all_dataset_train_mse_per_task)
            for task_idx in range(num_tasks_reg):
                logger.info(
                    f"\t\tTask {list(supported_properties.keys())[num_tasks + task_idx]} - Train MSE: {avg_all_dataset_train_mse_per_task[task_idx]}")
                logger.info(
                    f"\t\tTask {list(supported_properties.keys())[num_tasks + task_idx]} - Val MSE: {avg_all_dataset_val_mse_per_task[task_idx]}")
                logger.info(
                    f"\t\tTask {list(supported_properties.keys())[num_tasks + task_idx]} - Test MSE: {avg_all_dataset_test_mse_per_task[task_idx]}")

        # Log per-task metrics in WandB
        if args.wandb:
            wandb_log_dict = {
                "epoch": epoch,
                "train_loss": avg_epoch_loss,
                "train_auc": avg_all_dataset_train_auc,
                "train_ap": avg_all_dataset_train_ap,
                "train_mse": avg_all_dataset_train_mse,
                "val_auc": avg_all_dataset_val_auc,
                "val_ap": avg_all_dataset_val_ap,
                "val_mse": avg_all_dataset_val_mse,
                "test_auc": avg_all_dataset_test_auc,
                "test_ap": avg_all_dataset_test_ap,
                "test_mse": avg_all_dataset_test_mse,
            }

            # Add per-task metrics to WandB logging
            for task_idx in range(num_tasks):
                wandb_log_dict[f"train_auc_task_{list(supported_properties.keys())[task_idx]}"] = \
                    avg_all_dataset_train_auc_per_task[task_idx]
                wandb_log_dict[f"train_ap_task_{list(supported_properties.keys())[task_idx]}"] = \
                    avg_all_dataset_train_ap_per_task[task_idx]
                wandb_log_dict[f"val_auc_task_{list(supported_properties.keys())[task_idx]}"] = \
                    avg_all_dataset_val_auc_per_task[task_idx]
                wandb_log_dict[f"val_ap_task_{list(supported_properties.keys())[task_idx]}"] = \
                    avg_all_dataset_val_ap_per_task[task_idx]
                wandb_log_dict[f"test_auc_task_{list(supported_properties.keys())[task_idx]}"] = \
                    avg_all_dataset_test_auc_per_task[task_idx]
                wandb_log_dict[f"test_ap_task_{list(supported_properties.keys())[task_idx]}"] = \
                    avg_all_dataset_test_ap_per_task[task_idx]

            if REG_task_avilable:
                # For REG Tasks
                for task_idx in range(num_tasks_reg):
                    wandb_log_dict[f"train_mse_task_{list(supported_properties.keys())[num_tasks + task_idx]}"] = \
                        avg_all_dataset_train_mse_per_task[task_idx]

                    wandb_log_dict[f"val_mse_task_{list(supported_properties.keys())[num_tasks + task_idx]}"] = \
                        avg_all_dataset_val_mse_per_task[task_idx]

                    wandb_log_dict[f"test_mse_task_{list(supported_properties.keys())[num_tasks + task_idx]}"] = \
                        avg_all_dataset_test_mse_per_task[task_idx]

            wandb.log(wandb_log_dict)

        # early stopping
        val_metric_indicator = [
            ("val_auc", avg_all_dataset_val_auc, True),  # higher is better
            # ("val_ap", avg_all_dataset_val_ap, True),  # we can also use AP (or both) for early stopping
        ]
        early_stop, best_epoch = early_stopping.step(val_metric_indicator, epoch,
                                                     {"encoder": encoder.state_dict(), "decoder": decoder.state_dict(),
                                                      "attention_dos": attention_dos.state_dict()})

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
        "train_mse": epoch_train_mse[best_epoch],
        "val_auc": epoch_val_auc[best_epoch],
        "val_ap": epoch_val_ap[best_epoch],
        "val_mse": epoch_val_mse[best_epoch],
        "test_auc": epoch_test_auc[best_epoch],
        "test_ap": epoch_test_ap[best_epoch],
        "test_mse": epoch_test_mse[best_epoch],
        "avg_epoch_time": np.mean(epoch_times),
        "best_epoch_time": epoch_times[best_epoch],  # time for the best epoch
        "best_epoch_val_time": epoch_val_time[best_epoch],  # validation time for the best epoch
        "best_epoch_test_time": epoch_test_time[best_epoch],  # test time for the best epoch
    }

    # Log per-task metrics
    num_tasks = len(supported_properties)
    for task_idx in range(num_tasks):
        task_name = list(supported_properties.keys())[task_idx]  # Get task name from the reference list

        # Add per-task performance for the best epoch
        best_epoch_performance[f"train_auc_{task_name}"] = epoch_train_auc_per_task[best_epoch][task_idx]
        best_epoch_performance[f"train_ap_{task_name}"] = epoch_train_ap_per_task[best_epoch][task_idx]
        best_epoch_performance[f"val_auc_{task_name}"] = epoch_val_auc_per_task[best_epoch][task_idx]
        best_epoch_performance[f"val_ap_{task_name}"] = epoch_val_ap_per_task[best_epoch][task_idx]
        best_epoch_performance[f"test_auc_{task_name}"] = epoch_test_auc_per_task[best_epoch][task_idx]
        best_epoch_performance[f"test_ap_{task_name}"] = epoch_test_ap_per_task[best_epoch][task_idx]

    if REG_task_avilable:
        for task_idx in range(num_tasks_reg):
            task_name = list(supported_properties.keys())[num_tasks + task_idx]  # Get task name from the reference list

            # Add per-task performance for the best epoch
            best_epoch_performance[f"train_mse_{task_name}"] = epoch_train_mse_per_task[best_epoch][task_idx]

            best_epoch_performance[f"val_mse_{task_name}"] = epoch_val_mse_per_task[best_epoch][task_idx]

            best_epoch_performance[f"test_mse_{task_name}"] = epoch_test_mse_per_task[best_epoch][task_idx]

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
    nout = args.nout * model_parameter_scaling_factor  # NOTE: output dimension of the encoder

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

        mkdirs(f"{DATA_PATH}/output/ckpts")
        mkdirs(f"{DATA_PATH}/output/results")

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
        encoder = RecurrentGCN_TGCN_GRU_with_memory(
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

        # Add AttentionDOS to the optimizer
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()) + list(attention_dos.parameters()),
            lr=lr,
        )

        # define losss
        criterion_bc = torch.nn.BCEWithLogitsLoss()
        criterion_reg = torch.nn.MSELoss()

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
                    # ",".join(data_pack_list),
                ],
                config=args,
            )

        numel_encoder = sum([p.numel() for p in encoder.parameters()])
        numel_decoder = sum([p.numel() for p in decoder.parameters()])
        numel_attention_dos = sum([p.numel() for p in attention_dos.parameters()])

        logger.info(f"Number of parameters in encoder: {numel_encoder}")
        logger.info(f"Number of parameters in decoder: {numel_decoder}")
        logger.info(f"Number of parameters in attention_dos: {numel_attention_dos}")
        logger.info(f"Total Model Parameters: {numel_encoder + numel_decoder + numel_attention_dos}")

        # main training procedure
        best_epoch_performance = train_val_test(args, encoder, attention_dos, decoder, criterion_bc, criterion_reg,
                                                optimizer, logger,
                                                dataset_names, data_list, train_snapshots, val_snapshots,
                                                test_snapshots,
                                                num_datasets,
                                                dos_memory,
                                                num_snapshots,
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
