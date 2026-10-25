import os
import pickle

import numpy as np
import pandas as pd
import networkx as nx
import time
import torch
import hashlib
import sys
from sklearn.preprocessing import MinMaxScaler
import torch

from script.utils.TGS import TGS_Handler
from script.utils.tgf_param import supported_properties
from script.utils.make_edges_orign import mask_edges_det, mask_edges_prd, mask_edges_prd_new_by_marlin
from sklearn.preprocessing import MinMaxScaler
# from script.utils.DOS import graph_to_laplace_dos,normal_util
from script.utils.DOS.subroutines.dos import graph_to_laplace_dos
from script.utils.DOS.util import normal_util

from script.configs.data_spec import DATA_PATH
from script.utils.util import normalize_vector
import os
import pandas as pd
import numpy as np
import ast
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import kmapper as km
import networkx as nx

def mkdirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def multi_datasets_attributes_loading(args, readout_scheme='mean', multi_property=False):
    """
    Load graph features and labels for a list of datasets to train foundation models
    Should be used only when we want graph-pooled features. No node features is returned.
    """
    TG_labels_data = []
    TG_graph_feats_data = []
    TG_node_feats_data = []
    for dataset in args.dataset:
        print("INFO:Loading features and Labels for {} / Multi-property Mode is: {}".format(dataset, multi_property))
        if not args.node_feats:
            TG_labels, graph_feats = extra_dataset_attributes_loading(args, dataset, readout_scheme, multi_property)
            TG_labels_data.append(TG_labels)
            TG_graph_feats_data.append(graph_feats)
            TG_node_feats_data = None
        else:
            TG_labels, graph_feats, node_feats = extra_dataset_attributes_loading(args, dataset, readout_scheme, multi_property)
            TG_labels_data.append(TG_labels)
            TG_graph_feats_data.append(graph_feats)
            TG_node_feats_data.append(node_feats)



    return TG_labels_data, TG_graph_feats_data, TG_node_feats_data

def load_DOS_snapshot_feature(dataset, args=None, **dos_config):
    print("INFO: Generating DOS graph representation")
    # load and process graph-pooled and node-level features
    partial_path = f'{DATA_PATH}/input/'
    data_root = '{}/cached_feat/{}/'.format(partial_path, dataset)
    if not os.path.exists(data_root): os.makedirs(data_root)
    cached_feature_path = "{}/{}_DOS_feature.pkl".format(data_root, dataset)

    DOS_graph_feats = []  # List to store DOS graph features for each snapshot

    if os.path.exists(cached_feature_path):
        print(f"INFO: Loading cached DOS features for {dataset}")
        DOS_graph_feats = normal_util.load_object(cached_feature_path)
        if args:
            if args.normalize_dos:
                DOS_graph_feats = normalize_vector(DOS_graph_feats)
        return DOS_graph_feats
    else:
        print(f"INFO: Generating DOS features for {dataset}")
        edgelist_filename = f'{partial_path}/raw/edgelists/{dataset}_edgelist.txt'
        edgelist_df = pd.read_csv(edgelist_filename)
        uniq_ts_list = np.unique(edgelist_df['snapshot'])
        print(f"Number of unique snapshots: {len(uniq_ts_list)}")
        G_times = []
        for ts in uniq_ts_list:
            ts_edges = edgelist_df.loc[edgelist_df['snapshot'] == ts, ['source', 'destination', 'weight']]
            ts_G = nx.from_pandas_edgelist(ts_edges, source='source', target='destination')

            G_times.append(ts_G)
        DOS_embed = graph_to_laplace_dos(G_times, **dos_config)

        normal_util.save_object(DOS_embed, cached_feature_path)

    if args:
        if args.normalize_dos:
            DOS_embed = normalize_vector(DOS_embed)
    return DOS_embed




def compute_mapper_from_node_features(edge_index, node_features, feature_size=6):
    G = nx.Graph()
    edges = edge_index.cpu().numpy().T.tolist()
    G.add_edges_from(edges)
    G.add_nodes_from(range(node_features.shape[0]))

    lens = PCA(n_components=2).fit_transform(node_features)

    mapper = km.KeplerMapper(verbose=0)
    graph = mapper.map(
        lens,
        node_features,
        clusterer=DBSCAN(eps=0.5, min_samples=3),
        cover=km.Cover(n_cubes=10, perc_overlap=0.3)
    )

    num_nodes = len(graph["nodes"])
    num_edges = sum(len(v) for v in graph["links"].values())
    avg_cluster_size = np.mean([len(v) for v in graph["nodes"].values()]) if graph["nodes"] else 0
    max_cluster_size = max([len(v) for v in graph["nodes"].values()]) if graph["nodes"] else 0
    num_isolated = sum(1 for v in graph["links"].values() if len(v) == 0)
    link_entropy = np.log(1 + num_edges) / (1 + num_nodes) if num_nodes > 0 else 0

    summary_vector = np.array([
        num_nodes,
        num_edges,
        avg_cluster_size,
        max_cluster_size,
        num_isolated,
        link_entropy
    ], dtype=np.float32)

    if len(summary_vector) < feature_size:
        return np.concatenate([summary_vector, np.zeros(feature_size - len(summary_vector), dtype=np.float32)])
    return summary_vector[:feature_size]

def load_TDA_snapshot_feature(dataset, data_list, snapshots, dataset_idx, args, node_snapshots=None, feature_size=6, normalize=False):
    print("INFO: Loading TDA graph representation")

    partial_path = f'{DATA_PATH}/input/cached_feat/{dataset}'
    csv_cache_path = os.path.join(partial_path, f'{dataset}_TDA_features.csv')

    if os.path.exists(csv_cache_path):
        print(f"INFO: Loading cached TDA features from {csv_cache_path}")
        df = pd.read_csv(csv_cache_path)
        df['features'] = df['features'].apply(ast.literal_eval)
        df = df.sort_values(by='snapshot')
        TDA_feats = df['features'].tolist()
    else:
        print("INFO: No cache found. Computing TDA features from node features.")
        assert data_list is not None and node_snapshots is not None, \
            "Must provide `data_list` and `node_snapshots` to compute TDA."

        snapshot_ids = sorted(list(range(1, snapshots + 1)))
        TDA_list = []

        for snap in tqdm(snapshot_ids):
            edge_index = prepare(data_list[dataset_idx], snap, args)
            node_features = node_snapshots[snap]
            tda_vec = compute_mapper_from_node_features(edge_index, node_features, feature_size=feature_size)
            TDA_list.append((snap, tda_vec.tolist()))

        df_out = pd.DataFrame(TDA_list, columns=["snapshot", "features"])
        df_out.to_csv(csv_cache_path, index=False)
        TDA_feats = [x[1] for x in TDA_list]

    if normalize:
        TDA_feats = normalize_vector(TDA_feats)

    return TDA_feats


def load_labels(args, dataset, multi_property, partial_path):
    """Load labels for the temporal graph classification task."""
    if not multi_property:
        task = supported_properties["node_gs"]
        label_filename = f'{partial_path}/raw/labels/{task}/{dataset}_{task}_labels.csv'
        label_df = pd.read_csv(label_filename, header=None, names=['label'])
        TG_labels = torch.from_numpy(np.array(label_df['label'].tolist())).to(args.device)
    else:
        all_labels = []
        for property, task in supported_properties.items():
            label_filename = f'{partial_path}/raw/labels/{task}/{dataset}_{task}_labels.csv'
            if not os.path.exists(label_filename):
                raise FileNotFoundError(f"Label file for property '{property}' not found: {label_filename}")
            label_df = pd.read_csv(label_filename, header=None, names=['label'])
            all_labels.append(label_df['label'].tolist())
        multi_labels = list(map(list, zip(*all_labels)))
        TG_labels = torch.tensor(multi_labels, dtype=torch.float32).to(args.device)
    return TG_labels


def compute_node_features(edgelist_df, args):
    """Compute node-level features for each snapshot."""
    uniq_ts_list = np.unique(edgelist_df['snapshot'])
    # TG_node_features = np.zeros((args.num_nodes, args.nfeat))  # Fixed-size node feature list for each snapshot
    TG_node_feats_per_ts = []
    min_values = np.full((1, 4), np.inf)
    max_values = np.zeros((1, 4))
    mapped_nodes_ts = []

    for i, ts in enumerate(uniq_ts_list):
        print(f"INFO: Computing Node level features for snapshot {i} / {len(uniq_ts_list)}")
        ts_edges = edgelist_df.loc[edgelist_df['snapshot'] == ts, ['source', 'destination', 'weight']]
        ts_G = nx.from_pandas_edgelist(ts_edges, source='source', target='destination', edge_attr='weight',
                                       create_using=nx.MultiDiGraph)
        node_list = list(ts_G.nodes)

        indegree_list = np.array(ts_G.in_degree(node_list, weight=None))
        weighted_indegree_list = np.array(ts_G.in_degree(node_list, weight='weight'))
        outdegree_list = np.array(ts_G.out_degree(node_list, weight=None))
        weighted_outdegree_list = np.array(ts_G.out_degree(node_list, weight='weight'))

        TG_node_feat_ts = np.column_stack((
            indegree_list[:, 1].astype(float),
            weighted_indegree_list[:, 1].astype(float),
            outdegree_list[:, 1].astype(float),
            weighted_outdegree_list[:, 1].astype(float)
        ))

        snapshot_min_values = np.min(TG_node_feat_ts, axis=0)
        snapshot_max_values = np.max(TG_node_feat_ts, axis=0)
        min_values = np.minimum(snapshot_min_values, min_values)
        max_values = np.maximum(snapshot_max_values, max_values)
        # TODO: HERE we are adding the node float address Make sure we need it
        TG_node_feats_per_ts.append(TG_node_feat_ts)
        mapped_nodes_ts.append(map_addresses_to_floats(node_list))

    TG_node_feats_all_ts = []
    for ts, node_feat_ts in enumerate(TG_node_feats_per_ts):
        TG_node_features = np.zeros((args.num_nodes, args.nfeat))
        scaled_columns = []
        for i in range(node_feat_ts.shape[1]):
            scaled_column = normalize_column(
                node_feat_ts[:, i].reshape(-1, 1),
                col_min=min_values[0][i],
                col_max=max_values[0][i]
            )
            scaled_columns.append(scaled_column)

        scaled_columns.append(mapped_nodes_ts[ts])

        # Complete list of node features for all snapshots with fixed size
        scalced_node_features = np.column_stack(scaled_columns)
        TG_node_features[:scalced_node_features.shape[0]] = scalced_node_features
        # TODO: Reset TG_node_features
        TG_node_feats_all_ts.append(TG_node_features.copy())


        # TG_node_feats_all_ts.append(np.column_stack(scaled_columns))

    return TG_node_feats_all_ts


def compute_graph_features(edgelist_df, readout_scheme):
    """Compute graph-level features for each snapshot."""
    uniq_ts_list = np.unique(edgelist_df['snapshot'])
    TG_graph_feats = []

    for i, ts in enumerate(uniq_ts_list):
        print(f"INFO: Computing Graph level features for snapshot {i} / {len(uniq_ts_list)}")
        ts_edges = edgelist_df.loc[edgelist_df['snapshot'] == ts, ['source', 'destination', 'weight']]
        ts_G = nx.from_pandas_edgelist(ts_edges, source='source', target='destination', edge_attr='weight',
                                       create_using=nx.MultiDiGraph)
        node_list = list(ts_G.nodes)
        indegree_list = np.array(ts_G.in_degree(node_list))
        weighted_indegree_list = np.array(ts_G.in_degree(node_list, weight='weight'))
        outdegree_list = np.array(ts_G.out_degree(node_list))
        weighted_outdegree_list = np.array(ts_G.out_degree(node_list, weight='weight'))

        if readout_scheme == 'max':
            TG_this_ts_feat = np.array([np.max(indegree_list), np.max(weighted_indegree_list),
                                        np.max(outdegree_list), np.max(weighted_outdegree_list)])
        elif readout_scheme == 'mean':
            TG_this_ts_feat = np.array([np.mean(indegree_list[:, 1].astype(float)),
                                        np.mean(weighted_indegree_list[:, 1].astype(float)),
                                        np.mean(outdegree_list[:, 1].astype(float)),
                                        np.mean(weighted_outdegree_list[:, 1].astype(float))])
        elif readout_scheme == 'sum':
            TG_this_ts_feat = np.array([np.sum(indegree_list), np.sum(weighted_indegree_list),
                                        np.sum(outdegree_list), np.sum(weighted_outdegree_list)])
        else:
            raise ValueError("Readout scheme is Undefined!")

        TG_graph_feats.append(TG_this_ts_feat)

    return MinMaxScaler().fit_transform(TG_graph_feats)


def compute_and_cache_features(args, dataset, readout_scheme, partial_path, TG_labels):
    """Compute graph and node features and cache them."""
    edgelist_filename = f'{partial_path}/raw/edgelists/{dataset}_edgelist.txt'
    edgelist_df = pd.read_csv(edgelist_filename)

    print(f"INFO: Computing Graph level features for Dataset: {dataset}")
    TG_graph_feats = compute_graph_features(edgelist_df, readout_scheme)

    data_root = f'{partial_path}/cached_feat/{dataset}'
    if not os.path.exists(data_root):
        os.makedirs(data_root)

    cached_feature_path = f"{data_root}/{dataset}_features_{readout_scheme}.npz"
    np.savez(cached_feature_path, TG_graph_feats=TG_graph_feats)

    if args.node_feats:
        print(f"INFO: Computing Node level features for Dataset: {dataset}")
        TG_node_feats_all_ts = compute_node_features(edgelist_df, args)
        cached_node_feats_path = f"{data_root}/{dataset}_node_feats.pkl"
        with open(cached_node_feats_path, "wb") as f:
            pickle.dump(TG_node_feats_all_ts, f)
        # np.savez(cached_node_feats_path, TG_node_feats_all_ts=TG_node_feats_all_ts)
        return TG_labels, TG_graph_feats, TG_node_feats_all_ts

    return TG_labels, TG_graph_feats


def extra_dataset_attributes_loading(args, dataset, readout_scheme='mean', multi_property=False):
    """Main function to load and process additional dataset attributes."""
    print(f"INFO: Loading a Graph Feature and Labels for `Temporal Graph Foundation Model (TGFM)`: Dataset: {dataset}")
    partial_path = '../../data/input/'

    TG_labels = load_labels(args, dataset, multi_property, partial_path)

    data_root = f'{partial_path}/cached_feat/{dataset}'
    cached_feature_path = f"{data_root}/{dataset}_features_{readout_scheme}.npz"
    cached_node_feats_path = f"{data_root}/{dataset}_node_feats.pkl"

    print(f"LOADING cache from  : {cached_feature_path}")

    if os.path.exists(cached_feature_path):
        TG_graph_feats = np.load(cached_feature_path)['TG_graph_feats']
        if args.node_feats and os.path.exists(cached_node_feats_path):
            # TG_node_feats = np.load(cached_node_feats_path)['TG_node_feats_all_ts']
            with open(cached_node_feats_path, "rb") as f:
                TG_node_feats = pickle.load(f)

            print("INFO: Graph and node features already exist. Loaded directly.")
            return TG_labels, TG_graph_feats, TG_node_feats
        elif not args.node_feats:
            print("INFO: Graph features already exist. Loaded directly.")
            return TG_labels, TG_graph_feats

    print(f"INFO: Cached features not found. Generating new features for dataset: {dataset}.")
    return compute_and_cache_features(args, dataset, readout_scheme, partial_path, TG_labels)



def map_addresses_to_floats(addresses):
    # Convert addresses to bytes
    # address_bytes = [bytes.fromhex(str(addr).replace('0x', '')) for addr in addresses]
    address_bytes = []
    for addr in addresses:
        if isinstance(addr, str) and addr.startswith('0x') and all(c in '0123456789abcdefABCDEF' for c in addr[2:]):
            try:
                address_bytes.append(bytes.fromhex(str(addr).replace('0x', '')))
            except ValueError as e:
                print(f"Invalid address: {addr} Error: {e}")
                address_bytes.append(bytes(str(addr), 'utf-8'))  # Convert to bytes
        else:
            address_bytes.append(bytes(str(addr), 'utf-8')) # Convert to bytes

    # Hash each address using SHA-256
    hashed_addresses = [hashlib.sha256(addr).digest() for addr in address_bytes]
    hash_ints = [int.from_bytes(h, 'big') for h in hashed_addresses]
    hash_ints = np.array(hash_ints)
    float_values = (hash_ints % 10 ** 16) / 10 ** 16
    return 1 + (float_values * 9)


def normalize_column(column, col_max=None, col_min=None, new_min=0, new_max=1):
    '''
    Function to normalize node features per column (feature)
    '''
    if col_max is None:
        col_min = np.min(column)
        col_max = np.max(column)
    normalized_col = (column - col_min) / (col_max - col_min) * (new_max - new_min) + new_min
    return normalized_col


def load_TGC_dataset(dataset):
    print("INFO: Loading a Graph from `Temporal Graph Classification (TGC)` Category: {}".format(dataset))
    data = {}

    edgelist_rawfile = '{}/input/raw/edgelists/{}_edgelist.txt'.format(DATA_PATH, dataset)
    edgelist_df = pd.read_csv(edgelist_rawfile)
    uniq_ts_list = np.unique(edgelist_df['snapshot'])
    print("INFO: Number of unique snapshots: {}".format(len(uniq_ts_list)))
    adj_time_list = []
    for ts in uniq_ts_list:
        # NOTE: this code does not use any node or edge features
        ts_edges = edgelist_df.loc[edgelist_df['snapshot'] == ts, ['source', 'destination']]
        ts_G = nx.from_pandas_edgelist(ts_edges, 'source', 'destination')
        ts_A = nx.to_scipy_sparse_array(ts_G)
        adj_time_list.append(ts_A)

    # Now, exactly like "load_vgrnn_dataset_det"
    print('INFO: Generating edges, negative edges and new edges, wait for a while ...')
    edge_proc_start = time.time()
    data = {}
    edges, biedges = mask_edges_det(adj_time_list)  # list
    # pedges, nedges = mask_edges_prd(adj_time_list)  # list
    # new_pedges, new_nedges = mask_edges_prd_new_by_marlin(adj_time_list)  # list
    print('INFO: Processing finished! Elapsed time (sec.): {:.4}'.format(time.time() - edge_proc_start))
    # assert len(edges) == len(biedges) == len(pedges) == len(nedges) == len(new_nedges) == len(new_pedges)
    assert len(edges) == len(biedges)
    edge_index_list, pedges_list, nedges_list, new_nedges_list, new_pedges_list = [], [], [], [], []
    for t in range(len(biedges)):
        edge_index_list.append(torch.tensor(np.transpose(biedges[t]), dtype=torch.long))
        # pedges_list.append(torch.tensor(np.transpose(pedges[t]), dtype=torch.long))
        # nedges_list.append(torch.tensor(np.transpose(nedges[t]), dtype=torch.long))
        # new_pedges_list.append(torch.tensor(np.transpose(new_pedges[t]), dtype=torch.long))
        # new_nedges_list.append(torch.tensor(np.transpose(new_nedges[t]), dtype=torch.long))

    data['edge_index_list'] = edge_index_list
    # data['pedges'], data['nedges'] = pedges_list, nedges_list
    # data['new_pedges'], data['new_nedges'] = new_pedges_list, new_nedges_list  # list
    data['num_nodes'] = int(np.max(np.vstack(edges))) + 1

    data['time_length'] = len(edge_index_list)
    data['weights'] = None
    print('INFO: Data: {}'.format(dataset))
    print('INFO: Total length:{}'.format(len(edge_index_list)))
    print('INFO: Number nodes: {}'.format(data['num_nodes']))
    return data


def loader(dataset='enron10', neg_sample=''):
    # if cached, load directly
    data_root = '{}/input/cached/{}/'.format(DATA_PATH, dataset)
    filepath = mkdirs(data_root) + '{}.data'.format(dataset)  # the data will be saved here after generation.
    print("INFO: Loading Dataset: {}".format(dataset))
    print("DEBUG: Look for data at {}.".format(filepath))
    if os.path.isfile(filepath):
        print('INFO: Loading {} directly.'.format(dataset))
        return torch.load(filepath)

    # if not cached, to process and cached
    TGS_dataset_df = pd.read_csv(f"{DATA_PATH}/TGS_available_datasets.csv")
    TGS_social_dataset_df = pd.read_csv(f"{DATA_PATH}/TGS_available_datasets_social.csv")

    print(f"Social Datasets : {TGS_social_dataset_df}")

    TGS_available_dataset = TGS_dataset_df['token_name'].tolist() + TGS_social_dataset_df['social_name'].tolist()

    print(f"ALL available datasets : {TGS_available_dataset}")

    if dataset in TGS_available_dataset:
        edgelist_rawfile = '{}/input/raw/edgelists/{}_edgelist.txt'.format(DATA_PATH, dataset)
        if os.path.exists(edgelist_rawfile):
            print("Working on dataset {} at edge raw : {}".format(dataset, edgelist_rawfile))
            data = load_TGC_dataset(dataset)
        else:
            TGS_Handler("{}/input/tokens/raw/").creat_baseline_datasets_edge_GS(DATA_PATH, dataset)
            print("=============Done===============")
            data = load_TGC_dataset(dataset)
    else:
        raise ValueError("ERROR: Undefined dataset!")

    torch.save(data, filepath)
    print('INFO: Dataset is saved!')
    return data


def process_data_gaps(directory, min_size=10, max_size=4000):
    columns = ["blockNumber", "timestamp", "tokenAddress", "from", "to", "value", "fileBlock"]
    dataset_feature_file = open('dataset_features.txt', 'w')
    dataset_feature_file.writelines(["filename, start, end, duration, max_gap,networkSize\n"])
    file_count = len(os.listdir(directory))
    counter = 0
    for filename in os.listdir(directory):
        counter += 1
        filepath = directory + "/" + filename
        file_size = os.path.getsize(filepath) / (1024 * 1024)
        try:
            if filename.endswith('.csv') and file_size >= min_size and file_size <= max_size:
                data = pd.read_csv(filepath, usecols=columns, index_col=False)

                timestamps = pd.to_datetime(data["timestamp"], unit="s").dt.date
                start = timestamps[0]
                end = timestamps.iloc[-1]
                time_difference = (end - start).days
                if time_difference < 20:
                    raise Exception("Token network last less than 20 days")

                unique_timestamps = timestamps.unique()
                tot_len = len(unique_timestamps)
                gaps = max(set([(unique_timestamps[i + 1] - unique_timestamps[i]).days for i in range(tot_len - 1)]))
                dataset_feature_file.writelines(
                    [filename, ",", str(start), ",", str(end), ",", str(time_difference), ",", str(gaps), ",",
                     str(file_size), "\n"])
        except Exception as e:
            print("ERROR while processing {} due to\n {}".format(filename, e))

        print("Done processing {}/{}".format(counter, file_count))
    dataset_feature_file.close()


def select_datset_no_gap(filename, max_gap):
    dataset_df = pd.read_csv(filename)
    filtered_df = dataset_df[dataset_df[' max_gap'] <= max_gap]
    filtered_df.to_csv('dataset_no_gap_{}_day.csv'.format(max_gap), index=False)


def load_multiple_datasets(datasets_package_path=""):
    datasets_packages = []
    dataset_names = []

    print(datasets_package_path)
    if os.path.exists(datasets_package_path):
        print("Folder exists.")
    else:
        print("Folder does not exist.")
    i = 0
    text_path = "{}/input/data_list/{}".format(DATA_PATH, datasets_package_path)

    try:
        with open(text_path, 'r') as file:
            for line in file:
                print("INFO: Dataset: {}".format(line))
                datasets_packages.append(loader(dataset=line.strip()))
                dataset_names.append(line.strip())

    except Exception as e:
        print("ERROR: error in processing data pack {}".format(datasets_package_path))
        print(e)

    print("Number of dataset: {}".format(len(datasets_packages)))
    return dataset_names, datasets_packages


def load_multiple_datasets_modified(data_pack_list):
    """
    Load multiple datasets from a list of dataset names
    """
    datasets_packages = []
    dataset_names = []

    for dataset in data_pack_list:
        print("INFO: Dataset: {}".format(dataset))
        datasets_packages.append(loader(dataset=dataset))
        dataset_names.append(dataset)

    print("Number of dataset: {}".format(len(datasets_packages)))
    return dataset_names, datasets_packages


def find_max_node_id(dataname):
    data_path = '{}/input/raw/edgelists/{}_edgelist.txt'.format(DATA_PATH, dataname)
    data_df = pd.read_csv(data_path)
    unique_node = set()
    unique_node.update(data_df['source'].tolist())
    unique_node.update(data_df['destination'].tolist())
    return max(unique_node)


def find_max_node_id_package(datasets_package_file):
    text_path = "{}/{}".format(DATA_PATH, datasets_package_file)
    max_id_dataset = []
    try:
        with open(text_path, 'r') as file:
            for dataset in file:
                max_id_dataset.append(find_max_node_id(dataset.strip()))
        return int(max(max_id_dataset))

    except Exception as e:
        print("ERROR: error in processing data pack {}".format(datasets_package_file))
        print(e)


def prepare(data, t, args, detection=False):
    if detection == False:
        # obtain adj index
        edge_index = data['edge_index_list'][t].long().to(args.device)  # torch edge index
        # pos_index = data['pedges'][t].long().to(args.device)  # torch edge index
        # neg_index = data['nedges'][t].long().to(args.device)  # torch edge index
        # new_pos_index = data['new_pedges'][t].long().to(args.device)  # torch edge index
        # new_neg_index = data['new_nedges'][t].long().to(args.device)  # torch edge index
        # 2.Obtain current updated nodes
        # nodes = list(np.intersect1d(pos_index.numpy(), neg_index.numpy()))
        # 2.Obtain full related nodes
        # nodes = list(np.union1d(pos_index.cpu().numpy(), neg_index.cpu().numpy()))
        # weights = None
        return edge_index

    if detection == True:
        train_pos_edge_index = data['gdata'][t].train_pos_edge_index.long().to(args.device)

        val_pos_edge_index = data['gdata'][t].val_pos_edge_index.long().to(args.device)
        val_neg_edge_index = data['gdata'][t].val_neg_edge_index.long().to(args.device)

        test_pos_edge_index = data['gdata'][t].test_pos_edge_index.long().to(args.device)
        test_neg_edge_index = data['gdata'][t].test_neg_edge_index.long().to(args.device)
        return train_pos_edge_index, val_pos_edge_index, val_neg_edge_index, test_pos_edge_index, test_neg_edge_index

def normalize_label_file(file_path, output_path, range_max):
    with open(file_path, 'r') as file:
        numbers = [float(line.strip()) for line in file]

    if not numbers or max(numbers) == min(numbers):
        print(f"Skipping {file_path}: empty or identical values.")
        return

    # Normalize values to the range [0, range_max]
    min_val, max_val = min(numbers), max(numbers)
    normalized_numbers = [(num - min_val) / (max_val - min_val) * range_max for num in numbers]

    # Write the normalized values to the output file
    with open(output_path, 'w') as file:
        for normalized_num in normalized_numbers:
            file.write(f"{normalized_num}\n")

    print(f"Normalized file saved as {output_path}")

def normalize_all_label_files(input_dir, output_dir, range_max):
    # Ensure the output directory exists
    output_dir = '../../data/input/raw/labels/{}/'.format(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    input_dir = '../../data/input/raw/labels/{}/'.format(input_dir)
    # Iterate over all files in the input directory
    for file_name in os.listdir(input_dir):
        input_file_path = os.path.join(input_dir, file_name)
        output_file_name = f"{os.path.splitext(file_name)[0]}_normalized.txt"
        output_file_path = os.path.join(output_dir, output_file_name)

        # Normalize each file
        normalize_label_file(input_file_path, output_file_path, range_max)
