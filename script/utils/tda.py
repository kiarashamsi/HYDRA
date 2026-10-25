import pandas as pd
import statistics as stat
import torch
import kmapper as km
import sklearn
import networkx as nx
import numpy as np
import torch
from scipy.sparse import load_npz
import os
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

ref_tda_param = {
    "BAG":{"over_lap":0.15,"n_cube":2,"cls":5},
    "QOM":{"over_lap":0.2,"n_cube":2,"cls":5},
    "BEPRO":{"over_lap":0.2,"n_cube":2,"cls":5},
    "Metis":{"over_lap":0.15,"n_cube":2,"cls":5},
    "REPv2":{"over_lap":0.2,"n_cube":2,"cls":5},
    "stkAAVE":{"over_lap":0.3,"n_cube":5,"cls":2},
    "ADX":{"over_lap":0.15,"n_cube":2,"cls":5}

}

def average_node(token_df:pd.DataFrame)-> float:
    snapshot = number_of_snapshot(token_df)
    number_nodes_list = []
    for i in range(1,snapshot + 1):
        rows = token_df[token_df['snapshot'] == i]
        source = set(rows['source'].unique())
        target = set(rows['destination'].unique())
        snapshot_nodes = source | target
        number_nodes_list.append(len(snapshot_nodes))

    return stat.mean(number_nodes_list)

def average_edge(token_df:pd.DataFrame) -> float:
    snapshot = number_of_snapshot(token_df)
    number_edge_list = []
    for i in range(1,snapshot + 1):
        rows = token_df[token_df['snapshot'] == i]
        number_edge_list.append(len(rows))

    return stat.mean(number_edge_list)

def average_new_nodes(token_df:pd.DataFrame)-> float:
    snapshot = number_of_snapshot(token_df)
    number_new_nodes_list = []
    seen_nodes = set()
    for i in range(1,snapshot + 1):
        rows = token_df[token_df['snapshot'] == i]
        source = set(rows['source'].unique())
        target = set(rows['destination'].unique())
        snapshot_nodes = source | target

        new_nodes = snapshot_nodes.difference(seen_nodes)
        number_new_nodes_list.append(len(new_nodes))
        seen_nodes.update(snapshot_nodes)

    return stat.mean(number_new_nodes_list)

def number_of_snapshot(token_df:pd.DataFrame) -> int:
    return token_df['snapshot'].max()

def compute_static_representation(token_df:pd.DataFrame, metrics = [average_node,average_edge,average_new_nodes,number_of_snapshot]) -> torch.Tensor:
    representation = []
    for metric in metrics:
        representation.append(metric(token_df))
    return torch.Tensor(representation)

def tda_hyper_param_lookup(token_name:str,metrics = [average_node,average_edge,average_new_nodes,number_of_snapshot]):
    edgelist_path = "data/input/raw/edgelists"
    target_token_df = pd.read_csv(f"{edgelist_path}/{token_name}_edgelist.txt")
    similarity_dict = {}
    similarity_metric = torch.nn.CosineSimilarity(dim=0)
    target_representation = compute_static_representation(target_token_df,metrics)

    for ref_token in ref_tda_param:
        ref_token_df = pd.read_csv(f"{edgelist_path}/{ref_token}_edgelist.txt")
        ref_representation = compute_static_representation(ref_token_df,metrics)
        similarity_score = similarity_metric(ref_representation,target_representation)
        similarity_dict[ref_token] = similarity_score.item()

    return ref_tda_param[max(similarity_dict,key=similarity_dict.get)]

def extract_graph_features(graph):
    pr = nx.pagerank(graph,0.9)
    dc = nx.degree_centrality(graph)
    cc = nx.closeness_centrality(graph)
    bx = nx.betweenness_centrality(graph)
    c = nx.clustering(graph)

    #create list for each features
    pr_list =  [i for i in pr.values()]
    dc_list =  [i for i in dc.values()]
    cc_list =  [i for i in cc.values()]
    bx_list =  [i for i in bx.values()]
    c_list =  [i for i in c.values()]
    d_list = [val for (node, val) in graph.degree()]
    data = np.column_stack((pr_list,dc_list,cc_list,bx_list,c_list,d_list))
    return data

def TDA_transformation(data,cluster=5,overlap=0.7,n_cubes=2)->nx.Graph:
    Xfilt = data
    mapper = km.KeplerMapper()
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
    Xfilt = scaler.fit_transform(Xfilt)
    lens = mapper.fit_transform(Xfilt, projection=sklearn.manifold.TSNE())

    graph = mapper.map(lens,Xfilt,clusterer=sklearn.cluster.KMeans(n_clusters=cluster,random_state=1618033),
        cover=km.Cover(n_cubes=n_cubes, perc_overlap=overlap)) #2 - 0.3
    return km.to_nx(graph)

def from_networkx_graph_to_edgelist(graph)->torch.Tensor:
    csr_matrix = nx.to_scipy_sparse_array(graph)
    coo = csr_matrix.tocoo()
    edgelist = torch.tensor([[],[]])
    for source,dest in list(zip(coo.row, coo.col)):
        if source < dest:
            edgelist = torch.concat((edgelist,torch.Tensor([source,dest]).unsqueeze(dim=1)), dim= 1)
    return edgelist.to(dtype=torch.int32)

def generate_TDA_graph(dataset):
    partial_path = '../../data/input'
    cached_file = f"{partial_path}/cached/{dataset}/tda_graphs.pt"
    if os.path.exists(cached_file):
        return torch.load(cached_file)
    else:
        print("INFO: generate TDA graph. May take a while")
        edgelist_filename = f'{partial_path}/raw/edgelists/{dataset}_edgelist.txt'
        edgelist_df = pd.read_csv(edgelist_filename)
        uniq_ts_list = np.unique(edgelist_df['snapshot'])
        tda_edgelist_list = []
        for ts in tqdm(uniq_ts_list):
            ts_edges = edgelist_df.loc[edgelist_df['snapshot'] == ts, ['source', 'destination', 'weight']]
            ts_G = nx.from_pandas_edgelist(ts_edges, source='source', target='destination', edge_attr='weight',
                                        create_using=nx.MultiDiGraph)
            node_list = list(ts_G.nodes)
            # indegree_list = ts_G.in_degree(node_list)
            # weighted_indegree_list = ts_G.in_degree(node_list, weight='weight')
            # outdegree_list = ts_G.out_degree(node_list)
            # weighted_outdegree_list = ts_G.out_degree(node_list, weight='weight')

            indegree_list = [val for (node, val) in ts_G.in_degree(node_list)]
            weighted_indegree_list = [val for (node, val) in ts_G.in_degree(node_list, weight='weight')]
            outdegree_list = [val for (node, val) in ts_G.out_degree(node_list)]
            weighted_outdegree_list = [val for (node, val) in ts_G.out_degree(node_list, weight='weight')]


            data = np.column_stack((indegree_list,
                                    weighted_indegree_list,
                                    outdegree_list,
                                    weighted_outdegree_list))
            print(data.shape)
            print("=======")
            tda_graph = TDA_transformation(data)
            edge_list = from_networkx_graph_to_edgelist(tda_graph)
            print(edge_list.shape)
            tda_edgelist_list.append(edge_list)

        torch.save(tda_edgelist_list,cached_file)
        return edge_list


if __name__ == "__main__":
    # graph = nx.erdos_renyi_graph(300,0.4)
    # print(graph.number_of_nodes())
    # print(graph.number_of_edges())

    # data = extract_graph_features(graph)
    # tda_graph = TDA_transformation(data)
    # print(tda_graph.number_of_nodes())
    # print(tda_graph.number_of_edges())
    # dummy_list = []
    # dummy_list.append(from_networkx_graph_to_edgelist(tda_graph))
    # dummy_list.append(from_networkx_graph_to_edgelist(tda_graph))

    # torch.save(dummy_list,"test.pt")
    # load_list = torch.load("test.pt")
    # print(load_list)
    edge_list = generate_TDA_graph("BAG")
    print(edge_list)
