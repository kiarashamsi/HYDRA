import random
import os
import torch
import numpy as np
import logging
import pandas as pd
from scipy.linalg import eigh
from scipy.sparse import csgraph
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
from scipy.sparse.linalg import eigsh



logger = logging.getLogger()
disease_path = '../data/input/raw/disease/disease_lp.feats.npz'


def mkdirs(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    return folder

def prepare_dir(output_folder):
    mkdirs(output_folder)
    log_folder = mkdirs(output_folder)
    return log_folder


def write_to_text(args, results, seed):
    if args.to_compute_results:
        with open(args.result_txt, "a") as f:
            data = str(seed) + '\t' + '\t'.join(['{:.6f}'.format(result) for result in results[1:]]) + '\n'
            f.write(data)
    else:
        print('Not write to file')


def compute_mean_std(args):
    if args.to_compute_results:
        results = np.loadtxt(args.result_txt, delimiter='\t')
        results = results[:, 1:]
        mean_list = np.mean(results, axis=0)
        std_list = np.std(results, axis=0)
        with open(args.result_txt, "a") as f:
            data = 'results: ' + '\t'.join(
                ['{:.2f}+{:.2f}'.format(mean * 100, std * 100) for mean, std in zip(mean_list, std_list)]) + '\n'
            f.write(data)
    else:
        print('Not write to file')


def set_random(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print('INFO: fixed random seed')


def init_logger(log_file=None):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger


def adjust_learning_rate(lr, optimizer, epoch):
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def filters(edge_index, threshold):
    return np.where(np.where(edge_index >= threshold, 1, 0).sum(axis=0) == 0)


def neg_sampling_basic(edge_index, num_nodes):
    import random
    num_neg_samples = edge_index.shape[1]
    idx = edge_index[0] * num_nodes + edge_index[1]
    rng = range(num_nodes ** 2)
    perm = torch.tensor(random.sample(rng, num_neg_samples))
    mask = torch.from_numpy(np.isin(perm, idx)).to(torch.bool)
    rest = mask.nonzero().view(-1)
    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.tensor(random.sample(rng, rest.size(0)))
        mask = torch.from_numpy(np.isin(tmp, idx)).to(torch.bool)
        perm[rest] = tmp
        rest = rest[mask.nonzero().view(-1)]
    row, col = torch.true_divide(perm, num_nodes), perm % num_nodes
    return torch.stack([row, col], dim=0).long().numpy()


def negative_sampling(adjs, cum_nodes):
    '''
    negative sampling each time stamp without sampling previous time
    '''

    history_perm = []
    neg_adjs = []
    for t in range(len(adjs)):
        edge_index = adjs[t]
        num_nodes = len(cum_nodes[t])
        num_neg_samples = edge_index.shape[1]  # get number of negative sampling nodes
        current_perm = edge_index[0] * num_nodes + edge_index[1]  # obtain the total hash slots
        idx = np.concatenate([current_perm, history_perm])  # expand the slots with historical slots
        rng = range(num_nodes ** 2)  # get range of negative sampling
        perm = torch.tensor(random.sample(rng, num_neg_samples))  # get negative perm
        mask = torch.from_numpy(np.isin(perm, idx)).to(torch.bool)
        rest = mask.nonzero().view(-1)
        while rest.numel() > 0:  # pragma: no cover
            tmp = torch.tensor(random.sample(rng, rest.size(0)))
            mask = torch.from_numpy(np.isin(tmp, idx)).to(torch.bool)
            perm[rest] = tmp
            rest = rest[mask.nonzero().view(-1)]
        # row, col = perm / num_nodes, perm % num_nodes
        row, col = torch.true_divide(perm, num_nodes), perm % num_nodes
        neg_adjs.append(torch.stack([row, col], dim=0).long())
        history_perm += list(perm.numpy())
    return neg_adjs


def total_pairwise_similarity(x):
    # x = F.normalize(x)
    z = torch.matmul(x, x.transpose(0, 1))
    n = z.size(1) * (z.size(1) - 1)  # A_2^n
    total_similarity = torch.sum(z) - torch.sum(z.diagonal())
    average = total_similarity / n
    return 1 - average


# def compute_algebraic_connectivity(graph):
#     # Get the adjacency matrix of the graph
#     adj_matrix = nx.adjacency_matrix(graph).astype(float)  # Convert to float
#
#     # Compute the Laplacian matrix
#     laplacian = csgraph.laplacian(adj_matrix, normed=False)
#
#     # Add a small perturbation to avoid singularities
#     laplacian += np.eye(laplacian.shape[0]) * 1e-5  # Adding a small value to the diagonal
#
#     # Compute the second smallest eigenvalue (algebraic connectivity)
#     try:
#         eigenvalues, _ = eigsh(laplacian.astype(float), k=2, which='SM', maxiter=5000)  # Increase max iterations
#         return eigenvalues[1]  # Return the algebraic connectivity (second smallest eigenvalue)
#     except Exception as e:
#         print(f"Failed to compute eigenvalues: {e}")
#         return None


def compute_algebraic_connectivity(graph):
    # Get the adjacency matrix of the graph
    adj_matrix = nx.adjacency_matrix(graph).astype(float)  # Convert to float

    # Compute the Laplacian matrix
    degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
    laplacian = degree_matrix - adj_matrix

    # Add a small perturbation to avoid singularities
    laplacian += np.eye(laplacian.shape[0]) * 1e-5  # Adding a small value to the diagonal

    # Use scipy.linalg.eigh to compute all eigenvalues
    eigenvalues = eigh(laplacian, eigvals_only=True)

    # Sort and return the second smallest eigenvalue (algebraic connectivity)
    largest_cc = max(len(c) for c in nx.connected_components(graph))
    print("LEN : {}".format(largest_cc))
    eigenvalues = np.sort(eigenvalues)
    print(max(np.abs(eigenvalues)))
    return eigenvalues[1]


def compute_spectral_radious(graph):
    # Get the adjacency matrix of the graph
    adj_matrix = nx.adjacency_matrix(graph).astype(float)  # Convert to float

    # Compute the Laplacian matrix
    degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
    laplacian = degree_matrix - adj_matrix

    # Add a small perturbation to avoid singularities
    laplacian += np.eye(laplacian.shape[0]) * 1e-5  # Adding a small value to the diagonal

    # Use scipy.linalg.eigh to compute all eigenvalues
    eigenvalues = eigh(laplacian, eigvals_only=True)

    return max(np.abs(eigenvalues))

def compute_number_of_connected_components(graph):

    try:
        largest_cc = max(len(c) for c in nx.connected_components(graph))
    except Exception as e:
        largest_cc = 0
        print("No CONNECTED COMPONENT")
    return largest_cc


def save_or_append_dict_to_csv(data, filename):
    """
    save a dictionary to a file as a row of a dataframe
    """
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame([data])

    # Check if the CSV file already exists
    if os.path.isfile(filename):
        # If the file exists, read it and append the new row
        df_existing = pd.read_csv(filename)
        df_combined = pd.concat([df_existing, df], ignore_index=True)
        df_combined.to_csv(filename, index=False)
    else:
        # If the file doesn't exist, save the new DataFrame to CSV
        df.to_csv(filename, index=False)


import numpy as np


def normalize_vector(vector, method="min-max"):
    """
    Normalize a list of NumPy arrays using Min-Max scaling, Z-score normalization,
    L2 normalization, or Max normalization.

    Parameters:
    - vector: list of numpy arrays
    - method: str, "min-max" for Min-Max scaling, "z-score" for standardization,
              "l2" for L2 normalization, "max" for Max normalization

    Returns:
    - list of normalized numpy arrays
    """
    concatenated = np.concatenate(vector)  # Flatten to find global min/max or mean/std

    if method == "min-max":
        min_val = np.min(concatenated)
        max_val = np.max(concatenated)
        normalized = [(arr - min_val) / (max_val - min_val) for arr in vector]
    elif method == "z-score":
        mean_val = np.mean(concatenated)
        std_val = np.std(concatenated)
        normalized = [(arr - mean_val) / std_val for arr in vector]
    elif method == "l2":
        normalized = [arr / np.linalg.norm(arr, ord=2) for arr in vector]
    elif method == "max":
        max_val = np.max(concatenated)
        normalized = [arr / max_val for arr in vector]
    else:
        raise ValueError("Unsupported normalization method. Use 'min-max', 'z-score', 'l2', or 'max'.")

    return normalized
