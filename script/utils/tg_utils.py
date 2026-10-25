"""
Utility functions for aggregating graph data.

"""

import torch

def readout_function(embeddings, readout_scheme='mean'):
    """
    Read out function to generate a representation for the whole graph
    reference:    
        https://github.com/qbxlvnf11/graph-neural-networks-for-graph-classification/blob/master/readouts/basic_readout.py
    """
    # note: x.size(): [#nodes, args.n_out]
    if readout_scheme == 'max':
        readout = torch.max(embeddings, dim=0)[0].squeeze()  # max readout
    elif readout_scheme == 'mean':
        readout = torch.mean(embeddings, dim=0).squeeze()  # mean readout
    elif readout_scheme == 'sum':
        readout = torch.sum(embeddings, dim=0).squeeze()  # sum readout
    else:
        readout = None
        raise ValueError("Readout Method Undefined!")
    return readout


def load_feature(num_nodes, nfeats):
    """
    Load the features of the nodes in the graph
    TODO: right now, we are using random features;
    in the future, we should use the real features of the nodes
    """
    node_feat = torch.randn((num_nodes, nfeats))
    return node_feat