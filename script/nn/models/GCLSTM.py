import torch
from torch_geometric_temporal import GCLSTM
import torch.nn.functional as F
class RecurrentGCN(torch.nn.Module):
    """
        GCLSTM model from PyTorch Geometric Temporal
    """
    def __init__(self, node_feat_dim ,hidden_dim):
        super(RecurrentGCN, self).__init__()
        self.recurrent = GCLSTM(node_feat_dim, hidden_dim, 1)
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)


    def forward(self, x, edge_index, edge_weight, h, c):
        h_0, c_0 = self.recurrent(x, edge_index, edge_weight, h, c)
        h = F.relu(h_0)
        h = self.linear(h)
        return h, h_0, c_0

    def init_hiddens(self):
        self.h = None
        self.h_0 = None
        self.c_0 = None

        # Detach gradient

    def detach_hidden(self):
        self.h = self.h.detach()
        self.h_0 = self.h_0.detach()
        self.c_0 = self.c_0.detach()
