"""
Implement a graph decoder and corresponding utility functions.

"""

import torch



class MLP(torch.nn.Module):
    """
  MLP for binary classification.
  """

    def __init__(self, in_dim, hid_dim, out_dim=1, n_layers=3, drop=0.5):
        super().__init__()
        self.n_layers = n_layers
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(in_dim, hid_dim)
        ] + [
            torch.nn.Linear(hid_dim, hid_dim) for _ in range(n_layers - 2)
        ])
        self.layers.append(torch.nn.Linear(hid_dim, out_dim))
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=drop, inplace=False)

    def forward(self, x):
        for layer_idx in range(self.n_layers - 1):
            x = self.act(self.layers[layer_idx](x))
            x = self.dropout(x)
        return self.layers[-1](x).squeeze(dim=1)


class MultiPropertyMLP(torch.nn.Module):
    """
    MLP for multi-property prediction.
    Concatenates the output of multiple MLPs.
    """
    def __init__(self, in_dim, hid_dim, out_dim, num_properties, drop=0.5):
        super().__init__()
        self.mlps = torch.nn.ModuleList([
            MLP(in_dim, hid_dim, out_dim, n_layers=3, drop=drop)
            for _ in range(num_properties)
        ])

    def forward(self, x):
        return torch.cat([mlp(x) for mlp in self.mlps], dim=0)


class MLP_hardcoded(torch.nn.Module):
  """
  Binary Classifier to work as a decoder

  reference:
    https://github.com/twitter-research/tgn/blob/master/utils/utils.py
  """
  def __init__(self, in_dim, hidden_dim_1, hidden_dim_2, drop=0.5):
    super().__init__()
    self.fc_1 = torch.nn.Linear(in_dim, hidden_dim_1)
    self.fc_2 = torch.nn.Linear(hidden_dim_1, hidden_dim_2)
    self.fc_3 = torch.nn.Linear(hidden_dim_1, 1)
    self.act = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=drop, inplace=False)

  def forward(self, x):
    x = self.act(self.fc_1(x))
    x = self.dropout(x)
    x = self.act(self.fc_2(x))
    x = self.dropout(x)
    return self.fc_3(x).squeeze(dim=1)
