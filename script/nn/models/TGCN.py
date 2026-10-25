"""
PyTorch-Geometric implemnetation of Temporal Graph Convolutional Network (TGCN)
"""
import math

import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import TGCN
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from tgp.poolers import get_pooler  # SRC(L) unified API

class RecurrentGCN_TGCN(torch.nn.Module):
    def __init__(self, node_features, hid_dim, out_dim, n_layers=2):
        super(RecurrentGCN_TGCN, self).__init__()
        self.layers = []
        for layer_idx in range(n_layers):
            if layer_idx == 0:
                self.layers.append(TGCN(node_features, hid_dim))
            elif layer_idx == n_layers - 1:
                self.layers.append(torch.nn.Linear(hid_dim, out_dim))
            else:
                self.layers.append(TGCN(hid_dim, hid_dim))
        self.activ = F.relu
        self.n_layers = n_layers

    def forward(self, x, edge_index, edge_weight, prev_hidden_state):
        for layer_idx in range(self.n_layers - 1):
            prev_hidden_state = self.layers[layer_idx](x, edge_index, edge_weight, prev_hidden_state)
            x = self.activ(prev_hidden_state)
        # last layer is a linear layer
        y = self.layers[-1](x)
        return y, prev_hidden_state


class RecurrentGCN_TGCN_one_layer(torch.nn.Module):
    """
    Testing the function of the TGCN model with one layer
    """
    def __init__(self, node_features, hid_dim, out_dim):
        super(RecurrentGCN_TGCN_one_layer, self).__init__()
        self.layer = TGCN(node_features, hid_dim)
        self.activ = F.relu
        self.out_layer = torch.nn.Linear(hid_dim, out_dim)

    def forward(self, x, edge_index, edge_weight, prev_hidden_state):
        prev_hidden_state = self.layer(x, edge_index, edge_weight, prev_hidden_state)

        x = self.activ(prev_hidden_state)
        y = self.out_layer(x)
        return y, prev_hidden_state

class HiPPOStyleMemory(nn.Module):
    def __init__(self, input_dim, memory_dim):
        super().__init__()
        # Learnable decay rates, clamped to stay within a safe range
        self.A_diag = nn.Parameter(torch.linspace(0.1, 1.0, memory_dim))  # positive; will flip sign during exp
        self.B = nn.Linear(input_dim, memory_dim)
        self.activation = nn.Tanh()  # adds non-linearity for stability

    def forward(self, memory, x):
        # Clamp decay to prevent instability
        decay = torch.exp(-torch.clamp(self.A_diag, min=0.0, max=10.0))  # always in (0, 1]
        decay = decay.unsqueeze(0).expand_as(memory)  # broadcast to match batch size

        # Apply decay and add projected input
        decayed_memory = memory * decay
        update = self.activation(self.B(x))
        new_memory = decayed_memory + update

        # Final safety check to ensure no NaNs
        new_memory = torch.nan_to_num(new_memory, nan=0.0, posinf=1.0, neginf=-1.0)

        return new_memory


class RecurrentGCN_TGCN_one_layer_ssm(nn.Module):
    """
    Safe and stable TGCN + SSM memory model.
    No change to user-facing API.
    """
    def __init__(self, node_features, hid_dim, out_dim, memory_dim=32):
        super(RecurrentGCN_TGCN_one_layer_ssm, self).__init__()
        self.layer = TGCN(node_features, hid_dim)
        self.activ = F.relu
        self.out_layer = nn.Linear(hid_dim + memory_dim, out_dim)

        self.memory_dim = memory_dim
        self.memory_module = HiPPOStyleMemory(hid_dim, memory_dim)
        self.register_buffer("internal_memory", None)  # tracked safely on the right device

    def forward(self, x, edge_index, edge_weight, prev_hidden_state):
        h = self.layer(x, edge_index, edge_weight, prev_hidden_state)
        h = self.activ(h)

        # Init memory if needed
        if self.internal_memory is None or self.internal_memory.size(0) != h.size(0):
            self.internal_memory = torch.zeros(h.size(0), self.memory_dim, device=h.device)

        # Update memory safely with detached tensors to avoid backward-through-time issues
        self.internal_memory = self.memory_module(self.internal_memory.detach(), h.detach())

        # Combine GCN output with memory
        combined = torch.cat([h, self.internal_memory], dim=-1)
        y = self.out_layer(combined)

        return y, h



class RecurrentGCN_TGCN_with_attention_conv(nn.Module):
    def __init__(self, node_features, hid_dim, out_dim, memory_len=4):
        super().__init__()
        self.layer = TGCN(node_features, hid_dim)
        self.activ = F.relu
        self.out_layer = nn.Linear(hid_dim, out_dim)

        self.memory_len = memory_len
        self.hidden_dim = hid_dim

        # Self-attention projections
        self.proj_q = nn.Linear(hid_dim, hid_dim)
        self.proj_k = nn.Linear(hid_dim, hid_dim)
        self.proj_v = nn.Linear(hid_dim, hid_dim)

        # Temporal 1D conv
        # Define layer
        self.conv1d = nn.Conv1d(in_channels=hid_dim, out_channels=hid_dim, kernel_size=3, padding=1)

    def forward(self, x, edge_index, edge_weight, prev_hidden_state):
        """
        prev_hidden_state: shape [B, k, D] or None on first step
        """
        # === Compute current hidden state from TGCN ===
        h = self.layer(x, edge_index, edge_weight, None)  # [B, D]

        # === Handle initial case ===
        B, D = h.shape
        if prev_hidden_state is None:
            prev_hidden_state = torch.zeros(B, self.memory_len - 1, D, device=h.device)

        # === Append current h to past memory ===
        memory_tensor = torch.cat([prev_hidden_state, h.unsqueeze(1)], dim=1)  # [B, k, D]

        # === Self-attention over time ===
        q = self.proj_q(h).unsqueeze(1)       # [B, 1, D]
        k = self.proj_k(memory_tensor)        # [B, k, D]
        v = self.proj_v(memory_tensor)        # [B, k, D]
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.hidden_dim)  # [B, 1, k]
        attn_weights = torch.softmax(attn_scores, dim=-1)                     # [B, 1, k]
        attn_output = (attn_weights @ v).squeeze(1)                           # [B, D]

        # === Temporal trend via 1D conv ===
        # === Temporal trend via 1D conv ===
        # In forward
        conv_input = memory_tensor.transpose(1, 2)  # [B, D, k]
        conv_out = self.conv1d(conv_input).transpose(1, 2)  # [B, k, D]
        conv_out = conv_out[:, -1, :]  # take most recent time step

        # === Combine outputs ===
        enriched = h + attn_output + conv_out  # [B, D]
        x = self.activ(enriched)
        y = self.out_layer(x)  # [B, out_dim]

        # === Update memory: remove oldest and append current h ===
        updated_memory = torch.cat([memory_tensor[:, 1:], h.unsqueeze(1)], dim=1)  # [B, k, D]

        return y, updated_memory



class RecurrentGCN_TGCN_GRU_with_memory(nn.Module):
    def __init__(self, node_features, hid_dim, out_dim, memory_len=4):
        super().__init__()
        self.hid_dim = hid_dim
        self.memory_len = memory_len

        self.layer = TGCN(node_features, hid_dim)
        self.out_layer = nn.Linear(hid_dim, out_dim)
        self.activ = F.relu

        self.attn_vector = nn.Parameter(torch.randn(hid_dim))

    def forward(self, x, edge_index, edge_weight, h_prev, memory_bank):
        """
        Inputs:
            x             : [B, in_dim]
            edge_index    : [2, E]
            edge_weight   : [E]
            h_prev        : [B, D]        # last GRU hidden state
            memory_bank   : [B, k-1, D] or None

        Outputs:
            y             : [B, out_dim]
            h             : [B, D]        # new hidden state
            memory_bank   : [B, k-1, D]   # updated memory bank
        """

        B = x.size(0)

        # === Step 1: Run TGCN with GRU
        h = self.layer(x, edge_index, edge_weight, h_prev)  # [B, D]

        # === Step 2: Init memory if needed
        if memory_bank is None:
            memory_bank = torch.zeros(B, self.memory_len - 1, self.hid_dim, device=x.device)

        # === Step 3: Form full memory: [B, k, D]
        full_memory = torch.cat([memory_bank, h.unsqueeze(1)], dim=1)

        # === Step 4: Attention over memory
        attn_scores = full_memory @ self.attn_vector  # [B, k]
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # [B, k, 1]
        h_attn = (full_memory * attn_weights).sum(dim=1)  # [B, D]

        # === Step 5: Output
        out = self.activ(h + h_attn)
        y = self.out_layer(out)

        # === Step 6: Update memory (drop oldest)
        updated_memory_bank = torch.cat([memory_bank[:, 1:], h.unsqueeze(1)], dim=1)

        return y, h, updated_memory_bank


# pip install


# class RecurrentGCN_TGCN_SRC(nn.Module):
#     """
#     TGCN -> SRC pooling (TopK/SAG/MinCut/DiffPool) -> global readout -> head
#     """
#     def __init__(self, node_features, hid_dim, out_dim,
#                  pool_type: str = "sag",   # "topk" | "sag" | "mincut" | "diffpool"
#                  pool_ratio: float = 0.5):
#         super().__init__()
#         self.layer = TGCN(node_features, hid_dim)
#         self.activ = F.relu
#
#         # --- SRC pooling (drop-in) ---
#         self.pool = get_pooler(pool_type, in_channels=hid_dim, ratio=pool_ratio)
#
#         # Final graph readout head (after pooling + global pooling)
#         self.out_layer = nn.Linear(hid_dim, out_dim)
#
#     def forward(self, x, edge_index, edge_weight, prev_hidden_state, batch=None):
#         """
#         x            : [N, Fin]
#         edge_index   : [2, E]
#         edge_weight  : [E] or None
#         prev_hidden_state: [N, D] previous hidden for TGCN
#         batch        : [N] (zeros if single graph)
#         returns:
#             y_graph        : [B, out_dim] graph-level logits
#             next_hidden    : [N, D] updated hidden state before pooling
#             (optional) pooled graph parts if you want to inspect them
#         """
#         # Recurrent TGCN update (node-level)
#         next_hidden = self.layer(x, edge_index, edge_weight, prev_hidden_state)  # [N, D]
#         h = self.activ(next_hidden)
#
#         # Fabricate batch if you pass one graph at a time
#         if batch is None:
#             batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
#
#         # --- SRC: Select → Reduce → Connect on h ---
#         pout = self.pool(h, edge_index, edge_weight=edge_weight, batch=batch)
#         h_pooled, ei_pooled, ew_pooled, batch_pooled = pout.x, pout.edge_index, pout.edge_weight, pout.batch
#         # (pout may also include selection info at pout.so depending on pooler)
#
#         # Global readout on pooled graph (you can swap mean for attention/Set2Set if you like)
#         g = global_mean_pool(h_pooled, batch_pooled)  # [B, D]
#
#         # Graph-level head
#         y_graph = self.out_layer(g)  # [B, out_dim]
#
#         # Return graph-level prediction and the pre-pool hidden state (so you can keep your recurrence)
#         return y_graph, next_hidden


class RecurrentGCN_TGCN_SRC(nn.Module):
    """
    (TGCN x num_layers, chained) -> SRC pooling -> global readout -> head

    Args:
        node_features (int): input feature size (Fin)
        hid_dim (int): hidden size (D)
        out_dim (int): output size
        num_layers (int): number of stacked TGCN layers (>=1)
        pool_type (str): "topk" | "sag" | "mincut" | "diffpool"
        pool_ratio (float): pooling ratio

    Forward:
        x: [N, Fin]
        edge_index: [2, E]
        edge_weight: [E] or None
        prev_hidden_state: [N, D] (single hidden; will be threaded through depth)
        batch: [N] or None

    Returns:
        y_graph: [B, out_dim]
        next_hidden: [N, D]  # from the LAST TGCN layer
    """
    def __init__(self, node_features, hid_dim, out_dim,
                 num_layers: int = 1,
                 pool_type: str = "sag",
                 pool_ratio: float = 0.5):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"

        # Stack of TGCN layers: Fin->D then (D->D) x (L-1)
        layers = [TGCN(node_features, hid_dim)]
        for _ in range(num_layers - 1):
            layers.append(TGCN(hid_dim, hid_dim))
        self.layers = nn.ModuleList(layers)

        self.activ = F.relu
        self.pool = get_pooler(pool_type, in_channels=hid_dim, ratio=pool_ratio)
        self.out_layer = nn.Linear(hid_dim, out_dim)

        self.hid_dim = hid_dim
        self.num_layers = num_layers

    def _zeros_hidden(self, N, device):
        return torch.zeros(N, self.hid_dim, device=device)

    def forward(self, x, edge_index, edge_weight, prev_hidden_state, batch=None):
        N, device = x.size(0), x.device

        # Ensure we have a single hidden state to start the depth chain
        if prev_hidden_state is None:
            h_prev = self._zeros_hidden(N, device)     # [N, D]
        else:
            assert prev_hidden_state.shape == (N, self.hid_dim), \
                f"prev_hidden_state must be [N, {self.hid_dim}]"
            h_prev = prev_hidden_state

        h_cur = x
        # Thread one hidden through the TGCN stack (depth-wise recurrence)
        for l, tgcn in enumerate(self.layers):
            h_next = tgcn(h_cur, edge_index, edge_weight, h_prev)  # [N, D]
            # Activation between layers (prepares features for next layer)
            h_cur = self.activ(h_next)
            # Pass the just-produced hidden to the next depth step
            h_prev = h_next

        next_hidden = h_prev  # single tensor from the last TGCN

        # Fabricate batch if single-graph
        if batch is None:
            batch = torch.zeros(N, dtype=torch.long, device=device)

        # SRC pooling on final representation
        pout = self.pool(h_cur, edge_index, edge_weight=edge_weight, batch=batch)
        h_pooled, batch_pooled = pout.x, pout.batch

        # Global readout & head (unchanged)
        g = global_mean_pool(h_pooled, batch_pooled)  # [B, D]
        y_graph = self.out_layer(g)                   # [B, out_dim]

        return y_graph, next_hidden
