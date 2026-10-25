import torch
import torch.nn as nn

# without bacth norm

# class AttentionDOS(nn.Module):
#     def __init__(self, input_dim, memory_size):
#         """
#         Attention-based DOS calculation.
#
#         Args:
#         - input_dim (int): Dimension of each DOS vector.
#         - memory_size (int): Number of past snapshots to consider.
#         """
#         super(AttentionDOS, self).__init__()
#         self.memory_size = memory_size
#         self.input_dim = input_dim
#
#         # Multi-head attention mechanism
#         self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4)
#
#         # Linear layer to process attended features
#         self.fc_out = nn.Linear(input_dim, input_dim)
#
#     def forward(self, dos_memory, current_dos):
#         """
#         Computes attention-weighted DOS using memory.
#
#         Args:
#         - dos_memory: (batch_size, memory_size, feature_dim) -> Past DOS vectors.
#         - current_dos: (batch_size, feature_dim) -> Current DOS vector.
#
#         Returns:
#         - dos_final: (batch_size, feature_dim) -> Attention-weighted DOS.
#         """
#         # Concatenate past DOS memory with the current DOS vector
#         dos_input = torch.cat([dos_memory, current_dos.unsqueeze(1)],
#                               dim=1)  # Shape: (batch_size, memory_size+1, feature_dim)
#
#         # Apply multi-head self-attention
#         attn_output, _ = self.attention(dos_input, dos_input, dos_input)
#
#         # Aggregate attention outputs (mean pooling over time dimension)
#         dos_final = attn_output.mean(dim=1)  # Shape: (batch_size, feature_dim)
#
#         return self.fc_out(dos_final)  # Apply final transformation layer
#
#
#



class AttentionDOS(nn.Module):
    def __init__(self, input_dim, memory_size):
        """
        Attention-based DOS calculation with Layer Normalization.

        Args:
        - input_dim (int): Dimension of each DOS vector.
        - memory_size (int): Number of past snapshots to consider.
        """
        super(AttentionDOS, self).__init__()
        self.memory_size = memory_size
        self.input_dim = input_dim

        # Multi-head attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4)

        # Layer Normalization (works with batch size = 1)
        self.layer_norm = nn.LayerNorm(input_dim)

        # Linear layer to process attended features
        self.fc_out = nn.Linear(input_dim, input_dim)

    def forward(self, dos_memory, current_dos):
        """
        Computes attention-weighted DOS using memory.

        Args:
        - dos_memory: (batch_size, memory_size, feature_dim) -> Past DOS vectors.
        - current_dos: (batch_size, feature_dim) -> Current DOS vector.

        Returns:
        - dos_final: (batch_size, feature_dim) -> Attention-weighted DOS.
        """
        # Concatenate past DOS memory with the current DOS vector
        dos_input = torch.cat([dos_memory, current_dos.unsqueeze(1)],
                              dim=1)  # Shape: (batch_size, memory_size+1, feature_dim)

        # Apply multi-head self-attention
        attn_output, _ = self.attention(dos_input, dos_input, dos_input)

        # Aggregate attention outputs (mean pooling over time dimension)
        dos_final = attn_output.mean(dim=1)  # Shape: (batch_size, feature_dim)

        # Apply Layer Normalization (works for any batch size)
        dos_final = self.layer_norm(dos_final)

        return self.fc_out(dos_final)  # Apply final transformation layer
