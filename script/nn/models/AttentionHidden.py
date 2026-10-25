import torch
import torch.nn as nn


class AttentionHidden(nn.Module):
    def __init__(self, input_dim, memory_size):
        """
        Attention-based DOS calculation with Layer Normalization.

        Args:
        - input_dim (int): Dimension of each DOS vector.
        - memory_size (int): Number of past snapshots to consider.
        """
        super(AttentionHidden, self).__init__()
        self.memory_size = memory_size
        self.input_dim = input_dim

        # Multi-head attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4)

        # Layer Normalization (works with batch size = 1)
        self.layer_norm = nn.LayerNorm(input_dim)

        # Linear layer to process attended features
        self.fc_out = nn.Linear(input_dim, input_dim)

    def forward(self, hidden_memory, current_hidden):
        """
        Computes attention-weighted DOS using memory.

        Args:
        - dos_memory: (batch_size, memory_size, feature_dim) -> Past DOS vectors.
        - current_dos: (batch_size, feature_dim) -> Current DOS vector.

        Returns:
        - dos_final: (batch_size, feature_dim) -> Attention-weighted DOS.
        """
        # Concatenate past DOS memory with the current DOS vector
        hidden_input = torch.cat([hidden_memory.squeeze(0), current_hidden.unsqueeze(0)])  # Shape: (batch_size, memory_size+1, feature_dim)

        # Apply multi-head self-attention
        attn_output, _ = self.attention(hidden_input, hidden_input, hidden_input)

        # Aggregate attention outputs (mean pooling over time dimension)
        hidden_final = attn_output.mean(dim=0)  # Shape: (batch_size, feature_dim)

        # Apply Layer Normalization (works for any batch size)
        hidden_final = self.layer_norm(hidden_final.clone())

        return self.fc_out(hidden_final)  # Apply final transformation layer
