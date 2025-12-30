import torch
import torch.nn as nn

class MultiheadAttention(nn.Module):
    """Implement multi-head attention mechanism for practicing"""
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        """Initialize the multi-head attention module

        Args:
            embed_dim: Dimension of the input embeddings
            num_heads: Number of attention heads
            dropout: Dropout rate to apply on attention weights
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.head_dim = embed_dim // num_heads

        # Assert embed_dim is divisible by num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        # Define the projection layers for query, key, and value
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        
        # Output projection layer (standard in transformers)
        self.W_o = nn.Linear(embed_dim, embed_dim)

        """Consider implement deepseek-style attention later"""
    
    def forward(self, query, key, value, mask=None):
        """forward
        Args:
            query: Query tensor of shape (batch_size, seq_len, embed_dim)
            key: Key tensor of shape (batch_size, seq_len, embed_dim)
            value: Value tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional mask tensor to prevent attention to certain positions
        """
        # k, q, v (batch_size, seq_len, embed_dim)
        k = self.W_k(key) # k: (batch_size, seq_len, embed_dim)
        q = self.W_q(query) # q: (batch_size, seq_len, embed_dim)
        v = self.W_v(value) # v: (batch_size, seq_len, embed_dim)

        """Transform k, q, v to multiple heads 
        => k: (batch_size, seq_len, embed_dim)
        => k: (batch_size, seq_len, num_heads, head_dim)
        => k: (batch_size, num_heads, seq_len, head_dim)

        k, q, v: (batch_size, num_heads, seq_len, head_dim)
        """
        k = k.view(k.size(0), k.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        q = q.view(q.size(0), q.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        # score_ij = (q_i dot k_j) / sqrt(d_k)
        score = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5) # score: (batch_size, num_heads, seq_len, seq_len)
        if mask is not None:
            score = score.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(score, dim=-1) # attn_weights: (batch_size, num_heads, seq_len, seq_len)
        attn_weights = self.dropout(attn_weights)

        # output_i = sum over (attn_weights_ij * v_j)
        output = attn_weights @ v # output: (batch_size, number_of_heads, seq_len, head_dim)
        output = self.concat_heads(output)  # output: (batch_size, seq_len, embed_dim)
        return self.W_o(output)  # Apply output projection

    def concat_heads(self, tensor):
        """Concatenate multiple heads back to original embedding dimension
        Args:
            tensor: Tensor of shape (batch_size, num_heads, seq_len, head_dim)
        Returns:
            Tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, number_of_heads, seq_len, head_dim = tensor.size()
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, number_of_heads * head_dim)
        return tensor

        
