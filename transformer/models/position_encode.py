"""
Positional Encoding for Transformer
Implements sinusoidal positional encoding as described in "Attention is All You Need"
"""
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, embed_dim: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            embed_dim: Dimension of the model embeddings
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute the divisor term: 10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        # Apply sin to even indices and cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: (1, max_len, embed_dim)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter, but should be saved with model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    """Token Embedding layer that converts token IDs to dense vectors"""
    
    def __init__(self, vocab_size: int, embed_dim: int, padding_idx: int = 0):
        """
        Args:
            vocab_size: Size of the vocabulary
            embed_dim: Dimension of embeddings
            padding_idx: Index used for padding tokens
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.embed_dim = embed_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token IDs of shape (batch_size, seq_len)
        Returns:
            Embeddings of shape (batch_size, seq_len, embed_dim)
        """
        # Scale embeddings by sqrt(embed_dim) as in the original paper
        return self.embedding(x) * math.sqrt(self.embed_dim)


class TransformerEmbedding(nn.Module):
    """Combined Token Embedding + Positional Encoding"""
    
    def __init__(self, vocab_size: int, embed_dim: int, max_len: int = 5000, 
                 dropout: float = 0.1, padding_idx: int = 0):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embed_dim, padding_idx)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token IDs of shape (batch_size, seq_len)
        Returns:
            Embeddings with positional encoding of shape (batch_size, seq_len, embed_dim)
        """
        tok_emb = self.token_embedding(x)
        return self.positional_encoding(tok_emb)
