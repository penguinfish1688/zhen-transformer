"""
Configuration for Chinese-English Translation
"""
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class TranslationConfig:
    """Configuration for translation model and training"""
    
    # Model architecture
    embed_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    max_len: int = 256
    dropout: float = 0.1
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10
    warmup_steps: int = 4000
    
    # Data
    min_freq: int = 2  # Minimum word frequency to include in vocabulary
    max_samples: Optional[int] = None  # Limit samples for testing (None = all)
    
    # Tokenization
    src_lang: str = "zh"  # Chinese
    tgt_lang: str = "en"  # English
    
    # Special tokens
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    bos_token: str = "<bos>"  # Beginning of sentence
    eos_token: str = "<eos>"  # End of sentence
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paths
    data_dir: str = "data/translation"
    checkpoint_dir: str = "checkpoints"
    
    def __post_init__(self):
        """Ensure directories exist"""
        import os
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
