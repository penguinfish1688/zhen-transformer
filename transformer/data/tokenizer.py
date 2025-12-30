"""
Tokenizer for Chinese-English Translation

This module provides tokenizers for both Chinese and English text.
- Chinese: Uses jieba for word segmentation
- English: Uses basic tokenization with optional subword support
"""
import re
from typing import List, Dict, Optional, Tuple
from collections import Counter
import json
import os


class ChineseTokenizer:
    """
    Chinese tokenizer using jieba for word segmentation
    """
    
    def __init__(self):
        try:
            import jieba
            self.jieba = jieba
            # Disable jieba logging
            jieba.setLogLevel(jieba.logging.INFO)
        except ImportError:
            print("⚠️  jieba not installed. Install with: pip install jieba")
            self.jieba = None
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize Chinese text into words
        
        Args:
            text: Chinese text string
        Returns:
            List of Chinese tokens
        """
        if self.jieba is None:
            # Fallback: character-level tokenization
            return list(text.replace(" ", ""))
        
        # Use jieba for word segmentation
        tokens = list(self.jieba.cut(text))
        # Remove whitespace tokens
        tokens = [t.strip() for t in tokens if t.strip()]
        return tokens


class EnglishTokenizer:
    """
    Simple English tokenizer with basic preprocessing
    """
    
    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize English text
        
        Args:
            text: English text string
        Returns:
            List of English tokens
        """
        if self.lowercase:
            text = text.lower()
        
        # Basic tokenization: split on whitespace and punctuation
        # Keep punctuation as separate tokens
        text = re.sub(r"([.,!?;:'\"-])", r" \1 ", text)
        tokens = text.split()
        tokens = [t.strip() for t in tokens if t.strip()]
        return tokens


class Vocabulary:
    """
    Vocabulary class to map tokens to indices and vice versa
    """
    
    def __init__(self, pad_token: str = "<pad>", unk_token: str = "<unk>",
                 bos_token: str = "<bos>", eos_token: str = "<eos>"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        
        # Initialize with special tokens
        self.token2idx: Dict[str, int] = {}
        self.idx2token: Dict[int, str] = {}
        self.token_freq: Counter = Counter()
        
        # Add special tokens
        self._add_special_tokens()
    
    def _add_special_tokens(self):
        """Add special tokens to vocabulary"""
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for token in special_tokens:
            if token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token
    
    @property
    def pad_idx(self) -> int:
        return self.token2idx[self.pad_token]
    
    @property
    def unk_idx(self) -> int:
        return self.token2idx[self.unk_token]
    
    @property
    def bos_idx(self) -> int:
        return self.token2idx[self.bos_token]
    
    @property
    def eos_idx(self) -> int:
        return self.token2idx[self.eos_token]
    
    def __len__(self) -> int:
        return len(self.token2idx)
    
    def build_from_corpus(self, corpus: List[List[str]], min_freq: int = 1):
        """
        Build vocabulary from tokenized corpus
        
        Args:
            corpus: List of tokenized sentences (list of token lists)
            min_freq: Minimum frequency to include token
        """
        print(f"📊 Building vocabulary from {len(corpus)} sentences...")
        
        # Count all tokens
        for tokens in corpus:
            self.token_freq.update(tokens)
        
        # Add tokens meeting minimum frequency
        for token, freq in self.token_freq.items():
            if freq >= min_freq and token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token
        
        print(f"✅ Vocabulary built: {len(self)} tokens (min_freq={min_freq})")
    
    def encode(self, tokens: List[str], add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """
        Convert tokens to indices
        
        Args:
            tokens: List of tokens
            add_bos: Whether to add BOS token at start
            add_eos: Whether to add EOS token at end
        Returns:
            List of token indices
        """
        indices = []
        if add_bos:
            indices.append(self.bos_idx)
        
        for token in tokens:
            indices.append(self.token2idx.get(token, self.unk_idx))
        
        if add_eos:
            indices.append(self.eos_idx)
        
        return indices
    
    def decode(self, indices: List[int], remove_special: bool = True) -> List[str]:
        """
        Convert indices back to tokens
        
        Args:
            indices: List of token indices
            remove_special: Whether to remove special tokens
        Returns:
            List of tokens
        """
        special_indices = {self.pad_idx, self.bos_idx, self.eos_idx}
        tokens = []
        
        for idx in indices:
            if remove_special and idx in special_indices:
                continue
            token = self.idx2token.get(idx, self.unk_token)
            tokens.append(token)
        
        return tokens
    
    def save(self, path: str):
        """Save vocabulary to JSON file"""
        data = {
            "token2idx": self.token2idx,
            "special_tokens": {
                "pad": self.pad_token,
                "unk": self.unk_token,
                "bos": self.bos_token,
                "eos": self.eos_token
            }
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"💾 Vocabulary saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        """Load vocabulary from JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        vocab = cls(
            pad_token=data["special_tokens"]["pad"],
            unk_token=data["special_tokens"]["unk"],
            bos_token=data["special_tokens"]["bos"],
            eos_token=data["special_tokens"]["eos"]
        )
        vocab.token2idx = data["token2idx"]
        vocab.idx2token = {int(v): k for k, v in vocab.token2idx.items()}
        print(f"📂 Vocabulary loaded from {path}: {len(vocab)} tokens")
        return vocab


class TranslationTokenizer:
    """
    Combined tokenizer for Chinese-English translation
    """
    
    def __init__(self, config=None):
        self.chinese_tokenizer = ChineseTokenizer()
        self.english_tokenizer = EnglishTokenizer()
        
        if config:
            self.src_vocab = Vocabulary(
                config.pad_token, config.unk_token,
                config.bos_token, config.eos_token
            )
            self.tgt_vocab = Vocabulary(
                config.pad_token, config.unk_token,
                config.bos_token, config.eos_token
            )
        else:
            self.src_vocab = Vocabulary()
            self.tgt_vocab = Vocabulary()
    
    def tokenize_source(self, text: str) -> List[str]:
        """Tokenize Chinese source text"""
        return self.chinese_tokenizer.tokenize(text)
    
    def tokenize_target(self, text: str) -> List[str]:
        """Tokenize English target text"""
        return self.english_tokenizer.tokenize(text)
    
    def build_vocabularies(self, src_sentences: List[str], tgt_sentences: List[str], 
                          min_freq: int = 2):
        """
        Build vocabularies from parallel corpus
        
        Args:
            src_sentences: List of Chinese sentences
            tgt_sentences: List of English sentences
            min_freq: Minimum word frequency
        """
        print("\n" + "="*60)
        print("🔤 BUILDING VOCABULARIES")
        print("="*60)
        
        # Tokenize all sentences
        print("\n📝 Step 1: Tokenizing source sentences (Chinese)...")
        src_tokenized = [self.tokenize_source(s) for s in src_sentences]
        print(f"   ✅ Tokenized {len(src_tokenized)} Chinese sentences")
        print(f"   📊 Sample: {src_sentences[0][:50]}... → {src_tokenized[0][:10]}")
        
        print("\n📝 Step 2: Tokenizing target sentences (English)...")
        tgt_tokenized = [self.tokenize_target(s) for s in tgt_sentences]
        print(f"   ✅ Tokenized {len(tgt_tokenized)} English sentences")
        print(f"   📊 Sample: {tgt_sentences[0][:50]}... → {tgt_tokenized[0][:10]}")
        
        # Build vocabularies
        print("\n📚 Step 3: Building source vocabulary...")
        self.src_vocab.build_from_corpus(src_tokenized, min_freq)
        
        print("\n📚 Step 4: Building target vocabulary...")
        self.tgt_vocab.build_from_corpus(tgt_tokenized, min_freq)
        
        print("\n" + "="*60)
        print(f"✅ VOCABULARY BUILDING COMPLETE")
        print(f"   Source vocab size: {len(self.src_vocab)}")
        print(f"   Target vocab size: {len(self.tgt_vocab)}")
        print("="*60)
        
        return src_tokenized, tgt_tokenized
    
    def encode_pair(self, src_text: str, tgt_text: str) -> Tuple[List[int], List[int]]:
        """
        Encode a source-target pair
        
        Returns:
            Tuple of (source_ids, target_ids)
        """
        src_tokens = self.tokenize_source(src_text)
        tgt_tokens = self.tokenize_target(tgt_text)
        
        src_ids = self.src_vocab.encode(src_tokens)
        tgt_ids = self.tgt_vocab.encode(tgt_tokens)
        
        return src_ids, tgt_ids
    
    def decode_target(self, indices: List[int]) -> str:
        """Decode target indices back to text"""
        tokens = self.tgt_vocab.decode(indices)
        return " ".join(tokens)
    
    def save(self, directory: str):
        """Save both vocabularies"""
        os.makedirs(directory, exist_ok=True)
        self.src_vocab.save(os.path.join(directory, "src_vocab.json"))
        self.tgt_vocab.save(os.path.join(directory, "tgt_vocab.json"))
    
    @classmethod
    def load(cls, directory: str) -> "TranslationTokenizer":
        """Load tokenizer with saved vocabularies"""
        tokenizer = cls()
        tokenizer.src_vocab = Vocabulary.load(os.path.join(directory, "src_vocab.json"))
        tokenizer.tgt_vocab = Vocabulary.load(os.path.join(directory, "tgt_vocab.json"))
        return tokenizer
