"""
Deep Learning Document Classifier (PyTorch Neural Attention / Transformer)
Implements an end-to-end PyTorch neural document classifier with embedding, multi-head self-attention,
and a dense classification head. Computes calibrated softmax distributions and per-class logits.
"""

import time
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from documind.core.config import settings
from documind.extraction.schema import DocumentClassificationResult


class SimpleTokenizer:
    """
    Lightweight, deterministic subword/word tokenizer with fixed vocabulary.
    """
    def __init__(self, max_vocab_size: int = 8000, max_len: int = 512):
        self.max_vocab_size = max_vocab_size
        self.max_len = max_len
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.vocab: Dict[str, int] = {self.pad_token: 0, self.unk_token: 1}
        self.inv_vocab: Dict[int, str] = {0: self.pad_token, 1: self.unk_token}

    def build_vocab(self, texts: List[str]):
        counts: Dict[str, int] = {}
        for text in texts:
            for token in text.lower().replace("\n", " ").split():
                clean_tok = "".join(ch for ch in token if ch.isalnum() or ch in "-_.$")
                if clean_tok:
                    counts[clean_tok] = counts.get(clean_tok, 0) + 1

        sorted_tokens = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        for token, _ in sorted_tokens[: self.max_vocab_size - 2]:
            idx = len(self.vocab)
            self.vocab[token] = idx
            self.inv_vocab[idx] = token

    def encode(self, text: str) -> List[int]:
        tokens = text.lower().replace("\n", " ").split()
        encoded = []
        for t in tokens[: self.max_len]:
            clean_tok = "".join(ch for ch in t if ch.isalnum() or ch in "-_.$")
            encoded.append(self.vocab.get(clean_tok, self.vocab[self.unk_token]))

        # Pad to max_len
        if len(encoded) < self.max_len:
            encoded += [self.vocab[self.pad_token]] * (self.max_len - len(encoded))
        return encoded


class DocumentDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: SimpleTokenizer):
        self.tokenizer = tokenizer
        self.data = [torch.tensor(tokenizer.encode(t), dtype=torch.long) for t in texts]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class NeuralDocumentTransformer(nn.Module):
    """
    Multi-Head Attention Neural Document Classifier in PyTorch.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 128, num_heads: int = 4, num_classes: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoder = nn.Parameter(torch.randn(1, 512, embed_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        seq_len = x.size(1)
        emb = self.embedding(x) + self.pos_encoder[:, :seq_len, :]
        # Mask out padding tokens
        padding_mask = (x == 0)
        feat = self.transformer(emb, src_key_padding_mask=padding_mask)
        # Global average pooling over non-padding tokens
        mask_expanded = (~padding_mask).unsqueeze(-1).float()
        pooled = (feat * mask_expanded).sum(dim=1) / torch.clamp(mask_expanded.sum(dim=1), min=1.0)
        logits = self.fc(pooled)
        return logits


class DeepLearningDocumentClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() and settings.USE_CUDA_IF_AVAILABLE else "cpu")
        self.categories = settings.CATEGORIES
        self.label_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        self.idx_to_label = {idx: cat for cat, idx in self.label_to_idx.items()}
        
        self.tokenizer = SimpleTokenizer(max_vocab_size=8000, max_len=256)
        self.model: nn.Module = None
        self.is_trained: bool = False

    def train(self, texts: List[str], labels: List[str], epochs: int = 12, batch_size: int = 16, lr: float = 1e-3):
        """
        Trains the PyTorch Neural Transformer classifier.
        """
        self.tokenizer.build_vocab(texts)
        int_labels = [self.label_to_idx[l] for l in labels]
        
        dataset = DocumentDataset(texts, int_labels, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model = NeuralDocumentTransformer(
            vocab_size=len(self.tokenizer.vocab),
            embed_dim=128,
            num_heads=4,
            num_classes=len(self.categories),
            hidden_dim=256
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        self.model.train()
        for epoch in range(epochs):
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                
        self.is_trained = True

    def predict(self, text: str) -> DocumentClassificationResult:
        """
        Inference with latency measurement and softmax confidence scores.
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Deep Learning model has not been trained or loaded yet.")
            
        t0 = time.time()
        self.model.eval()
        with torch.no_grad():
            encoded = self.tokenizer.encode(text)
            input_tensor = torch.tensor([encoded], dtype=torch.long).to(self.device)
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
        latency_ms = round((time.time() - t0) * 1000.0, 2)
        
        prob_dict = {
            self.idx_to_label[i]: float(round(probs[i], 4))
            for i in range(len(self.categories))
        }
        best_idx = int(probs.argmax())
        best_cat = self.idx_to_label[best_idx]
        best_conf = float(round(probs[best_idx], 4))
        
        return DocumentClassificationResult(
            predicted_category=best_cat,
            confidence=best_conf,
            probabilities=prob_dict,
            model_used="deep_learning_transformer",
            inference_time_ms=latency_ms
        )

    def save(self, model_path: Path = settings.DL_MODEL_PATH):
        """
        Saves weights and tokenizer vocabulary.
        """
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "vocab": self.tokenizer.vocab,
            "categories": self.categories
        }, model_path)

    def load(self, model_path: Path = settings.DL_MODEL_PATH) -> bool:
        """
        Loads weights and reconstructs the model.
        """
        if not model_path.exists():
            return False
            
        checkpoint = torch.load(model_path, map_location=self.device)
        self.categories = checkpoint["categories"]
        self.tokenizer.vocab = checkpoint["vocab"]
        self.tokenizer.inv_vocab = {v: k for k, v in self.tokenizer.vocab.items()}
        
        self.model = NeuralDocumentTransformer(
            vocab_size=len(self.tokenizer.vocab),
            embed_dim=128,
            num_heads=4,
            num_classes=len(self.categories),
            hidden_dim=256
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.is_trained = True
        return True
