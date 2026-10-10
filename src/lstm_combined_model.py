"""Ablation variant: template-id embedding (learned) + frozen semantic
template embedding (raw-text proxy), concatenated and projected down to
the same width as the LSTM's embedding dim, then the rest of the
architecture is identical to DeepLogLSTM.

This is one half of the ablation (the other half is the plain
DeepLogLSTM — "template-only"). The raw-text-only variant from the
original 3-variant plan was dropped as an explicit scoping decision
earlier in the project.
"""
import numpy as np
import torch
import torch.nn as nn


class CombinedLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        semantic_embeddings: np.ndarray,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)

        semantic_dim = semantic_embeddings.shape[1]
        # Frozen — this is a fixed proxy for "what the raw text says", not
        # something the task should be allowed to reshape via gradient.
        self.semantic_embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(semantic_embeddings).float(), freeze=True, padding_idx=0
        )
        self.project = nn.Linear(embed_dim + semantic_dim, embed_dim)

        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_emb = self.token_embedding(x)
        sem_emb = self.semantic_embedding(x)
        combined = self.project(torch.cat([token_emb, sem_emb], dim=-1))
        out, (h_n, _) = self.lstm(combined)
        last_hidden = h_n[-1]
        return self.fc(last_hidden)
