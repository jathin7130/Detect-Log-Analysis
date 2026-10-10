"""Compact transformer next-event predictor (LogBERT/LogFormer-inspired).

Same task framing as the DeepLog LSTM (src/lstm_model.py): given a fixed
window of preceding event-template ids, predict the next one. The window is
short and fixed-length, so positions are embedded with a learned table
rather than a sinusoidal scheme, and the encoder attends bidirectionally
over the window (there's no causal constraint — the window is context, not
a sequence being generated token-by-token).

Deliberately "compact": the proposal's goal is reproducibility and a fair
comparison against the LSTM baseline, not replicating LogBERT/LogFormer's
full architecture.
"""
import torch
import torch.nn as nn


class CompactTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        window_size: int,
        embed_dim: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        # +1 because event ids are 1-indexed and 0 is reserved for padding.
        self.token_embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(window_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size + 1)
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window) of event ids
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        emb = self.token_embedding(x) + self.position_embedding(positions)
        encoded = self.encoder(emb)  # (batch, window, embed_dim)
        pooled = encoded.mean(dim=1)  # mean-pool over the window
        logits = self.fc(pooled)
        return logits
