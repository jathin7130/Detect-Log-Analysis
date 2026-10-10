"""DeepLog-style LSTM: predicts the next event-template id given a fixed-size
window of preceding event ids (Du et al., 2017).

Trained on normal sequences only — DeepLog's anomaly signal is "the real
next event wasn't one the model considered likely", which only means
something if the model learned what normal workflows look like.
"""
import torch
import torch.nn as nn


class DeepLogLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        # +1 because event ids are 1-indexed and 0 is reserved for padding.
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window) of event ids
        emb = self.embedding(x)
        out, (h_n, _) = self.lstm(emb)
        last_hidden = h_n[-1]  # (batch, hidden_dim) — final layer's last hidden state
        logits = self.fc(last_hidden)
        return logits


def build_windows(sequences: list[list[int]], window_size: int):
    """Slide a fixed window over each sequence, predicting the next event id.

    Sequences shorter than window_size+1 are skipped (not enough context to
    form even one (window, next) pair). Returns NumPy arrays, built with
    stride tricks instead of a pure-Python nested loop + list-of-lists —
    the latter is fine for thousands of windows but falls over (many
    minutes, single-threaded) once you're building millions of them, both
    in the loop itself and in the subsequent torch.tensor() conversion.
    """
    import numpy as np

    x_chunks, y_chunks = [], []
    for seq in sequences:
        n = len(seq)
        if n <= window_size:
            continue
        arr = np.asarray(seq, dtype=np.int64)
        n_windows = n - window_size
        # sliding_window_view gives a zero-copy (n_windows, window_size) view
        windows = np.lib.stride_tricks.sliding_window_view(arr, window_size)[:n_windows]
        x_chunks.append(windows)
        y_chunks.append(arr[window_size:window_size + n_windows])

    if not x_chunks:
        return np.empty((0, window_size), dtype=np.int64), np.empty((0,), dtype=np.int64)

    return np.concatenate(x_chunks, axis=0), np.concatenate(y_chunks, axis=0)
