"""Embedding + BiLSTM encoder with masked max and mean pooling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


class Encoder(nn.Module):
    """Encode token ids into a 2*hidden representation.

    Args:
        vocab_size: Number of embedding rows.
        emb_dim: Embedding width.
        hidden: Total BiLSTM output width, split evenly across directions.
        dropout: Dropout applied to the embedding output and before pooling.
        padding_idx: Id of the pad token.
        glove_path: Optional path to a GloVe-style text file (``token v1 ... v100``).
            The weight matrix is left as an nn.Parameter, so initialisation only.
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 100,
        hidden: int = 128,
        dropout: float = 0.3,
        padding_idx: int = 0,
        glove_path: str | None = None,
        vocab: dict[str, int] | None = None,
    ):
        super().__init__()
        if hidden % 2 != 0:
            raise ValueError("hidden must be even so both directions have equal width")
        # ``hidden`` is the total BiLSTM width (128 by default), so each
        # direction has hidden/2 units and pooling yields 2*hidden = 256.
        per_direction = hidden // 2
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.embed_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            emb_dim,
            per_direction,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.output_dropout = nn.Dropout(dropout)
        self.output_dim = 2 * hidden
        if glove_path:
            if vocab is None:
                raise ValueError("glove_path requires the vocabulary to align ids")
            self.load_glove(glove_path, vocab, emb_dim)

    def load_glove(self, path: str | Path, vocab: dict[str, int], emb_dim: int) -> int:
        """Initialise embeddings from a GloVe text file. Returns matched rows."""
        matched = 0
        with Path(path).open("r", encoding="utf-8", errors="ignore") as handle, torch.no_grad():
            for line in handle:
                parts = line.rstrip().split(" ")
                if len(parts) != emb_dim + 1:
                    continue
                index = vocab.get(parts[0])
                if index is None:
                    continue
                self.embedding.weight[index] = torch.from_numpy(
                    np.asarray(parts[1:], dtype=np.float32)
                )
                matched += 1
        if matched == 0:
            raise ValueError(f"no vocabulary tokens matched GloVe vectors in {path}")
        return matched

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return the pooled representation ``z`` of shape ``(batch, 2*hidden)``."""
        embedded = self.embed_dropout(self.embedding(ids))
        outputs, _ = self.lstm(embedded)
        outputs = self.output_dropout(outputs)
        mask_f = mask.unsqueeze(-1)
        lengths = mask_f.sum(dim=1).clamp(min=1.0)
        mean_pool = (outputs * mask_f).sum(dim=1) / lengths
        neg_inf = torch.finfo(outputs.dtype).min
        max_pool = outputs.masked_fill(mask_f == 0, neg_inf).max(dim=1).values
        return torch.cat([mean_pool, max_pool], dim=-1)
