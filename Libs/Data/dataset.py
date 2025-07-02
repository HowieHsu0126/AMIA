"""Dataset helpers for NGC-AKI time-series models.

This module centralises all logic for **loading** the processed tensor produced
by :pyfunc:`Libs.run_pipeline.stage_preprocess` and exposing it as either a raw
``torch.Tensor`` or a streaming :class:`torch.utils.data.Dataset` suitable for
mini-batch training.

Why another wrapper?
--------------------
Keeping dataset I/O in a single location avoids peppering ``torch.load`` calls
throughout the codebase and makes it easier to switch to memory-mapped formats
(NPY, Parquet, Zarr, etc.) later.

Public API
~~~~~~~~~~
load_tensor_dataset() – one-shot load returning a ``torch.Tensor``
TimeSeriesTensorDataset – PyTorch Dataset wrapper around the tensor file
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

__all__ = [
    "load_tensor_dataset",
    "TimeSeriesTensorDataset",
]


def load_tensor_dataset(tensor_path: str | Path, *, device: torch.device | str | None = None) -> torch.Tensor:
    """Load pre-processed tensor saved by the preprocessing stage.

    Parameters
    ----------
    tensor_path:
        Path to ``tensor.pt`` produced by the pipeline.
    device:
        Target device to move the tensor to. If *None* leave on CPU.

    Returns
    -------
    torch.Tensor
        Shape ``(N, horizon, p)`` – identical to original save.
    """
    tensor_path = Path(tensor_path).expanduser()
    if not tensor_path.is_file():
        raise FileNotFoundError(tensor_path)

    X = torch.load(tensor_path)
    if device is not None:
        X = X.to(device)
    return X


class TimeSeriesTensorDataset(Dataset):
    """Thin ``Dataset`` wrapper around the pre-processed 3-D tensor.

    The class enables sampling mini-batches for optimisers like Adam while still
    allowing full-dataset operations for deterministic algorithms such as
    GISTA.
    """

    def __init__(self, tensor: torch.Tensor):
        if tensor.ndim != 3:
            raise ValueError(
                "Expected tensor of shape (N, horizon, p), got %s" % (tensor.shape,))
        self.X = tensor

    def __len__(self) -> int:  # noqa: D401
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:  # noqa: D401
        return self.X[idx]
