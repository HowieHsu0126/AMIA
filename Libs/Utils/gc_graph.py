"""Utility helpers for exporting learned Granger–Causality matrices.

This module contains convenience functions that load the `GC_matrix.pt`
artifact produced by ``Libs.run_pipeline`` and export a directed NetworkX
``DiGraph`` in GraphML format for downstream visualisation in Gephi / Cytoscape
or Python notebooks.

Example
-------
>>> from pathlib import Path
>>> from Libs.Utils.gc_graph import export_graphml
>>> export_graphml(Path("Output/GC_matrix.pt"), output_dir=Path("Output/Results"))
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import torch
import networkx as nx

__all__ = [
    "tensor_to_digraph",
    "export_graphml",
]

def tensor_to_digraph(GC: torch.Tensor, *, threshold: float | None = 1e-3, labels: Sequence[str] | None = None) -> nx.DiGraph:  # noqa: N802
    """Convert a PyTorch GC adjacency tensor to a ``networkx.DiGraph``.

    Parameters
    ----------
    GC:
        Tensor of shape *(p, p)* or *(p, p, lag)*. Values are typically L2
        norms of first-layer filters; higher means stronger causality.
    threshold:
        If *None* use raw edge weights (no pruning). Otherwise keep edges with
        value strictly greater than *threshold*.
    labels:
        Optional list/sequence of human-readable node labels with length *p*.
        If *None* numeric indices are used.

    Returns
    -------
    nx.DiGraph
        Directed graph representing Granger causality.
    """
    if GC.ndim == 3:
        # Collapse lags by L2-norm across last axis
        GC = torch.norm(GC, dim=-1)
    elif GC.ndim != 2:
        raise ValueError("GC tensor must have 2 or 3 dimensions, got %s" % (GC.ndim,))

    p = GC.shape[0]
    if GC.shape[0] != GC.shape[1]:
        raise ValueError("GC tensor must be square (p, p)")

    if labels is not None and len(labels) != p:
        raise ValueError("Length of labels (%d) must match tensor size (%d)" % (len(labels), p))

    G = nx.DiGraph()
    idx_to_label = {i: (labels[i] if labels else str(i)) for i in range(p)}

    # Add nodes first with label attribute
    for i in range(p):
        G.add_node(i, label=idx_to_label[i])

    # Add directed edges
    GC_np = GC.cpu().detach().numpy()
    for i in range(p):
        for j in range(p):
            if i == j:
                continue  # skip self-loops
            weight = float(GC_np[i, j])
            if threshold is None or weight > threshold:
                G.add_edge(i, j, weight=weight)

    return G

def export_graphml(
    gc_matrix_path: str | Path,
    *,
    output_dir: str | Path = "Output/Results",
    threshold: float | None = 1e-3,
    labels: Iterable[str] | None = None,
    graphml_name: str | None = None,
) -> Path:
    """Export GC graph to *GraphML* file.

    Parameters
    ----------
    gc_matrix_path:
        Path to ``GC_matrix.pt`` saved by the training stage.
    output_dir:
        Directory where the ``.graphml`` file is written. Created if missing.
    threshold:
        Edge-weight cutoff. Use *None* to keep all edges.
    labels:
        Optional iterable of node labels.
    graphml_name:
        Override default filename. If *None*, automatically derived from
        *threshold*.

    Returns
    -------
    pathlib.Path
        Full path to the generated GraphML file.
    """
    gc_matrix_path = Path(gc_matrix_path).expanduser()
    if not gc_matrix_path.is_file():
        raise FileNotFoundError(gc_matrix_path)

    GC = torch.load(gc_matrix_path)
    if not isinstance(GC, torch.Tensor):
        raise TypeError("Loaded GC matrix is not a torch.Tensor: %r" % type(GC))

    G = tensor_to_digraph(GC, threshold=threshold, labels=list(labels) if labels else None)

    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if graphml_name is None:
        thr_str = "all" if threshold is None else f"thr{threshold}"
        graphml_name = f"Granger_Causality_Graph_{thr_str}.graphml"

    graphml_path = output_dir / graphml_name
    nx.write_graphml(G, graphml_path)

    return graphml_path 