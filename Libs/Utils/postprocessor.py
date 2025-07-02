"""Post-processing utilities for visualising learned Granger graphs.

Currently provides *GrootRank* (PageRank variant) bar-plot export based on the
example notebook supplied by the user.  The plot is saved as PDF so that it can
be embedded in reports without rasterisation artefacts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import logging

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "compute_grootrank",
    "plot_grootrank",
]


def _load_graph(graph_path: str | Path) -> nx.DiGraph:
    graph_path = Path(graph_path).expanduser()
    if not graph_path.is_file():
        raise FileNotFoundError(graph_path)
    return nx.read_graphml(graph_path)


def compute_grootrank(graph_path: str | Path) -> pd.DataFrame:
    """Compute personalised PageRank (dubbed *GrootRank*) for each node.

    The personalisation follows the user-provided rule: dangling nodes get 1.0
    while all others receive 0.5.
    """
    G = _load_graph(graph_path)

    # Identify dangling nodes (no outgoing edges) ---------------------------
    dangling = {n for n, out_deg in G.out_degree() if out_deg == 0}
    pers = {n: (1.0 if n in dangling else 0.5) for n in G.nodes()}

    rank = nx.pagerank(G, personalization=pers)

    # Construct tidy DataFrame ---------------------------------------------
    labels = {
        n: (data.get("label") if isinstance(data, dict) else str(n))
        for n, data in G.nodes(data=True)
    }
    df = (
        pd.DataFrame(rank.items(), columns=["Node", "GrootRank"])  # type: ignore[arg-type]
        .assign(Label=lambda d: d["Node"].map(labels))
        .sort_values("GrootRank", ascending=False, ignore_index=True)
    )
    return df


def plot_grootrank(
    graph_path: str | Path,
    *,
    output_pdf: str | Path | None = None,
    top_k: int = 10,
) -> Path:
    """Generate a PDF bar plot for the top-*k* GrootRank nodes.

    Parameters
    ----------
    graph_path: Path to ``.graphml`` file.
    output_pdf: Destination; defaults to sibling file
        ``GrootRank_top{top_k}.pdf`` under same directory.
    top_k: Number of highest-ranked nodes to plot.
    """
    df = compute_grootrank(graph_path)
    top = df.head(top_k)

    # Matplotlib/SciencePlots styling --------------------------------------
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "no-latex"])
    except ImportError:
        plt.style.use("default")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(top["Label"], top["GrootRank"], color="#4C72B0")
    ax.set_xlabel("Node", fontsize=12)
    ax.set_ylabel("PageRank Value", fontsize=12)
    ax.set_title("Top GrootRank Nodes", fontsize=14)
    ax.set_xticklabels(top["Label"], rotation=45, ha="right", fontsize=10)
    fig.tight_layout()

    graph_path = Path(graph_path)
    if output_pdf is None:
        output_pdf = graph_path.with_name(f"GrootRank_top{top_k}.pdf")
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_pdf, format="pdf")
    plt.close(fig)

    logger.info("GrootRank plot saved ➜ %s", output_pdf)
    return output_pdf 