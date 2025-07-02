import torch
import networkx as nx

from Libs.Utils.gc_graph import tensor_to_digraph


def test_tensor_to_digraph_basic():
    """Ensure conversion returns expected nodes/edges and labels."""
    GC = torch.tensor([
        [0.0, 0.8, 0.0],
        [0.2, 0.0, 0.4],
        [0.0, 0.0, 0.0],
    ])
    labels = ["A", "B", "C"]
    G = tensor_to_digraph(GC, threshold=0.1, labels=labels)

    assert isinstance(G, nx.DiGraph)
    assert G.number_of_nodes() == 3

    # Edges above threshold: (0->1, 1->0, 1->2)
    expected_edges = {(0, 1), (1, 0), (1, 2)}
    assert set(G.edges()) == expected_edges

    # Check label attribute
    for i, lab in enumerate(labels):
        assert G.nodes[i]["label"] == lab 