from typing import List, Tuple
import networkx as nx
import numpy as np
import scipy.sparse as sp


def to_edge_adj(edgelist: List[Tuple[int, int]]) -> Tuple[List[List[int]], int]:
    """
    Create an "edge-adjacency" matrix Q of shape (num_edges, num_edges),
    where
        Q_(c, d),(a, b) :=
            - 1 if b == c and a != d
            - 0 else
    :param edgelist: list of edges in the form (source, target)
    :return: edge-adjacency matrix Q and the number of non-zero entries in Q
    """
    elen = len(edgelist)
    nodes = set()
    for source, target in edgelist:
        nodes.add(source)
        nodes.add(target)
    print(f'found {elen} edges')
    print(f'found {len(nodes)} nodes')
    num_pos = 0
    q = [[0] * elen for _ in range(elen)]
    for i in range(elen):
        for j in range(elen):
            c, d = edgelist[i]
            a, b = edgelist[j]
            if b == c and a != d:
                num_pos += 1
                q[i][j] = 1
            else:
                q[i][j] = 0

    print(f'non-zero entries in edge-adjacency matrix: {num_pos}')
    return np.array(q)


def create_heads_matrix(edgelist: List[Tuple[int, int]], graph: nx.DiGraph) -> np.array:
    degrees = [graph.degree(head) for (head, tail) in edgelist]
    return np.diag(degrees)


def create_tails_matrix(edgelist: List[Tuple[int, int]], graph: nx.DiGraph) -> np.array:
    degrees = [graph.degree(tail) for (head, tail) in edgelist]
    return np.diag(degrees)


def create_normalized_edge_adj(eadj, heads, tails) -> np.ndarray:
    """
    return T^-1/2 Q H^-1/2
    """
    t_inv_sqrt = np.power(tails, -0.5)
    t_inv_sqrt[np.isinf(t_inv_sqrt)] = 0.
    h_inv_sqrt = np.power(heads, -0.5)
    h_inv_sqrt[np.isinf(h_inv_sqrt)] = 0.
    n = np.matmul(eadj, h_inv_sqrt)
    n = np.matmul(t_inv_sqrt, n)
    return n


def add_self_loops(graph: nx.DiGraph) -> nx.DiGraph:
    """
    add self loops to graph
    """
    graph.add_edges_from([(u, u) for u in graph.nodes])
    return graph


def create_msg_matrix(edgelist: List[Tuple[int, int]], features: np.ndarray) -> np.ndarray:
    """
    create initial message matrix of shape (num_edges, num_feats)
    where the message on the edge from node i to node j is
        m_(i, j) = x_i
    and the self-message at node i (also defined as the hidden state of node i) is
        m_(i, i) = x_i
    """
    elen = len(edgelist)
    m = np.ndarray((elen, features.shape[1]))
    for i in range(elen):
        head_node = edgelist[i][0]
        m[i] = features[head_node]
    return m
