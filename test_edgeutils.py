import pytest
from edge_utils import *
import networkx as nx

@pytest.fixture
def edgelist():
    return [
        (1, 2),
        (1, 4),
        (2, 3),
    ]


@pytest.fixture
def edgelist_undirected():
    """
    bidirectional interpretation of an undirected graph
    """
    return [
        (1, 2),
        (1, 4),
        (2, 1),
        (2, 3),
        (3, 2),
        (4, 1),
    ]


@pytest.fixture
def graph(edgelist):
    return nx.from_edgelist(edgelist)


@pytest.fixture
def graph_undirected(edgelist_undirected):
    return nx.from_edgelist(edgelist_undirected, create_using=nx.DiGraph)


def test_to_edge_adj(edgelist):
    q = to_edge_adj(edgelist)
    assert np.all(q == [
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 0]
    ])


def test_to_edge_adj_undirected(edgelist_undirected):
    q = to_edge_adj(edgelist_undirected)
    assert np.all(q == [
        [0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ])


def test_create_heads_matrix_undirected(graph_undirected):
    heads = create_heads_matrix(graph_undirected)
    assert np.all(heads == np.array([
        [4, 0, 0, 0, 0, 0],
        [0, 4, 0, 0, 0, 0],
        [0, 0, 4, 0, 0, 0],
        [0, 0, 0, 4, 0, 0],
        [0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 2],
    ]))


def test_create_tails_matrix_undirected(graph_undirected):
    tails = create_tails_matrix(graph_undirected)
    assert np.all(tails == np.array([
        [4, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0],
        [0, 0, 4, 0, 0, 0],
        [0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 4, 0],
        [0, 0, 0, 0, 0, 4],
    ]))


def test_create_heads_matrix(graph):
    heads = create_heads_matrix(graph)
    assert np.all(heads == np.array([
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2],
    ]))


def test_create_tails_matrix(graph):
    tails = create_tails_matrix(graph)
    assert np.all(tails == np.array([
        [2, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]))
