import pytest
from edge_utils import *
import networkx as nx
import math


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
def edgelist_augmented():
    """
    bidirectional interpretation of an undirected graph with self loops
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
def features():
    """
    initial feature vector
    """
    return np.array([
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4],
    ])


@pytest.fixture
def heads_undirected():
    return np.array([
        [4, 0, 0, 0, 0, 0],
        [0, 4, 0, 0, 0, 0],
        [0, 0, 4, 0, 0, 0],
        [0, 0, 0, 4, 0, 0],
        [0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 2],
    ])


@pytest.fixture
def tails_undirected():
    return np.array([
        [4, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0],
        [0, 0, 4, 0, 0, 0],
        [0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 4, 0],
        [0, 0, 0, 0, 0, 4],
    ])


@pytest.fixture
def graph(edgelist):
    return nx.from_edgelist(edgelist)


@pytest.fixture
def graph_undirected(edgelist_undirected):
    return nx.from_edgelist(edgelist_undirected, create_using=nx.DiGraph)


@pytest.fixture
def edge_adj_undirected():
    return np.array([
        [0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ])


def test_to_edge_adj(edgelist):
    q = to_edge_adj(edgelist)
    assert np.all(q == [
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 0]
    ])


def test_to_edge_adj_undirected(edgelist_undirected, edge_adj_undirected):
    q = to_edge_adj(edgelist_undirected)
    assert np.all(q == edge_adj_undirected)


def test_create_heads_matrix_undirected(edgelist_undirected, graph_undirected, heads_undirected):
    heads = create_heads_matrix(edgelist_undirected, graph_undirected)
    assert np.all(heads == heads_undirected)


def test_create_tails_matrix_undirected(edgelist_undirected, graph_undirected, tails_undirected):
    tails = create_tails_matrix(edgelist_undirected, graph_undirected)
    assert np.all(tails == tails_undirected)


def test_create_heads_matrix(edgelist, graph):
    heads = create_heads_matrix(edgelist, graph)
    assert np.all(heads == np.array([
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2],
    ]))


def test_create_tails_matrix(edgelist, graph):
    tails = create_tails_matrix(edgelist, graph)
    assert np.all(tails == np.array([
        [2, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]))


def test_create_normalized_edge_adj(heads_undirected, tails_undirected, edge_adj_undirected):
    n = create_normalized_edge_adj(
        eadj=edge_adj_undirected,
        heads=heads_undirected,
        tails=tails_undirected,
    )
    val = math.pow(2, 1/2) / 4
    assert np.all(np.isclose(n, np.array([
        [0, 0, 0, 0, 0, val],
        [0, 0, val, 0, 0, 0],
        [0, 0, 0, 0, val, 0],
        [val, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],

    ])))


def test_create_msg_matrix(edgelist, features):
    assert np.all(create_msg_matrix(edgelist, features) == np.array([
        [1, 1],
        [1, 1],
        [2, 2]
    ]))


def test_add_self_loops(graph):
    graph = add_self_loops(graph)
    assert sorted(nx.to_pandas_edgelist(graph).values.tolist()) == [
        [1, 1],
        [1, 2],
        [1, 4],
        [2, 2],
        [2, 3],
        [3, 3],
        [4, 4],
    ]