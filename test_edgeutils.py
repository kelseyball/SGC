import pytest
from edge_utils import to_edge_adj

@pytest.fixture
def edgelist():
    return [
        (1, 2),
        (2, 3),
        (1, 4),
    ]


@pytest.fixture
def edgelist_undirected():
    return [
        (1, 2),
        (2, 1),
        (2, 3),
        (3, 2),
        (1, 4),
        (4, 1),
    ]


def test_to_edge_adj(edgelist):
    q = to_edge_adj(edgelist)
    assert q == [
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 0]
    ]


def test_to_edge_adj_undirected(edgelist_undirected):
    q = to_edge_adj(edgelist_undirected)
    assert q == [
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]