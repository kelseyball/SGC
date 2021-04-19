from typing import List, Tuple


def to_edge_adj(edgelist: List[Tuple[int]]) -> Tuple[List[List[int]], int]:
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
    return q
