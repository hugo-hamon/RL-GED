import itertools
import numpy as np

def compute_ged(g1, g2, f1, f2, assignementMatrix):
    """
        Compute the Graph Edit Distance between graph1 and graph2 with the transport plan assignementMatrix
        g1 - 1st graph with edges (nx.Graph)
        g2 - 2nd graph with edges (nx.Graph)
        n1 - list of nodes features of the 1st graph (size n)
        n2 - list of nodes features of the 2nd graph (size m)
        assignementMatrix - transport plan (size (n + m) x (n + m))
    """
    graph1, graph2, features1, features2 = convert_to_balanced_graph(g1, g2, f1, f2)

    nodeCost = default_node_cost(features1, features2)

    #Calculs des matrices d'adjacence et de la matrice de coût des arêtes
    edgeMatrix1 = edge_matrix(graph1)
    edgeMatrix2 = edge_matrix(graph2)
    edgeCost = default_edge_cost(edgeMatrix1, edgeMatrix2)

    return cout(assignementMatrix, nodeCost, edgeCost)

def convert_to_balanced_graph(graph1, graph2, f1, f2):
    """
        adds dummy nodes with no features to both graphs to make them the same size
    """
    graph1 = graph1.copy()
    graph2 = graph2.copy()
    features1 = f1.copy()
    features2 = f2.copy()

    size_1 = graph1.number_of_nodes()
    size_2 = graph2.number_of_nodes()

    for i in range(size_1):
        graph2.add_node(size_2 + i)
        features2.append(-1)
    for i in range(size_2):
        graph1.add_node(size_1 + i)
        features1.append(-1)

    return graph1, graph2, features1, features2

def default_node_cost(n1, n2):
    """
        Returns the cost matrix of the nodes defined between each pair of nodes
        n1 - list of nodes features of the 1st graph
        n2 - list of nodes features of the 2nd graph
    """
    return np.where(np.equal(np.array(n1)[:, None], np.array(n2)[None, :]), 0, 1)

def default_edge_cost(a1, a2):
    """
        Return the cost matrix of the edges defined
        as a1 XOR a2 for each pair of edges with
        a1 - adjacency matrix of the 1st graph (Size n x n)
        a2 - adjacency matrix of the 2nd graph (Size n x n)
    """
    return (a1[:, None, :, None] + a2[None, :, None, :] -
            2 * np.einsum('ik,jl->ijkl', a1, a2))

def cout(pt, nodeCost, edgeCost):
    """
        Compute the cost of the transport plan pt
        pt - transport plan (size n x n)
        nodeCost - cost matrix of the nodes (size n x n)
        edgeCost - cost matrix of the edges (size n x n x n x n)
    """
    res = 0
    # Calcule du cout des noeuds
    # Equivalent à:
    # for i in range(size_1):
    #   for j in range(size_2):
    #     res += pt[i,j] * nodeCost[i,j]
    res += np.sum(pt * nodeCost)

    # Calcul du cout des arêtes
    # Equivalent à:
    # for i in range(size_1):
    #   for j in range(size_1):
    #     for k in range(size_2):
    #       for l in range(size_2):
    #         res += pt[i,j] * edgeCost[i,j,k,l] * pt[k,l]
    res += np.einsum('ij,ijkl,kl->', pt, edgeCost, pt) / 2
    return res

def edge_matrix(graph):
    """
    Returns the adjacency matrix of the graph
    """
    size = graph.number_of_nodes()
    edgeMatrix = np.zeros((size, size))
    for i, j in itertools.product(range(size), range(size)):
        edgeMatrix[i][j] = 1 if (i, j) in graph.edges else 0
    return edgeMatrix
    # return nx.adjacency_matrix(graph).toarray() 
