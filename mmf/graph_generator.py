import itertools

import networkx
import numpy as np


def check_degree_sequence(motifs, hyperstub_degree_sequence):
    hyperstub_degree_sums = np.sum(np.array(hyperstub_degree_sequence), axis=0)
    curr_index = 0
    for motif in motifs:
        N = len(motif)
        arr = hyperstub_degree_sums[curr_index:curr_index+N]
        curr_index += N
        if not np.all(arr == arr[0]):
            return False
    return True


def sample_hyperstub_degree_sequence(N, hyperstub_degree_distribution):
    shape_ranges = [range(hyperstub_degree_distribution.shape[i]) for i in range(len(hyperstub_degree_distribution.shape))]
    hyperstub_degree_tuples = np.array(list(itertools.product(*shape_ranges)))
    hyperstub_degree_distribution = hyperstub_degree_distribution.flatten()

    while True:
        hyperstub_degree_sequence = hyperstub_degree_tuples[np.random.choice(hyperstub_degree_tuples.shape[0],
                                                                             N,
                                                                             replace=True,
                                                                             p=hyperstub_degree_distribution)]
        return hyperstub_degree_sequence


def connect_hyperstubs(G, hyperstubs_chosen, motif):
    for (u, v) in motif.edges:
        G.add_edge(hyperstubs_chosen[u], hyperstubs_chosen[v])


def generate_graph(N, motifs, hyperstub_degree_distribution):
    hyperstub_degree_sequence = sample_hyperstub_degree_sequence(N, hyperstub_degree_distribution)

    G = networkx.Graph()
    G.add_nodes_from(range(N))

    curr_index = 0
    for motif in motifs:
        hyperstub_lists = [[hyperstub_node
                           for node, num_hyperstubs in zip(range(N), hyperstub_degree_sequence[:, i])
                           for hyperstub_node in [node] * num_hyperstubs]
                          for i in range(curr_index, curr_index + len(motif))]
        curr_index += len(motif)

        while np.all(np.array([len(hyperstub_lists[i]) for i in range(len(hyperstub_lists))]) > 0):
            hyperstubs_chosen = []
            for i in range(len(hyperstub_lists)):
                hyperstub_list = hyperstub_lists[i]
                random_index = np.random.randint(0, len(hyperstub_list))
                hyperstubs_chosen.append(hyperstub_list[random_index])
                del hyperstub_list[random_index]
            connect_hyperstubs(G, hyperstubs_chosen, motif)

    G.remove_edges_from(networkx.selfloop_edges(G))
    return G
