import itertools
from collections import defaultdict

import EoN
import networkx
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multinomial

from mmf.approximations import PA, HMF, MMF_from_distribution, TAME_from_distribution
from mmf.graph_generator import generate_graph
from mmf.utils import encode_x_nary


def plot_MMF(motifs, P_D, rate_fcn, mu_0, tmax, log=None, from_log=True):
    if not from_log or 'GAME' not in log:
        t, results = MMF_from_distribution(motifs, P_D, rate_fcn, mu_0, tmax=tmax)
        if log is not None:
            log['GAME'] = (t, results)
    else:
        t, results = log['GAME'][0], log['GAME'][1]
    plt.plot(t, results[1], '-.', label='MMF', linewidth=2)


def plot_MMF_SIR(motifs, P_D, rate_fcn, mu_0, tmax, log=None, from_log=True):
    if not from_log or 'GAME' not in log:
        t, results = MMF_from_distribution(motifs, P_D, rate_fcn, mu_0, tmax=tmax)
        if log is not None:
            log['GAME'] = (t, results)
    else:
        t, results = log['GAME'][0], log['GAME'][1]
    plt.plot(t, results[0], '-.', label=r'MMF ($\rho_S$)', linewidth=2, color='blue')
    plt.plot(t, results[1], '-.', label=r'MMF ($\rho_I$)', linewidth=2, color='red')
    plt.plot(t, results[2], '-.', label=r'MMF ($\rho_R$)', linewidth=2, color='green')


def plot_AME(Pst, F, R, rho, tmax, log=None, from_log=True):
    if not from_log or 'AME' not in log:
        t, S, I = TAME_from_distribution(Pst, F, R, rho=rho, tmax=tmax, reduce_to='AME')
        if log is not None:
            log['AME'] = (t, S, I)
    else:
        t, S, I = log['AME'][0], log['AME'][1], log['AME'][2]
    plt.plot(t, I, '--', label='AME', linewidth=2)


def plot_HPA(Pk, F, R, rho, tmax, log=None, from_log=True):
    if not from_log or 'HPA' not in log:
        t, S, I = PA(rho, F, R, Pk, tmax=tmax)
        if log is not None:
            log['HPA'] = (t, S, I)
    else:
        t, S, I = log['HPA'][0], log['HPA'][1], log['HPA'][2]
    plt.plot(t, I, ':', label='HPA', linewidth=2)


def plot_HMF(Pk, F, R, rho, tmax, log=None, from_log=True):
    if not from_log or 'HMF' not in log:
        t, S, I = HMF(rho, F, R, Pk, tmax=tmax)
        if log is not None:
            log['HMF'] = (t, S, I)
    else:
        t, S, I = log['HMF'][0], log['HMF'][1], log['HMF'][2]
    plt.plot(t, I, '-', label='HMF', linewidth=2)


def generate_motifs_triangles():
    edge = networkx.Graph()
    edge.add_nodes_from([0, 1])
    edge.add_edge(0, 1)

    triangle = networkx.Graph()
    triangle.add_nodes_from([0, 1, 2])
    triangle.add_edge(0, 1)
    triangle.add_edge(0, 2)
    triangle.add_edge(1, 2)

    return [edge, triangle]


def generate_motifs_squares():
    edge = networkx.Graph()
    edge.add_nodes_from([0, 1])
    edge.add_edge(0, 1)

    square = networkx.Graph()
    square.add_nodes_from([0, 1, 2, 3])
    square.add_edge(0, 1)
    square.add_edge(1, 2)
    square.add_edge(2, 3)
    square.add_edge(3, 1)

    return [edge, square]


def generate_hyperstub_distribution_uniform_over_roles(d, d_max, motif):
    P_unif = np.zeros([d_max+1] * len(motif))
    for indices in itertools.product(*[list(range(d + 1))] * len(motif)):
        if d == 0:
            P_unif[indices] = 1
        elif sum(indices) == d:
            P_unif[indices] = multinomial.pmf(indices, n=d, p=[1 / len(indices)] * len(indices))
    return P_unif


def generate_hyperstub_degree_distribution_diagonal(d_max, motif_pair):
    length = len(motif_pair[0]) + len(motif_pair[1])
    P_D = np.zeros([d_max+1] * length)
    for i in range(d_max+1):
        P_base = 1 / (d_max+1)
        P_0 = generate_hyperstub_distribution_uniform_over_roles(i, d_max, motif_pair[0])
        P_1 = generate_hyperstub_distribution_uniform_over_roles(i, d_max, motif_pair[1])
        P_D += P_base * np.expand_dims(P_0, axis=tuple(range(len(motif_pair[0]), length))) * np.expand_dims(P_1, axis=tuple(range(0, len(motif_pair[0]))))
    return P_D


def generate_hyperstub_degree_distribution_antidiagonal(d_max, motif_pair):
    length = len(motif_pair[0]) + len(motif_pair[1])
    P_D = np.zeros([d_max+1] * length)
    for i in range(d_max+1):
        P_base = 1 / (d_max+1)
        P_0 = generate_hyperstub_distribution_uniform_over_roles(i, d_max, motif_pair[0])
        P_1 = generate_hyperstub_distribution_uniform_over_roles(d_max - i, d_max, motif_pair[1])
        P_D += P_base * np.expand_dims(P_0, axis=tuple(range(len(motif_pair[0]), length))) * np.expand_dims(P_1, axis=tuple(range(0, len(motif_pair[0]))))
    return P_D


def generate_hyperstub_degree_distribution_uniform_up_to(d_max, motif_pair):
    length = len(motif_pair[0]) + len(motif_pair[1])
    count_ds = sum([1 for u, v in itertools.product(range(d_max+1), range(d_max+1)) if u+v <= d_max])
    P_D = np.zeros([d_max+1] * length)
    for u, v in itertools.product(range(d_max+1), range(d_max+1)):
        if u + v <= d_max:
            P_base = 1 / count_ds
            P_0 = generate_hyperstub_distribution_uniform_over_roles(u, d_max, motif_pair[0])
            P_1 = generate_hyperstub_distribution_uniform_over_roles(v, d_max, motif_pair[1])
            P_D += P_base * np.expand_dims(P_0, axis=tuple(range(len(motif_pair[0]), length))) * np.expand_dims(P_1, axis=tuple(range(0, len(motif_pair[0]))))
    return P_D


def compute_degree_triangle_distribution(P_D):
    max_s = P_D.shape[0] + P_D.shape[1]
    max_t = P_D.shape[2] + P_D.shape[3] + P_D.shape[4]
    max_k = max_s + 2 * max_t
    Pst = np.zeros((max_s, max_t))
    Pk = np.zeros(max_k)

    shape_ranges = [range(P_D.shape[i]) for i in range(len(P_D.shape))]
    for s1, s2, t1, t2, t3 in itertools.product(*shape_ranges):
        Pst[s1 + s2][t1 + t2 + t3] += P_D[s1, s2, t1, t2, t3]
        Pk[s1 + s2 + 2 * (t1 + t2 + t3)] += P_D[s1, s2, t1, t2, t3]
    return Pst, Pk


def compute_degree_square_distribution(P_D):
    max_s = P_D.shape[0] + P_D.shape[1]
    max_t = P_D.shape[2] + P_D.shape[3] + P_D.shape[4] + P_D.shape[5]
    max_k = max_s + 2 * max_t
    Pst = np.zeros((max_s, max_t))
    Pk = np.zeros(max_k)

    shape_ranges = [range(P_D.shape[i]) for i in range(len(P_D.shape))]
    for s1, s2, t1, t2, t3, t4 in itertools.product(*shape_ranges):
        Pst[s1 + s2][t1 + t2 + t3 + t4] += P_D[s1, s2, t1, t2, t3, t4]
        Pk[s1 + s2 + 2 * (t1 + t2 + t3 + t4)] += P_D[s1, s2, t1, t2, t3, t4]
    return Pst, Pk


def plot_Gillespie(N, motifs, P_D, rate_function, rho, tmax, num_runs=5):
    label = 'Gillespie'

    def transition_choice(G, node, status, parameters):
        if status[node] == 'S':
            return 'I'
        elif status[node] == 'I':
            return 'S'

    def get_influence_set(G, node, status, parameters):
        return G.neighbors(node)

    for i in range(num_runs):
        print(f"Running Gillespie iteration {i}")
        graph = generate_graph(N, motifs, P_D)
        IC = defaultdict(lambda: 'S')
        for node in range(len(graph)):
            if np.random.rand() < rho:
                IC[node] = 'I'

        t, S, I = EoN.Gillespie_complex_contagion(graph, rate_function,
                                                  transition_choice, get_influence_set, IC,
                                                  return_statuses=('S', 'I'),
                                                  tmax=tmax)
        plt.plot(t, I / len(graph), '-', label=label, alpha=0.4, linewidth=1, color="black")
        label = '__nolabel__'


def plot_Gillespie_SIR(N, motifs, P_D, rate_function, rho, tmax, num_runs=5):
    label = 'Gillespie'

    def transition_choice(G, node, status, parameters):
        if status[node] == 'S':
            return 'I'
        elif status[node] == 'I':
            return 'R'

    def get_influence_set(G, node, status, parameters):
        return G.neighbors(node)

    for i in range(num_runs):
        print(f"Running Gillespie iteration {i}")
        graph = generate_graph(N, motifs, P_D)
        IC = defaultdict(lambda: 'S')
        for node in range(len(graph)):
            if np.random.rand() < rho:
                IC[node] = 'I'

        t, S, I, R = EoN.Gillespie_complex_contagion(graph, rate_function,
                                                  transition_choice, get_influence_set, IC,
                                                  return_statuses=('S', 'I', 'R'),
                                                  tmax=tmax)
        plt.plot(t, S / len(graph), ':', label=label, alpha=0.4, linewidth=1, color="blue")
        plt.plot(t, I / len(graph), ':', label=label, alpha=0.4, linewidth=1, color="red")
        plt.plot(t, R / len(graph), ':', label=label, alpha=0.4, linewidth=1, color="green")
        label = '__nolabel__'
