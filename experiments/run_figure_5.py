import itertools
import pickle
import string

import matplotlib.pyplot as plt
import numpy as np

from mmf.utils import encode_x_nary
from experiments.common import plot_TAME, plot_AME, plot_HPA, plot_HMF, plot_Gillespie, \
    generate_hyperstub_degree_distribution_uniform_up_to, generate_motifs_triangles, \
    generate_hyperstub_degree_distribution_antidiagonal, generate_hyperstub_degree_distribution_diagonal, \
    compute_degree_triangle_distribution, plot_MMF


def rate_function_SIS_Gillespie_simplicial(G, node, status, parameters):
    triangles = [set(triple) for triple in set(
        frozenset((node, nbr, nbr2)) for nbr, nbr2 in itertools.combinations(G[node], 2) if nbr in G[nbr2])]
    num_infected_neighbors = len([nbr for nbr in G.neighbors(node) if status[nbr] == 'I'])
    sum_q = sum(
        [(n1 == node) * ((status[n2] == 'S') * (status[n3] == 'I') + (status[n2] == 'I') * (status[n3] == 'S')) +
         (n2 == node) * ((status[n1] == 'S') * (status[n3] == 'I') + (status[n1] == 'I') * (status[n3] == 'S')) +
         (n3 == node) * ((status[n1] == 'S') * (status[n2] == 'I') + (status[n1] == 'I') * (status[n2] == 'S'))
         for (n1, n2, n3) in triangles])
    sum_r = sum([(n1 == node) * ((status[n2] == 'I') * (status[n3] == 'I')) +
                 (n2 == node) * ((status[n1] == 'I') * (status[n3] == 'I')) +
                 (n3 == node) * ((status[n1] == 'I') * (status[n2] == 'I'))
                 for (n1, n2, n3) in triangles])
    num_infected_triangle_neighbors = sum_q + 2 * sum_r

    if status[node] == 'S':
        return tau * (num_infected_neighbors + num_infected_triangle_neighbors)
    elif status[node] == 'I':
        return gamma


def rate_function_SIS_simplicial(state_idx, curr_hyperstub_qs, idx2motifs, idx2hyperstub_idxs):
    rates = np.zeros(2)
    num_states = 2

    if state_idx == 1:
        rates[0] = gamma
        return rates
    else:
        curr_idx = 0
        for degree_idx, motif, hyperstub_idx in zip(range(len(idx2motifs)), idx2motifs, idx2hyperstub_idxs):
            num_configs_of_motif = num_states ** len(motif)
            for conf_idx in range(num_configs_of_motif):
                hyperstub_config_count = curr_hyperstub_qs[curr_idx + conf_idx]
                motif_config = encode_x_nary(len(motif), conf_idx, num_states)
                num_infected_neighbors = sum([motif.has_edge(hyperstub_idx, v) * (motif_config[v] == 1) for v in motif.neighbors(hyperstub_idx)])
                num_infected_triangles = sum([motif.has_edge(hyperstub_idx, v) * motif.has_edge(hyperstub_idx, w) * motif.has_edge(v, w) *
                                              ((motif_config[v] == 1) + (motif_config[w] == 1)) for v, w in itertools.product(motif.neighbors(hyperstub_idx), motif.neighbors(hyperstub_idx))]) // 2
                rates[1] += hyperstub_config_count * (num_infected_neighbors + num_infected_triangles) * tau
            curr_idx += num_configs_of_motif
        return rates


if __name__ == '__main__':
    plt.clf()
    plt.rcParams.update({
        "figure.dpi": 600,
        "text.usetex": False,
        "font.size": 20,
    })

    motifs = generate_motifs_triangles()
    # Ns = [10000, 10000]
    Ns = [100000, 100000, 100000]
    taus = [0.3] * 3  # transmission rate
    gammas = [0.9] * 3  # recovery rate
    rhos = [0.2, 0.2, 0.4]  # random fraction initially infected
    tmax = 10
    P_Ds = [generate_hyperstub_degree_distribution_antidiagonal(3, motifs),
            generate_hyperstub_degree_distribution_uniform_up_to(3, motifs),
            generate_hyperstub_degree_distribution_diagonal(2, motifs)]

    for i, N, tau, gamma, rho, P_D in zip(range(len(Ns)), Ns, taus, gammas, rhos, P_Ds):
        plt.subplot(1, len(Ns), 1+i)
        plt.gca().text(-0.01, 1.03, '(' + string.ascii_lowercase[i - 1] + ')', transform=plt.gca().transAxes,
                       size=18, weight='bold')

        log = {}
        try:
            with open(f'./results/5_logs_{i}.pkl', 'rb') as f:
                log = pickle.load(f)
                print("loaded " + f'./results/5_logs_{i}.pkl')
        except FileNotFoundError:
            pass

        F = lambda s, t, p, q, r: tau * (p + 2 * q + 4 * r)
        R = lambda s, t, p, q, r: gamma
        Pst, Pk = compute_degree_triangle_distribution(P_D)

        # plot_HMF(Pk, F, R, rho, tmax, log=log)
        # plot_HPA(Pk, F, R, rho, tmax, log=log)
        # plot_AME(Pst, F, R, rho, tmax, log=log)
        # plot_TAME(Pst, F, R, rho, tmax, log=log)
        plot_MMF(motifs, P_D, rate_function_SIS_simplicial, [1 - rho, rho], tmax, log=log)
        plot_Gillespie(N, motifs, P_D, rate_function_SIS_Gillespie_simplicial, rho, tmax)

        with open(f'./results/5_logs_{i}.pkl', 'wb') as f:
            pickle.dump(log, f, 4)

        plt.xlabel('$t$')
        plt.ylabel(r'$\rho(I)$')
        plt.legend()
        # plt.title(fr"$\tau$={tau}, $\gamma$={gamma} $\rho$={rho}")

    plt.gcf().set_size_inches(18, 6)
    plt.tight_layout()
    plt.savefig('./figures/Fig5_SIS_simplicial.pdf', bbox_inches='tight', transparent=True, pad_inches=0)
    plt.show()
