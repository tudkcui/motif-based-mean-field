import pickle
import string

import matplotlib.pyplot as plt
import numpy as np

from mmf.utils import encode_x_nary
from experiments.common import plot_AME, plot_HPA, plot_HMF, plot_Gillespie, \
    generate_hyperstub_degree_distribution_uniform_up_to, generate_motifs_triangles, \
    generate_hyperstub_degree_distribution_antidiagonal, generate_hyperstub_degree_distribution_diagonal, \
    compute_degree_triangle_distribution, plot_MMF


def rate_function_Ising_Gillespie(G, node, status, parameters):
    num_neighbors = len([nbr for nbr in G.neighbors(node)])
    num_infected_neighbors = len([nbr for nbr in G.neighbors(node) if status[nbr] == 'I'])
    if status[node] == 'S':
        return 1 / (1 + np.exp(2 / T * (num_neighbors - 2 * num_infected_neighbors)))
    elif status[node] == 'I':
        return 1 - 1 / (1 + np.exp(2 / T * (num_neighbors - 2 * num_infected_neighbors)))


def rate_function_Ising(state_idx, curr_hyperstub_qs, idx2motifs, idx2hyperstub_idxs):
    rates = np.zeros(2)
    num_states = 2
    num_neighbors = 0
    num_infected_neighbors = 0

    curr_idx = 0
    for degree_idx, motif, hyperstub_idx in zip(range(len(idx2motifs)), idx2motifs, idx2hyperstub_idxs):
        num_configs_of_motif = num_states ** len(motif)
        for conf_idx in range(num_configs_of_motif):
            hyperstub_config_count = curr_hyperstub_qs[curr_idx + conf_idx]
            motif_config = encode_x_nary(len(motif), conf_idx, num_states)
            num_neighbors += hyperstub_config_count * sum([1 for v in motif.neighbors(hyperstub_idx)])
            num_infected_neighbors += hyperstub_config_count * sum([motif.has_edge(hyperstub_idx, v) * (motif_config[v] == 1) for v in motif.neighbors(hyperstub_idx)])
        curr_idx += num_configs_of_motif
    if state_idx == 0:
        rates[1] = 1 / (1 + np.exp(2 / T * (num_neighbors - 2 * num_infected_neighbors)))
    elif state_idx == 1:
        rates[0] = 1 - 1 / (1 + np.exp(2 / T * (num_neighbors - 2 * num_infected_neighbors)))
    return rates

if __name__ == '__main__':
    plt.clf()
    plt.rcParams.update({
        "figure.dpi": 600,
        "text.usetex": False,
        "font.size": 20,
    })

    motifs = generate_motifs_triangles()
    Ns = [100000, 100000, 100000]
    Ts = [1, 3, 4]
    rhos = [0.33] * len(Ns)
    tmax = 10
    P_Ds = [generate_hyperstub_degree_distribution_antidiagonal(3, motifs)] * 4

    for i, N, T, rho, P_D in zip(range(len(Ns)), Ns, Ts, rhos, P_Ds):
        plt.subplot(1, len(Ns), 1+i)
        plt.gca().text(-0.01, 1.03, '(' + string.ascii_lowercase[3+i] + ')', transform=plt.gca().transAxes,
                       size=18, weight='bold')

        log = {}
        try:
            with open(f'./results/4b_logs_{i}.pkl', 'rb') as f:
                log = pickle.load(f)
                print("loaded " + f'./results/4b_logs_{i}.pkl')
        except FileNotFoundError:
            pass

        F = lambda s, t, p, q, r: 1 / (1 + np.exp(2 / T * (s + 2*t - 2 * (p + q + 2*r))))
        R = lambda s, t, p, q, r: 1 - 1 / (1 + np.exp(2 / T * (s + 2*t - 2 * (p + q + 2*r))))
        Pst, Pk = compute_degree_triangle_distribution(P_D)

        plot_HMF(Pk, F, R, rho, tmax, log=log)
        plot_HPA(Pk, F, R, rho, tmax, log=log)
        plot_AME(Pst, F, R, rho, tmax, log=log)
        plot_MMF(motifs, P_D, rate_function_Ising, [1 - rho, rho], tmax, log=log)
        plot_Gillespie(N, motifs, P_D, rate_function_Ising_Gillespie, rho, tmax)

        with open(f'./results/4b_logs_{i}.pkl', 'wb') as f:
            pickle.dump(log, f, 4)

        plt.xlabel('$t$')
        plt.ylabel(r'$\rho(D)$')
        plt.legend()
        # plt.title(fr"$T/J$={T}, $\rho$={rho}")

    plt.gcf().set_size_inches(18, 6)
    plt.tight_layout()
    plt.savefig('./figures/Fig4b_IsingGlauber_Ts.pdf', bbox_inches='tight', transparent=True, pad_inches=0)
    plt.show()
