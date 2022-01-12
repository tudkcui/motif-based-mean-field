import itertools

import numpy as np
import scipy
from scipy import integrate
from scipy.special import binom, comb

from mmf.utils import multinomial, encode_x_nary, decode_x_nary


def _dTAME_(X, t, original_shape, F, R, Pst):
    print(t)
    ksq = np.prod(original_shape)
    S_stpqr = X[:ksq]
    I_stpqr = X[ksq:]
    S_stpqr.shape = original_shape
    I_stpqr.shape = original_shape

    beta1s = sum([F(s, t, p, q, r) * (s - p) * S_stpqr[s][t][p][q][r] * Pst[(s, t)]
                  for r in range(original_shape[4]) for q in range(original_shape[3])
                  for p in range(original_shape[2]) for t in range(original_shape[1])
                  for s in range(original_shape[0])]) / \
             (sum([(s - p) * S_stpqr[s][t][p][q][r] * Pst[(s, t)]
                  for r in range(original_shape[4]) for q in range(original_shape[3])
                  for p in range(original_shape[2]) for t in range(original_shape[1])
                  for s in range(original_shape[0])]) + 1e-10)

    beta2s = sum([2 * F(s, t, p, q, r) * (t - q - r) * S_stpqr[s][t][p][q][r] * Pst[(s, t)]
                  for r in range(original_shape[4]) for q in range(original_shape[3])
                  for p in range(original_shape[2]) for t in range(original_shape[1])
                  for s in range(original_shape[0])]) / \
             (sum([(t - q - r) * S_stpqr[s][t][p][q][r] * Pst[(s, t)]
                  for r in range(original_shape[4]) for q in range(original_shape[3])
                  for p in range(original_shape[2]) for t in range(original_shape[1])
                  for s in range(original_shape[0])]) + 1e-10)

    beta3s = sum([F(s, t, p, q, r) * (q) * S_stpqr[s][t][p][q][r] * Pst[(s, t)]
                  for r in range(original_shape[4]) for q in range(original_shape[3])
                  for p in range(original_shape[2]) for t in range(original_shape[1])
                  for s in range(original_shape[0])]) / \
             (sum([(q) * S_stpqr[s][t][p][q][r] * Pst[(s, t)]
                  for r in range(original_shape[4]) for q in range(original_shape[3])
                  for p in range(original_shape[2]) for t in range(original_shape[1])
                  for s in range(original_shape[0])]) + 1e-10)

    gamma1s = sum([R(s, t, p, q, r) * (s - p) * I_stpqr[s][t][p][q][r] * Pst[(s, t)]
                  for r in range(original_shape[4]) for q in range(original_shape[3])
                  for p in range(original_shape[2]) for t in range(original_shape[1])
                  for s in range(original_shape[0])]) / \
            (sum([(s - p) * I_stpqr[s][t][p][q][r] * Pst[(s, t)]
                  for r in range(original_shape[4]) for q in range(original_shape[3])
                  for p in range(original_shape[2]) for t in range(original_shape[1])
                  for s in range(original_shape[0])]) + 1e-10)

    gamma2s = sum([R(s, t, p, q, r) * (t - q - r) * I_stpqr[s][t][p][q][r] * Pst[(s, t)]
                  for r in range(original_shape[4]) for q in range(original_shape[3])
                  for p in range(original_shape[2]) for t in range(original_shape[1])
                  for s in range(original_shape[0])]) / \
            (sum([(t - q - r) * I_stpqr[s][t][p][q][r] * Pst[(s, t)]
                  for r in range(original_shape[4]) for q in range(original_shape[3])
                  for p in range(original_shape[2]) for t in range(original_shape[1])
                  for s in range(original_shape[0])]) + 1e-10)

    gamma3s = sum([2 * R(s, t, p, q, r) * (q) * I_stpqr[s][t][p][q][r] * Pst[(s, t)]
                  for r in range(original_shape[4]) for q in range(original_shape[3])
                  for p in range(original_shape[2]) for t in range(original_shape[1])
                  for s in range(original_shape[0])]) / \
            (sum([(q) * I_stpqr[s][t][p][q][r] * Pst[(s, t)]
                  for r in range(original_shape[4]) for q in range(original_shape[3])
                  for p in range(original_shape[2]) for t in range(original_shape[1])
                  for s in range(original_shape[0])]) + 1e-10)

    beta1i = sum([F(s, t, p, q, r) * (p) * S_stpqr[s][t][p][q][r] * Pst[(s, t)]
                  for r in range(original_shape[4]) for q in range(original_shape[3])
                  for p in range(original_shape[2]) for t in range(original_shape[1])
                  for s in range(original_shape[0])]) / \
            (sum([(p) * S_stpqr[s][t][p][q][r] * Pst[(s, t)]
                  for r in range(original_shape[4]) for q in range(original_shape[3])
                  for p in range(original_shape[2]) for t in range(original_shape[1])
                  for s in range(original_shape[0])]) + 1e-10)

    beta2i = sum([2 * F(s, t, p, q, r) * (q) * S_stpqr[s][t][p][q][r] * Pst[(s, t)]
                  for r in range(original_shape[4]) for q in range(original_shape[3])
                  for p in range(original_shape[2]) for t in range(original_shape[1])
                  for s in range(original_shape[0])]) / \
            (sum([(q) * S_stpqr[s][t][p][q][r] * Pst[(s, t)]
                  for r in range(original_shape[4]) for q in range(original_shape[3])
                  for p in range(original_shape[2]) for t in range(original_shape[1])
                  for s in range(original_shape[0])]) + 1e-10)

    beta3i = sum([F(s, t, p, q, r) * (r) * S_stpqr[s][t][p][q][r] * Pst[(s, t)]
                  for r in range(original_shape[4]) for q in range(original_shape[3])
                  for p in range(original_shape[2]) for t in range(original_shape[1])
                  for s in range(original_shape[0])]) / \
            (sum([(r) * S_stpqr[s][t][p][q][r] * Pst[(s, t)]
                  for r in range(original_shape[4]) for q in range(original_shape[3])
                  for p in range(original_shape[2]) for t in range(original_shape[1])
                  for s in range(original_shape[0])]) + 1e-10)

    gamma1i = sum([R(s, t, p, q, r) * (p) * I_stpqr[s][t][p][q][r] * Pst[(s, t)]
                  for r in range(original_shape[4]) for q in range(original_shape[3])
                  for p in range(original_shape[2]) for t in range(original_shape[1])
                  for s in range(original_shape[0])]) / \
            (sum([(p) * I_stpqr[s][t][p][q][r] * Pst[(s, t)]
                  for r in range(original_shape[4]) for q in range(original_shape[3])
                  for p in range(original_shape[2]) for t in range(original_shape[1])
                  for s in range(original_shape[0])]) + 1e-10)

    gamma2i = sum([R(s, t, p, q, r) * (q) * I_stpqr[s][t][p][q][r] * Pst[(s, t)]
                  for r in range(original_shape[4]) for q in range(original_shape[3])
                  for p in range(original_shape[2]) for t in range(original_shape[1])
                  for s in range(original_shape[0])]) / \
            (sum([(q) * I_stpqr[s][t][p][q][r] * Pst[(s, t)]
                  for r in range(original_shape[4]) for q in range(original_shape[3])
                  for p in range(original_shape[2]) for t in range(original_shape[1])
                  for s in range(original_shape[0])]) + 1e-10)

    gamma3i = sum([2 * R(s, t, p, q, r) * (r) * I_stpqr[s][t][p][q][r] * Pst[(s, t)]
                  for r in range(original_shape[4]) for q in range(original_shape[3])
                  for p in range(original_shape[2]) for t in range(original_shape[1])
                  for s in range(original_shape[0])]) / \
            (sum([(r) * I_stpqr[s][t][p][q][r] * Pst[(s, t)]
                  for r in range(original_shape[4]) for q in range(original_shape[3])
                  for p in range(original_shape[2]) for t in range(original_shape[1])
                  for s in range(original_shape[0])]) + 1e-10)

    dS_stpqr_center = np.zeros(original_shape)
    dS_stpqr_neighbor = np.zeros(original_shape)
    dI_stpqr_center = np.zeros(original_shape)
    dI_stpqr_neighbor = np.zeros(original_shape)

    """ Center jumps """
    for s in range(original_shape[0]):
        for t in range(original_shape[1]):
            for p in range(s + 1):
                for q in range(t + 1):
                    for r in range(t - q + 1):
                        dS_stpqr_center[s][t][p][q][r] += R(s, t, p, q, r) * I_stpqr[s][t][p][q][r] \
                                                  - F(s, t, p, q, r) * S_stpqr[s][t][p][q][r]
                        dI_stpqr_center[s][t][p][q][r] += - R(s, t, p, q, r) * I_stpqr[s][t][p][q][r] \
                                                  + F(s, t, p, q, r) * S_stpqr[s][t][p][q][r]

    """ Neighbor jumps """
    for s in range(original_shape[0]):
        for t in range(original_shape[1]):
            for p in range(s + 1):
                for q in range(t + 1):
                    for r in range(t - q + 1):
                        Spi = 0 if p==0 else S_stpqr[s][t][p-1][q][r]
                        Ipi = 0 if p==0 else I_stpqr[s][t][p-1][q][r]

                        Sqi = 0 if q==0 else S_stpqr[s][t][p][q-1][r]
                        Iqi = 0 if q==0 else I_stpqr[s][t][p][q-1][r]

                        Sri = 0 if r==0 or q==t else S_stpqr[s][t][p][q+1][r-1]
                        Iri = 0 if r==0 or q==t else I_stpqr[s][t][p][q+1][r-1]

                        Spi2 = 0 if p==s else S_stpqr[s][t][p+1][q][r]
                        Ipi2 = 0 if p==s else I_stpqr[s][t][p+1][q][r]

                        Sqi2 = 0 if q==t else S_stpqr[s][t][p][q+1][r]
                        Iqi2 = 0 if q==t else I_stpqr[s][t][p][q+1][r]

                        Sri2 = 0 if r==t or q==0 else S_stpqr[s][t][p][q-1][r+1]
                        Iri2 = 0 if r==t or q==0 else I_stpqr[s][t][p][q-1][r+1]

                        dS_stpqr_neighbor[s][t][p][q][r] += beta1s * (s - p + 1) * Spi - gamma1s * p * S_stpqr[s][t][p][q][r] \
                                + beta2s * (t - q - r + 1) * Sqi - gamma2s * q * S_stpqr[s][t][p][q][r] \
                                + beta3s * (q + 1) * Sri - gamma3s * r * S_stpqr[s][t][p][q][r] \
                                + gamma1s * (p + 1) * Spi2 - beta1s * (s - p) * S_stpqr[s][t][p][q][r] \
                                + gamma2s * (q + 1) * Sqi2 - beta2s * (t - q - r) * S_stpqr[s][t][p][q][r] \
                                + gamma3s * (r + 1) * Sri2 - beta3s * q * S_stpqr[s][t][p][q][r]

                        dI_stpqr_neighbor[s][t][p][q][r] += beta1i * (s - p + 1) * Ipi - gamma1i * p * I_stpqr[s][t][p][q][r] \
                                + beta2i * (t - q - r + 1) * Iqi - gamma2i * q * I_stpqr[s][t][p][q][r] \
                                + beta3i * (q + 1) * Iri - gamma3i * r * I_stpqr[s][t][p][q][r] \
                                + gamma1i * (p + 1) * Ipi2 - beta1i * (s - p) * I_stpqr[s][t][p][q][r] \
                                + gamma2i * (q + 1) * Iqi2 - beta2i * (t - q - r) * I_stpqr[s][t][p][q][r] \
                                + gamma3i * (r + 1) * Iri2 - beta3i * q * I_stpqr[s][t][p][q][r]

    dS_stpqr = dS_stpqr_center + dS_stpqr_neighbor
    dI_stpqr = dI_stpqr_center + dI_stpqr_neighbor

    dS_stpqr.shape = ksq
    dI_stpqr.shape = ksq

    return np.concatenate((dS_stpqr, dI_stpqr), axis=0)


def TAME(S_stpqr0, I_stpqr0, F, R, Pst, tmin=0, tmax=100,
         tcount=1001, reduce_to=None, return_full_data=False):
    times = np.linspace(tmin, tmax, tcount)
    original_shape = S_stpqr0.shape
    ksq = np.prod(original_shape)
    S_stpqr0.shape = (1, ksq)
    I_stpqr0.shape = (1, ksq)

    X0 = np.concatenate((S_stpqr0[0], I_stpqr0[0]), axis=0)

    times, Ps = [], []
    times.append(tmin)
    Ps.append(X0)
    integrator = integrate.RK23(lambda t, x: _dTAME_(x, t, original_shape, F, R, Pst), tmin, X0, tmax, 0.1)
    while integrator.status != 'finished':
        integrator.step()
        times.append(integrator.t)
        Ps.append(integrator.y)
    X = np.array(Ps)

    S_stpqr = X.T[0:ksq]
    I_stpqr = X.T[ksq:]

    flat_fracs = (np.expand_dims(Pst, axis=(2, 3, 4)) * np.ones(original_shape)).reshape((-1, 1))

    S = (flat_fracs * S_stpqr).sum(axis=0)
    I = (flat_fracs * I_stpqr).sum(axis=0)
    if return_full_data:
        S_stpqr.shape = (original_shape[0], original_shape[1], original_shape[2], original_shape[3], original_shape[4], tcount)
        I_stpqr.shape = (original_shape[0], original_shape[1], original_shape[2], original_shape[3], original_shape[4], tcount)
        return times, S, I, S_stpqr, I_stpqr
    else:
        return times, S, I


def TAME_from_distribution(Pst, F, R, rho, tmin=0,
                           tmax=100, tcount=1001,
                           return_full_data=False,
                           reduce_to=None):
    if reduce_to == 'AME':
        maxs = Pst.shape[0] + 2 * Pst.shape[1]
        Pst_new = np.zeros((maxs + 1, 1))
        for s in range(Pst.shape[0]):
            for t in range(Pst.shape[1]):
                Pst_new[s + 2*t] += Pst[s][t]
        Pst = Pst_new

    """ Find initial configuration """
    S_stpqr0 = np.zeros((Pst.shape[0], Pst.shape[1], Pst.shape[0], Pst.shape[1], Pst.shape[1]))
    I_stpqr0 = np.zeros((Pst.shape[0], Pst.shape[1], Pst.shape[0], Pst.shape[1], Pst.shape[1]))
    for s in range(Pst.shape[0]):
        for t in range(Pst.shape[1]):
            for p in range(s + 1):
                for q in range(t + 1):
                    for r in range(t - q + 1):
                        Nst_frac = Pst[(s, t)]
                        if Nst_frac == 0:
                            continue
                        binomial_result = multinomial([s - p, p])
                        trinomial_result = multinomial([t - q - r, q, r])
                        if binomial_result < float('Inf') and trinomial_result < float('Inf'):
                            S_stpqr0[s, t, p, q, r] = (1 - rho) * binomial_result * trinomial_result\
                                          * rho ** p * (1 - rho) ** (s-p)\
                                            * (rho ** 2) ** r * (2 * rho * (1-rho)) ** q * ((1-rho) ** 2) ** (t - q - r)
                            I_stpqr0[s, t, p, q, r] = rho * binomial_result * trinomial_result\
                                          * rho ** p * (1 - rho) ** (s-p)\
                                            * (rho ** 2) ** r * (2 * rho * (1-rho)) ** q * ((1-rho) ** 2) ** (t - q - r)
                        else:
                            S_stpqr0[s, t, p, q, r] = 0
                            I_stpqr0[s, t, p, q, r] = 0

    return TAME(S_stpqr0, I_stpqr0, F, R, Pst, tmin=tmin,
                tmax=tmax, tcount=tcount, reduce_to=reduce_to,
                return_full_data=return_full_data)


def B(k, m, q):
    return scipy.special.binom(k, m) * (q ** m) * ((1 - q) ** (k - m))


def _dPA_(X, t, F, R, Pk):
    kmax = len(Pk) - 1
    rho = X[:kmax+1]
    p = X[kmax+1:2*kmax+2]
    q = X[2*kmax+2:]

    drho = np.zeros_like(rho)
    dp = np.zeros_like(p)
    dq = np.zeros_like(q)

    beta_s = sum([Pk[k] * (k-m) * F(k, 0, m, 0, 0) * (1-rho[k]) * B(k, m, p[k])
                  for k in range(kmax + 1) for m in range(k+1)]) / \
             sum([Pk[k] * (k-m) * (1-rho[k]) * B(k, m, p[k])
                  for k in range(kmax + 1) for m in range(k+1)])
    gamma_s = sum([Pk[k] * (k-m) * R(k, 0, m, 0, 0) * rho[k] * B(k, m, q[k])
                  for k in range(kmax + 1) for m in range(k+1)]) / \
              sum([Pk[k] * (k-m) * rho[k] * B(k, m, q[k])
                  for k in range(kmax + 1) for m in range(k+1)])
    beta_i = sum([Pk[k] * m * F(k, 0, m, 0, 0) * (1-rho[k]) * B(k, m, p[k])
                  for k in range(kmax + 1) for m in range(k+1)]) / \
             sum([Pk[k] * m * (1-rho[k]) * B(k, m, p[k])
                  for k in range(kmax + 1) for m in range(k+1)])
    gamma_i = sum([Pk[k] * m * R(k, 0, m, 0, 0) * rho[k] * B(k, m, q[k])
                  for k in range(kmax + 1) for m in range(k+1)]) / \
              sum([Pk[k] * m * rho[k] * B(k, m, q[k])
                  for k in range(kmax + 1) for m in range(k+1)])

    for k in range(kmax+1):
        drho[k] = sum([- rho[k] * R(k, 0, m, 0, 0) * B(k, m, q[k])
                       + (1-rho[k]) * F(k, 0, m, 0, 0) * B(k, m, p[k])
                       for m in range(k+1)])

    for k in range(1, kmax+1):
        dp[k] = sum([(p[k] - m/k) * (F(k, 0, m, 0, 0) * B(k, m, p[k])
                                     - rho[k] / (1-rho[k]) * R(k, 0, m, 0, 0) * B(k, m, q[k]))
                     for m in range(k+1)]) + beta_s * (1-p[k]) - gamma_s * p[k]
        dq[k] = sum([(q[k] - m/k) * (R(k, 0, m, 0, 0) * B(k, m, q[k])
                                     - (1-rho[k]) / rho[k] * F(k, 0, m, 0, 0) * B(k, m, p[k]))
                     for m in range(k+1)]) + beta_i * (1-q[k]) - gamma_i * q[k]

    return np.concatenate((drho, dp, dq), axis=0)


def _dHMF_(rho, t, F, R, Pk):
    kmax = len(Pk) - 1
    drho = np.zeros_like(rho)

    omega = sum([Pk[k] * k * rho[k] for k in range(kmax + 1)]) / \
            sum([Pk[k] * k for k in range(kmax + 1)])

    for k in range(kmax+1):
        drho[k] = sum([- rho[k] * R(k, 0, m, 0, 0) * B(k, m, omega)
                       + (1-rho[k]) * F(k, 0, m, 0, 0) * B(k, m, omega)
                       for m in range(k+1)])

    return drho


def PA(rho_0, F, R, Pk, tmin=0, tmax=100,
                         tcount=1001, return_full_data=False):
    times = np.linspace(tmin, tmax, tcount)

    X0 = np.concatenate((rho_0 * np.ones_like(Pk), rho_0 * np.ones_like(Pk), rho_0 * np.ones_like(Pk)), axis=0)

    times, Ps = [], []
    times.append(tmin)
    Ps.append(X0)
    integrator = integrate.RK23(lambda t, x: _dPA_(x, t, F, R, Pk), tmin, X0, tmax, 0.1)
    while integrator.status != 'finished':
        integrator.step()
        times.append(integrator.t)
        Ps.append(integrator.y)
    X = np.array(Ps)

    kmax = len(Pk) - 1
    rho = X.T[:kmax+1]
    p = X.T[kmax+1:2*kmax+2]
    q = X.T[2*kmax+2:]

    S = 1 - (rho * np.expand_dims(Pk, axis=1)).sum(axis=0)
    I = (rho * np.expand_dims(Pk, axis=1)).sum(axis=0)
    if return_full_data:
        return times, S, I, rho, p, q
    else:
        return times, S, I


def HMF(rho_0, F, R, Pk, tmin=0, tmax=100,
                         tcount=1001, return_full_data=False):
    times = np.linspace(tmin, tmax, tcount)

    X0 = rho_0 * np.ones_like(Pk)

    times, Ps = [], []
    times.append(tmin)
    Ps.append(X0)
    integrator = integrate.RK23(lambda t, x: _dHMF_(x, t, F, R, Pk), tmin, X0, tmax, 0.1)
    while integrator.status != 'finished':
        integrator.step()
        times.append(integrator.t)
        Ps.append(integrator.y)
    X = np.array(Ps)

    rho = X.T[:]

    S = 1 - (rho * np.expand_dims(Pk, axis=1)).sum(axis=0)
    I = (rho * np.expand_dims(Pk, axis=1)).sum(axis=0)
    if return_full_data:
        return times, S, I, rho
    else:
        return times, S, I


def MMF_from_distribution(motifs, hyperstub_d_distribution, rate_function, mu_0,
                          tmin=0, tmax=100, tcount=1001):
    mmf = MMF(motifs, hyperstub_d_distribution, rate_function, mu_0)

    times, Ps = [], []
    times.append(tmin)
    Ps.append(mmf.P_0)

    """ Simulate """
    integrator = integrate.RK23(mmf.f, tmin, mmf.P_0, tmax, 0.1)
    while integrator.status != 'finished':
        integrator.step()
        times.append(integrator.t)
        Ps.append(integrator.y)
    P = np.array(Ps).T

    """ Extract infection data """
    results = mmf.extract_data(P)
    return times, results


class MMF:
    def __init__(self, motifs, hyperstub_d_distribution, rate_function, mu_0):
        self.motifs = motifs
        self.P_D = hyperstub_d_distribution
        self.rate_fcn = rate_function
        self.mu_0 = mu_0

        self.diophantine_cache = {}
        self.rates_cache = {}
        self.rates_avg_cache = {}
        self.split_idx_cache = {}

        self.idx2motifs = []
        self.idx2hyperstub_idxs = []
        for motif in motifs:
            for hyperstub_idx in range(len(motif)):
                self.idx2motifs.append(motif)
                self.idx2hyperstub_idxs.append(hyperstub_idx)

        """ Make all possible combinations of D """
        Ds = np.array(list(itertools.product(
            *[range(hyperstub_d_distribution.shape[i]) for i in range(len(hyperstub_d_distribution.shape))])))
        P_D_flat = hyperstub_d_distribution.flatten()
        self.P_D_sparse = list(map(lambda x: (x[0], x[1][0], x[1][1]),
                                   zip(range(len(Ds)), filter(lambda idx_d_p: idx_d_p[1] > 0, zip(Ds, P_D_flat)))))

        """ Build indices """
        self.nX = len(mu_0)
        self.nEqs_per_state = 0
        self.idx2offset_D = {}
        self.idx2numconfs = {}
        self.P_D_vec = []
        for D_idx, D, P_D in self.P_D_sparse:
            self.idx2offset_D[D_idx] = self.nEqs_per_state
            nEqs_for_D = 1
            for i in range(len(D)):
                motif = self.idx2motifs[i]
                d = D[i]
                nEqs_for_D *= comb(self.nX ** len(motif) + d - 1, d, exact=True)
            self.idx2numconfs[D_idx] = nEqs_for_D
            self.nEqs_per_state += nEqs_for_D
            self.P_D_vec += [P_D] * nEqs_for_D
        self.P_D_vec = np.array(self.P_D_vec)

        """ Find initial configuration """
        self.P_0 = []
        for state_idx in range(self.nX):
            for D_idx, D, P_D in self.P_D_sparse:
                nEqs_for_D = self.idx2numconfs[D_idx]

                for Z_idx in range(nEqs_for_D):
                    Z = self.decode_zs(Z_idx, D, D_idx)

                    """ For given zs, get its probability """
                    curr_idx = 0
                    P_0_conf = mu_0[state_idx]
                    for motif, center_idx in zip(self.idx2motifs, self.idx2hyperstub_idxs):
                        nConf_motif = self.nX ** len(motif)
                        P_Z = []
                        for conf_idx in range(nConf_motif):
                            xs_motif = encode_x_nary(len(motif), conf_idx, self.nX)
                            if xs_motif[center_idx] != state_idx:
                                P_Z.append(0.)
                            else:
                                xs_without_center = np.delete(xs_motif, [center_idx])
                                P_conf_motif = np.prod([mu_0[x] for x in xs_without_center])
                                P_Z.append(P_conf_motif)
                        P_0_conf *= multinomial(Z[curr_idx:curr_idx + nConf_motif])
                        P_0_conf *= np.prod([P_Z[z_idx] ** z for z_idx, z in zip(range(nConf_motif),
                                                                                 Z[
                                                                                 curr_idx:curr_idx + nConf_motif])])
                        curr_idx += nConf_motif
                    self.P_0.append(P_0_conf)
        self.P_0 = np.array(self.P_0)
        print((np.concatenate([self.P_D_vec] * len(mu_0)) * np.array(self.P_0)).sum())

    def get_diophantine_solutions(self, total, N):
        if (total, N) not in self.diophantine_cache:
            solutions = []
            self.add_diophantine_solutions(total, N, solutions, np.zeros(N, dtype=int))
            self.diophantine_cache[(total, N)] = solutions
        return self.diophantine_cache[(total, N)]

    def add_diophantine_solutions(self, total, N, solutions, solution):
        if total > 0:
            for i in range(N):
                solution_new = np.copy(solution)
                solution_new[i] += 1
                self.add_diophantine_solutions(total - 1, N, solutions, solution_new)
        else:
            if tuple(solution) not in solutions:
                solutions.append(tuple(solution))

    def split_idx_for(self, idx, D, D_idx):
        cache_key = (idx, D_idx)
        if cache_key not in self.split_idx_cache:
            motif_Z_idxs = []
            for d, motif in zip(D, self.idx2motifs):
                nEqs_for_D = comb(self.nX ** len(motif) + d - 1, d, exact=True)
                motif_Z_idx = idx % nEqs_for_D
                idx //= nEqs_for_D
                motif_Z_idxs.append(motif_Z_idx)
            self.split_idx_cache[cache_key] = motif_Z_idxs
        return self.split_idx_cache[cache_key]

    def split_idx_from_zs(self, Z, D):
        motif_Z_idxs = []
        curr_idx = 0
        for d, motif in zip(D, self.idx2motifs):
            nConf_motif = self.nX ** len(motif)
            motif_Z_idx = self.encode_z_for_hyperstub(Z[curr_idx:curr_idx + nConf_motif], d, nConf_motif)
            motif_Z_idxs.append(motif_Z_idx)
            curr_idx += nConf_motif
        return motif_Z_idxs

    def decode_z_for_hyperstub(self, Z_sub_idx, d, nConf_motif):
        return self.get_diophantine_solutions(d, nConf_motif)[Z_sub_idx]

    def encode_z_for_hyperstub(self, Z_sub, d, nConf_motif):
        diophantine_solutions = self.get_diophantine_solutions(d, nConf_motif)
        for idx, combination in zip(range(len(diophantine_solutions)), diophantine_solutions):
            if np.array_equal(combination, Z_sub):
                return idx

    def compute_idx_from_split(self, motif_Z_idxs, D):
        idx = 0
        for d, motif, motif_Z_idx in reversed(list(zip(D, self.idx2motifs, motif_Z_idxs))):
            nEqs_for_D = comb(self.nX ** len(motif) + d - 1, d, exact=True)
            idx *= nEqs_for_D
            idx += motif_Z_idx
        return idx

    def decode_zs(self, zs_Z_idx, D, D_idx):
        hyperstub_zs = []
        hyperstub_Z_idxs = self.split_idx_for(zs_Z_idx, D, D_idx)
        for hyperstub_Z_idx, d, motif in zip(hyperstub_Z_idxs, D, self.idx2motifs):
            nConf_motif = self.nX ** len(motif)
            hyperstub_zs.append(self.decode_z_for_hyperstub(hyperstub_Z_idx, d, nConf_motif))
        return np.concatenate(hyperstub_zs)

    def encode_zs(self, hyperstub_zs, D):
        hyperstub_Z_idxs = self.split_idx_from_zs(hyperstub_zs, D)
        Z_idx = self.compute_idx_from_split(hyperstub_Z_idxs, D)
        return Z_idx

    def get_center_jump_target_Z(self, x_new, Z, D):
        new_hyperstub_zs = np.zeros_like(Z)
        curr_idx = 0
        for d, motif, hyperstub_idx in zip(D, self.idx2motifs, self.idx2hyperstub_idxs):
            nConf_motif = self.nX ** len(motif)
            for conf_idx in range(nConf_motif):
                config = encode_x_nary(len(motif), conf_idx, self.nX)
                config[hyperstub_idx] = x_new
                new_conf_idx = decode_x_nary(config, self.nX)
                new_hyperstub_zs[curr_idx + new_conf_idx] += Z[curr_idx + conf_idx]
            curr_idx += nConf_motif
        return new_hyperstub_zs

    def get_motif_jump_target_Z(self, role_idx, target_config_idx, source_config_idx, Z, D):
        Z_new = np.copy(Z)
        curr_idx = 0
        for idx, d, motif, hyperstub_idx in zip(range(len(D)), D, self.idx2motifs, self.idx2hyperstub_idxs):
            nConf_motif = self.nX ** len(motif)
            if role_idx == idx:
                for conf_idx in range(nConf_motif):
                    if conf_idx == target_config_idx:
                        Z_new[curr_idx + conf_idx] += 1
                    if conf_idx == source_config_idx:
                        Z_new[curr_idx + conf_idx] -= 1
            curr_idx += nConf_motif
        return Z_new

    def find_z_idx_of_motif_and_role(self, role_idx, Z_idx, D):
        curr_idx = 0
        for idx, d, motif, hyperstub_idx in zip(range(len(D)), D, self.idx2motifs, self.idx2hyperstub_idxs):
            nConf_motif = self.nX ** len(motif)
            if role_idx == idx:
                return curr_idx + Z_idx
            curr_idx += nConf_motif

    def get_rates_avg(self, P, x_v, goal_role_idx, Z_idx):
        cache_key = (x_v, goal_role_idx, Z_idx)
        if cache_key not in self.rates_avg_cache:
            numerator = np.zeros(self.nX)
            denominator = 1e-10 * np.ones(self.nX)
            for y in range(self.nX):
                for D_prime_idx, D_prime, P_D_prime in self.P_D_sparse:
                    nEqs_for_D = self.idx2numconfs[D_prime_idx]

                    for Z_idx_prime in range(nEqs_for_D):
                        ijk = self.get_ijk(Z_idx_prime, D_prime_idx, x_v)
                        Z_prime = self.decode_zs(Z_idx_prime, D_prime, D_prime_idx)

                        """ Add center node jump rate times multiplicity of config times probability of this state """
                        rates = self.get_rates(x_v, Z_prime, Z_idx_prime, D_prime_idx)
                        ivk = self.find_z_idx_of_motif_and_role(goal_role_idx, Z_idx, D_prime)
                        numerator += rates * Z_prime[ivk] * P[ijk] * P_D_prime
                        denominator += np.ones(self.nX) * Z_prime[ivk] * P[ijk] * P_D_prime
            self.rates_avg_cache[cache_key] = numerator / denominator

        return self.rates_avg_cache[cache_key]

    def get_motif_rates(self, P, Z, D):
        curr_idx = 0
        motif_rates = {}
        for ij, G_i, j in zip(range(len(self.idx2motifs)), self.idx2motifs, self.idx2hyperstub_idxs):
            nConf_motif = self.nX ** len(G_i)
            zs_motif = Z[curr_idx:curr_idx + nConf_motif]

            for k, z_ijk in zip(range(nConf_motif), zs_motif):
                if z_ijk > 0:
                    xs_motif = encode_x_nary(len(G_i), k, self.nX)
                    V_i_no_j = np.delete(list(range(len(G_i))), [j])
                    for v in V_i_no_j:
                        x_v = xs_motif[v]
                        goal_role_idx = ij - (j - v)
                        rates = self.get_rates_avg(P, x_v, goal_role_idx, k)

                        for x_new, rate in zip(range(self.nX), rates):
                            target_xs_motif = np.copy(xs_motif)
                            target_xs_motif[v] = x_new
                            target_xs_motif_idx = decode_x_nary(target_xs_motif, self.nX)
                            target_Z = self.get_motif_jump_target_Z(ij, target_xs_motif_idx, k, Z, D)
                            target_idx = self.encode_zs(target_Z, D)

                            if target_idx in motif_rates:
                                motif_rates[target_idx] = motif_rates[target_idx] + z_ijk * rate
                            else:
                                motif_rates[target_idx] = z_ijk * rate

            curr_idx += nConf_motif
        return motif_rates

    def f(self, t, P):
        print(t, flush=True)
        dP_neighbors = np.zeros_like(P)
        dP_centers = np.zeros_like(P)

        """ Reset cache jump rates """
        self.rates_avg_cache.clear()
        self.split_idx_cache.clear()

        """ Compute center jump rates """
        for x in range(self.nX):
            for D_idx, D, P_D in self.P_D_sparse:
                nEqs_for_D = self.idx2numconfs[D_idx]

                for Z_idx in range(nEqs_for_D):
                    ijk = self.get_ijk(Z_idx, D_idx, x)
                    Z = self.decode_zs(Z_idx, D, D_idx)

                    rates = self.get_rates(x, Z, Z_idx, D_idx)
                    for x_new in range(self.nX):
                        rate = rates[x_new]
                        target_Z = self.get_center_jump_target_Z(x_new, Z, D)
                        target_Z_idx = self.encode_zs(target_Z, D)
                        target_idx = self.get_ijk(target_Z_idx, D_idx, x_new)
                        dP_centers[target_idx] += rate * P[ijk]
                        dP_centers[ijk] -= rate * P[ijk]

        """ Compute neighboring jump rates """
        for x in range(self.nX):
            for D_idx, D, P_D in self.P_D_sparse:
                nEqs_for_D = self.idx2numconfs[D_idx]

                for Z_idx in range(nEqs_for_D):
                    ijk = self.get_ijk(Z_idx, D_idx, x)
                    Z = self.decode_zs(Z_idx, D, D_idx)

                    motif_rates = self.get_motif_rates(P, Z, D)
                    for target_Z_idx, rate in motif_rates.items():
                        target_idx = self.get_ijk(target_Z_idx, D_idx, x)
                        dP_neighbors[target_idx] += rate * P[ijk]
                        dP_neighbors[ijk] -= rate * P[ijk]

        dP = dP_centers + dP_neighbors
        return dP

    def get_ijk(self, Z_idx, D_idx, x):
        return self.nEqs_per_state * x + self.idx2offset_D[D_idx] + Z_idx

    def extract_data(self, P):
        results = []
        for x in range(self.nX):
            results.append((P[self.nEqs_per_state * x:self.nEqs_per_state * (x + 1)]
                            * np.expand_dims(self.P_D_vec, axis=1)).sum(axis=0))
        return results

    def get_rates(self, x, Z, Z_idx, D_idx):
        if (x, Z_idx, D_idx) not in self.rates_cache:
            self.rates_cache[(x, Z_idx, D_idx)] = self.rate_fcn(x, Z, self.idx2motifs, self.idx2hyperstub_idxs)
        return self.rates_cache[(x, Z_idx, D_idx)]
