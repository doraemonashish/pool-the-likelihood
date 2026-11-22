"""Exponential-kernel Hawkes processes: exact simulation and exact conditional distributions.

Intensity: lambda(t) = mu + sum_{t_i < t} alpha * beta * exp(-beta (t - t_i)).
The branching ratio (mean number of children per event) is ``alpha``; the process is stationary
for alpha < 1 and "critical" as alpha -> 1 (where real order-flow data are often estimated to sit).

Between events the excitation E(t) = lambda(t) - mu decays deterministically, so the next
inter-event time has survival S(tau) = exp(-mu tau - (E / beta)(1 - exp(-beta tau))), where E is
the excitation just after the last event. Sampling uses the exact two-competing-clocks method of
Dassios & Zhao (2013), vectorised over a batch of independent sequences.
"""
from __future__ import annotations

import numpy as np


def sample_params(rng: np.random.Generator, B: int, mu=(0.2, 2.0), alpha=(0.0, 0.9), beta=(0.5, 10.0)) -> dict:
    """mu and beta log-uniform, alpha uniform."""
    lu = lambda lo, hi: np.exp(rng.uniform(np.log(lo), np.log(hi), B))  # noqa: E731
    return {"mu": lu(*mu), "alpha": rng.uniform(alpha[0], alpha[1], B), "beta": lu(*beta)}


def simulate(mu, alpha, beta, n_events: int, rng: np.random.Generator) -> dict:
    """Simulate B sequences of exactly ``n_events`` events.

    Returns times (B, N), intervals (B, N) (interval *before* each event, the first measured from 0),
    and E_after (B, N): excitation lambda - mu immediately after each event (including its own jump).
    """
    mu, alpha, beta = (np.asarray(x, dtype=np.float64) for x in (mu, alpha, beta))
    B = mu.shape[0]
    t = np.zeros(B)
    E = np.zeros(B)
    times = np.zeros((B, n_events))
    E_after = np.zeros((B, n_events))
    for i in range(n_events):
        u1 = rng.random(B)
        u2 = rng.random(B)
        tau1 = -np.log(u1) / mu
        with np.errstate(divide="ignore", invalid="ignore"):
            D = 1.0 + beta * np.log(u2) / np.where(E > 0, E, 1.0)
        D = np.where(E > 0, D, -1.0)
        with np.errstate(divide="ignore"):
            tau2 = np.where(D > 0, -np.log(np.where(D > 0, D, 1.0)) / beta, np.inf)
        tau = np.minimum(tau1, tau2)
        t = t + tau
        E = E * np.exp(-beta * tau) + alpha * beta
        times[:, i] = t
        E_after[:, i] = E
    intervals = np.diff(np.concatenate([np.zeros((B, 1)), times], axis=1), axis=1)
    return {"times": times, "intervals": intervals, "E_after": E_after}


def survival(tau, mu, E, beta):
    """S(tau) for the next interval given the excitation E just after the last event (broadcasts)."""
    tau = np.asarray(tau, dtype=np.float64)
    with np.errstate(over="ignore"):
        return np.exp(-mu * tau - (E / beta) * (1.0 - np.exp(-beta * tau)))


def bin_edges(n_bins: int = 48, lo: float = 1e-3, hi: float = 1e2) -> np.ndarray:
    """Edges [0, lo, ..., hi, inf]: one underflow bin, n_bins-2 log-spaced bins, one overflow bin."""
    inner = np.exp(np.linspace(np.log(lo), np.log(hi), n_bins - 1))
    return np.concatenate([[0.0], inner, [np.inf]])


def to_bins(tau: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(edges, tau, side="right") - 1, 0, len(edges) - 2)


def oracle_bin_probs(mu, E, beta, edges: np.ndarray) -> np.ndarray:
    """Exact probability of each interval bin for the next event. mu, E, beta broadcast to shape (...)."""
    mu, E, beta = np.broadcast_arrays(np.asarray(mu, float), np.asarray(E, float), np.asarray(beta, float))
    S = survival(edges[None, :], mu.reshape(-1, 1), E.reshape(-1, 1), beta.reshape(-1, 1))
    S[:, -1] = 0.0
    P = np.clip(S[:, :-1] - S[:, 1:], 1e-12, None)
    P = P / P.sum(1, keepdims=True)
    return P.reshape(*mu.shape, len(edges) - 1)


def poisson_bin_probs(rate: np.ndarray, edges: np.ndarray) -> np.ndarray:
    rate = np.asarray(rate, float)
    S = np.exp(-rate.reshape(-1, 1) * edges[None, :])
    S[:, -1] = 0.0
    P = np.clip(S[:, :-1] - S[:, 1:], 1e-12, None)
    P = P / P.sum(1, keepdims=True)
    return P.reshape(*rate.shape, len(edges) - 1)


# ----------------------------------------------------------------------------- power-law kernel
# phi(x) = alpha * (gamma - 1) / c * (1 + x / c) ** (-gamma), integrating to alpha (the branching ratio).
# Long memory: the tail decays like a power of the time gap, as estimated for order flow in real markets.
def sample_params_powerlaw(rng: np.random.Generator, B: int, mu=(0.2, 2.0), alpha=(0.0, 0.9), c=(0.1, 2.0),
                           gamma=(1.5, 3.0)) -> dict:
    lu = lambda lo, hi: np.exp(rng.uniform(np.log(lo), np.log(hi), B))  # noqa: E731
    return {"mu": lu(*mu), "alpha": rng.uniform(alpha[0], alpha[1], B), "c": lu(*c), "gamma": rng.uniform(gamma[0], gamma[1], B)}


def _pl_kernel_sum(t: np.ndarray, past: np.ndarray, alpha, c, gamma) -> np.ndarray:
    """sum_j phi(t - past_j) for each row; past (B, k), t (B,)."""
    if past.shape[1] == 0:
        return np.zeros(len(t))
    x = np.maximum(t[:, None] - past, 0.0)
    amp = (alpha * (gamma - 1) / c)[:, None]
    return (amp * (1 + x / c[:, None]) ** (-gamma[:, None])).sum(1)


def simulate_powerlaw(mu, alpha, c, gamma, n_events: int, rng: np.random.Generator, max_iter: int = 100000) -> dict:
    """Exact simulation by Ogata thinning (the intensity only decays between events, so its current value
    bounds it from above). Returns times, intervals and E_after (excitation just after each event)."""
    mu, alpha, c, gamma = (np.asarray(x, dtype=np.float64) for x in (mu, alpha, c, gamma))
    B = len(mu)
    times = np.zeros((B, n_events))
    t = np.zeros(B)
    for i in range(n_events):
        done = np.zeros(B, dtype=bool)
        past = times[:, :i]
        for _ in range(max_iter):
            lam_bar = mu + _pl_kernel_sum(t, past, alpha, c, gamma)
            tc = t + rng.exponential(1.0 / lam_bar)
            lam_c = mu + _pl_kernel_sum(tc, past, alpha, c, gamma)
            accept = rng.random(B) * lam_bar <= lam_c
            newly = ~done & accept
            times[newly, i] = tc[newly]
            t = np.where(done, t, tc)
            done |= newly
            if done.all():
                break
    E = np.zeros((B, n_events))
    for i in range(n_events):
        E[:, i] = _pl_kernel_sum(times[:, i], times[:, :i + 1], alpha, c, gamma)
    intervals = np.diff(np.concatenate([np.zeros((B, 1)), times], axis=1), axis=1)
    return {"times": times, "intervals": intervals, "E_after": E}


def oracle_bin_probs_powerlaw(times: np.ndarray, mu, alpha, c, gamma, edges: np.ndarray) -> np.ndarray:
    """Exact next-interval bin probabilities after every event, (B, N, n_bins).

    Compensator over [t_i, t_i + tau]: mu * tau + sum_j [Phi(t_i - t_j + tau) - Phi(t_i - t_j)],
    with Phi(x) = alpha * (1 - (1 + x / c) ** (1 - gamma)).
    """
    B, N = times.shape
    ef = edges[1:-1]
    a, cc, g = alpha[:, None, None], c[:, None, None], gamma[:, None, None]
    Phi = lambda x: a * (1 - (1 + x / cc) ** (1 - g))  # noqa: E731
    out = np.zeros((B, N, len(edges) - 1))
    for i in range(N):
        x0 = (times[:, i:i + 1] - times[:, :i + 1])[:, :, None]
        comp = mu[:, None] * ef[None, :] + (Phi(x0 + ef[None, None, :]) - Phi(x0)).sum(1)
        S = np.concatenate([np.ones((B, 1)), np.exp(-comp), np.zeros((B, 1))], 1)
        P = np.clip(S[:, :-1] - S[:, 1:], 1e-12, None)
        out[:, i] = P / P.sum(1, keepdims=True)
    return out
