"""Gaussian mixture EM and merge-based order estimation for Lab 6.

Contracts are recorded in SPECIFICATIONS.json. All computations use float64.
"""
import numpy as np

EPSILON = 1e-6

def _log_joint(Y, weights, means, covariances):
    Y, weights, means, covariances = [np.asarray(a, dtype=np.float64)
                                     for a in (Y, weights, means, covariances)]
    if Y.ndim != 2 or min(Y.shape) < 1 or weights.ndim != 1 or len(weights) < 1:
        raise ValueError('Invalid data or weight shape')
    N, p = Y.shape
    M = len(weights)
    if means.shape != (M, p) or covariances.shape != (M, p, p):
        raise ValueError('Invalid parameter shapes')
    if any(not np.isfinite(a).all() for a in (Y, weights, means, covariances)):
        raise ValueError('Inputs must be finite')
    if np.any(weights <= 0) or not np.isclose(weights.sum(), 1., rtol=0, atol=1e-10):
        raise ValueError('Weights must be positive and sum to one')
    result = np.empty((N, M))
    for m in range(M):
        R = covariances[m]
        if not np.allclose(R, R.T, rtol=0, atol=1e-10):
            raise ValueError('Covariances must be symmetric')
        try:
            L = np.linalg.cholesky(R)
        except np.linalg.LinAlgError as exc:
            raise ValueError('Covariances must be positive definite') from exc
        z = np.linalg.solve(L, (Y - means[m]).T)
        result[:, m] = np.log(weights[m]) - .5 * (
            p * np.log(2 * np.pi) + 2 * np.log(np.diag(L)).sum()
            + np.einsum('ij,ij->j', z, z))
    return result

def _logsumexp(a):
    peak = np.max(a, axis=1, keepdims=True)
    return peak[:, 0] + np.log(np.exp(a - peak).sum(axis=1))

def e_step(Y, weights, means, covariances):
    """Equation (6); see the e_step specification for the full contract."""
    joint = _log_joint(Y, weights, means, covariances)
    return np.exp(joint - _logsumexp(joint)[:, None])

def m_step(Y, P):
    """Equations (7)-(8), with ML normalization and an additive ridge."""
    Y, P = np.asarray(Y, dtype=np.float64), np.asarray(P, dtype=np.float64)
    if Y.ndim != 2 or min(Y.shape) < 1 or P.ndim != 2 or P.shape[0] != len(Y) or P.shape[1] < 1:
        raise ValueError('Invalid data or responsibility shapes')
    if not np.isfinite(Y).all() or not np.isfinite(P).all() or np.any(P < 0):
        raise ValueError('Invalid data or probabilities')
    if not np.allclose(P.sum(axis=1), 1., rtol=0, atol=1e-10):
        raise ValueError('Responsibility rows must sum to one')
    counts = P.sum(axis=0)
    if np.any(counts <= 0):
        raise ValueError('Empty component')
    weights = counts / len(Y)
    means = P.T @ Y / counts[:, None]
    covariances = np.empty((len(counts), Y.shape[1], Y.shape[1]))
    for m in range(len(counts)):
        centered = Y - means[m]
        R = (centered.T * P[:, m]) @ centered / counts[m]
        covariances[m] = (R + R.T) / 2 + EPSILON * np.eye(Y.shape[1])
    return weights, means, covariances

def log_likelihood(Y, weights, means, covariances):
    """Total observed-data log likelihood, equation (3)."""
    return float(_logsumexp(_log_joint(Y, weights, means, covariances)).sum())

def run_em(Y, weights, means, covariances, num_iterations):
    """Fixed-length EM, returning the initial and subsequent likelihoods."""
    if not isinstance(num_iterations, (int, np.integer)) or num_iterations < 0:
        raise ValueError('num_iterations must be a nonnegative integer')
    weights, means, covariances = [np.array(a, dtype=np.float64, copy=True)
                                    for a in (weights, means, covariances)]
    history = [log_likelihood(Y, weights, means, covariances)]
    for _ in range(num_iterations):
        P = e_step(Y, weights, means, covariances)
        weights, means, covariances = m_step(Y, P)
        history.append(log_likelihood(Y, weights, means, covariances))
    return weights, means, covariances, history

def mdl(Y, weights, means, covariances):
    """MDL with the lab's log(N*p) penalty, equations (12)-(13)."""
    N, p = np.shape(Y)
    M = len(weights)
    parameters = M * (1 + p + p * (p + 1) / 2) - 1
    return float(-log_likelihood(Y, weights, means, covariances)
                 + .5 * parameters * np.log(N * p))

def _closest_pair(Y, weights, means, covariances):
    _log_joint(Y, weights, means, covariances)  # Validate the input contract.
    if len(weights) < 2:
        raise ValueError('At least two components are required for merging')
    w, mu, R = [np.asarray(a) for a in (weights, means, covariances)]
    logdet = np.linalg.slogdet(R)[1]
    best = None
    for l in range(len(w)):
        for m in range(l + 1, len(w)):
            merged_w = w[l] + w[m]
            merged_mu = (w[l] * mu[l] + w[m] * mu[m]) / merged_w
            dl, dm = mu[l] - merged_mu, mu[m] - merged_mu
            merged_R = (w[l] * (R[l] + np.outer(dl, dl))
                        + w[m] * (R[m] + np.outer(dm, dm))) / merged_w
            merged_R = (merged_R + merged_R.T) / 2
            logdet_merged = np.linalg.slogdet(merged_R)[1]
            distance = len(Y) / 2 * (w[l] * (logdet_merged - logdet[l])
                                    + w[m] * (logdet_merged - logdet[m]))
            if best is None or distance < best[0]:
                best = (float(distance), l, m, merged_w, merged_mu, merged_R)
    return best

def merge_closest(Y, weights, means, covariances):
    """Merge the pair minimizing equation (14), using (15)-(16)."""
    _, l, m, merged_w, merged_mu, merged_R = _closest_pair(Y, weights, means, covariances)
    w, mu, R = [np.array(a, dtype=np.float64, copy=True)
                for a in (weights, means, covariances)]
    w[l], mu[l], R[l] = merged_w, merged_mu, merged_R
    return np.delete(w, m, 0), np.delete(mu, m, 0), np.delete(R, m, 0)

def estimate_order(Y, max_order, initial_means, num_iterations):
    """Warm-start search from max_order down to one, with merge events."""
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2 or min(Y.shape) < 1:
        raise ValueError('Y must be a nonempty matrix')
    if not isinstance(max_order, (int, np.integer)) or max_order < 1:
        raise ValueError('max_order must be a positive integer')
    if not isinstance(num_iterations, (int, np.integer)) or num_iterations < 1:
        raise ValueError('num_iterations must be a positive integer')
    N, p = Y.shape
    w = np.full(max_order, 1 / max_order)
    mu = np.array(initial_means, dtype=np.float64, copy=True)
    R = np.tile(np.eye(p), (max_order, 1, 1))
    models, trace, merges = {}, [], []
    cumulative = 0
    for M in range(max_order, 0, -1):
        w, mu, R, history = run_em(Y, w, mu, R, num_iterations)
        penalty = .5 * (M * (1 + p + p * (p + 1) / 2) - 1) * np.log(N * p)
        for value in history[1:]:
            cumulative += 1
            trace.append(dict(iteration=cumulative, order=M, event='em', mdl=float(-value + penalty)))
        models[M] = dict(weights=w.copy(), means=mu.copy(), covariances=R.copy(),
                         mdl=mdl(Y, w, mu, R), history=history.copy())
        if M > 1:
            distance, l, m, *_ = _closest_pair(Y, w, mu, R)
            w, mu, R = merge_closest(Y, w, mu, R)
            merges.append(dict(iteration=cumulative, from_order=M, pair=[l, m], distance=distance,
                               initial_means=mu.copy()))
            trace.append(dict(iteration=cumulative, order=M - 1, event='merge', mdl=mdl(Y, w, mu, R)))
    best_order = min(models, key=lambda M: (models[M]['mdl'], M))
    return dict(models=models, trace=trace, merges=merges, best_order=best_order)
