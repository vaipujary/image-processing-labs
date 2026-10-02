import torch
from lab3.lab3 import forward, adjoint, load_pgm, load_kernel


def _parameters(sigma_x, p, q, T):
    """Validate positive scalar scales and 1 <= q < p <= 2; return None."""
    if sigma_x <= 0 or T <= 0 or not (1 <= q < p <= 2):
        raise ValueError("Require sigma_x > 0, T > 0, and 1 <= q < p <= 2")


def rho(delta, sigma_x, p, q, T):
    """Evaluate Eq. (3) on a float64 tensor; return its elementwise potential.

    sigma_x and T are positive scales; 1 <= q < p <= 2. The reciprocal
    form avoids the negative power. Only the power's argument is floored
    at 1e-8; the numerator retains the actual difference, including zero.
    """
    _parameters(sigma_x, p, q, T)
    magnitude = delta.abs()
    u = (magnitude.clamp_min(1e-8) / (T * sigma_x)).pow(p - q)
    return magnitude.pow(p) / (p * sigma_x**p * (1 + u))


def rho_prime(delta, sigma_x, p, q, T):
    """Evaluate Eq. (4); return a float64 influence tensor shaped like delta.

    Inputs have the same meaning as rho. Below the 1e-8 numerical floor,
    differentiate the implemented floored denominator exactly.
    """
    _parameters(sigma_x, p, q, T)
    magnitude = delta.abs()
    u = (magnitude.clamp_min(1e-8) / (T * sigma_x)).pow(p - q)
    factor = (1 + (q / p) * u) / (1 + u).square()
    factor = torch.where(magnitude < 1e-8, 1 / (1 + u), factor)
    return delta.sign() * magnitude.pow(p - 1) * factor / sigma_x**p


def neighbor_diffs(x):
    """Return (diffs, valid), each (8,H,W), for a float64 (H,W) image.

    Plane order is NW, N, NE, W, E, SW, S, SE. At pixel (h,w), plane
    (dh,dw) is x[h,w]-x[h+dh,w+dw]. Invalid differences are zero, with
    a false bool mask. Only the eight offsets are looped over, never pixels.
    """
    offsets = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
               (0, 1), (1, -1), (1, 0), (1, 1))
    rows = torch.arange(x.shape[0], device=x.device)[:, None]
    cols = torch.arange(x.shape[1], device=x.device)[None, :]
    differences, masks = [], []
    for dh, dw in offsets:
        valid = ((rows + dh >= 0) & (rows + dh < x.shape[0]) &
                 (cols + dw >= 0) & (cols + dw < x.shape[1]))
        diff = x - torch.roll(x, shifts=(-dh, -dw), dims=(0, 1))
        differences.append(torch.where(valid, diff, 0.0))
        masks.append(valid)
    return torch.stack(differences), torch.stack(masks)


def _base_weights(x):
    """Return (8,1,1) neighbor weights with x's dtype and device."""
    return x.new_tensor([1/12, 1/6, 1/12, 1/6, 1/6, 1/12, 1/6, 1/12])[:, None, None]


def surrogate_weights(x, sigma_x, p, q, T):
    """Return Eq. (14)-(15) float64 (8,H,W) weights for (H,W) image x.

    p must be 2; sigma_x, q, T are the rho parameters. Invalid pairs have
    zero weight. Differences at or below 1e-8 use b/(2*sigma_x**2).
    """
    _parameters(sigma_x, p, q, T)
    if p != 2:
        raise ValueError("The bounded-curvature surrogate requires p=2")
    diffs, valid = neighbor_diffs(x)
    magnitude = diffs.abs()
    u = (magnitude.clamp_min(1e-8) / (T * sigma_x)).pow(p - q)
    factor = (1 + (q / p) * u) / (1 + u).square()
    factor = torch.where(magnitude <= 1e-8, torch.ones_like(factor), factor)
    return valid * _base_weights(x) * factor / (2 * sigma_x**2)


def true_cost(x, y, a, sigma_w, sigma_x, p, q, T):
    """Return differentiable scalar float64 MAP cost from Eq. (1).

    x,y: (H,W) float64 images; a: odd-sized float64 blur kernel;
    sigma_w>0: noise SD; sigma_x,p,q,T: rho parameters. Masking occurs
    after rho, and the 1/2 corrects double counting of neighboring pairs.
    """
    if sigma_w <= 0:
        raise ValueError("sigma_w must be positive")
    diffs, valid = neighbor_diffs(x)
    prior = 0.5 * (_base_weights(x) * valid * rho(diffs, sigma_x, p, q, T)).sum()
    return (forward(x, a) - y).square().sum() / (2 * sigma_w**2) + prior


def true_gradient(x, y, a, sigma_w, sigma_x, p, q, T):
    """Return the (H,W) float64 gradient of true_cost using autograd.

    Arguments match true_cost; x must have requires_grad=True. No
    handwritten gradient is used by the reconstruction algorithm.
    """
    with torch.enable_grad():
        return torch.autograd.grad(true_cost(x, y, a, sigma_w, sigma_x, p, q, T), x)[0]


def optimal_step_size(x, g, a, sigma_w, sigma_x, p, q, T):
    """Return scalar float64 alpha=(g^T g)/(g^T H g), Eq. (17)-(18).

    x,g: (H,W) current image and gradient; a: blur kernel; remaining
    arguments match true_cost. Weights are frozen at x; no y is needed.
    A zero gradient returns zero so a stationary image remains unchanged.
    """
    if sigma_w <= 0:
        raise ValueError("sigma_w must be positive")
    with torch.no_grad():
        weights = surrogate_weights(x, sigma_x, p, q, T)
        gd, _ = neighbor_diffs(g)
        numerator = g.square().sum()
        denominator = forward(g, a).square().sum() / sigma_w**2 + (weights * gd.square()).sum()
        if numerator == 0:
            return numerator
        if not torch.isfinite(denominator) or denominator <= 0:
            raise ValueError("Surrogate directional curvature must be positive and finite")
        return numerator / denominator


def reconstruct(y, a, sigma_w, sigma_x, p, q, T, num_iters, omega):
    """Return (x, cost_history) after num_iters surrogate descent updates.

    Start at y, a float64 (H,W) tensor. a is a float64 odd-sized blur
    kernel; scale and potential parameters match true_cost; p=2 and
    0<omega<2. The history has num_iters+1 true costs starting with f(y).
    x is unconstrained; neither measurements nor estimates are clipped.
    """
    _parameters(sigma_x, p, q, T)
    if p != 2 or not 0 < omega < 2:
        raise ValueError("Require p=2 and 0<omega<2")
    if not isinstance(num_iters, int) or num_iters < 0:
        raise ValueError("num_iters must be a nonnegative integer")
    if y.dtype != torch.float64 or a.dtype != torch.float64:
        raise ValueError("Use float64 for both images and kernels")
    x = y.detach().clone()
    with torch.no_grad():
        history = [true_cost(x, y, a, sigma_w, sigma_x, p, q, T).item()]
    for _ in range(num_iters):
        x = x.detach().requires_grad_(True)
        g = true_gradient(x, y, a, sigma_w, sigma_x, p, q, T)
        alpha = optimal_step_size(x, g, a, sigma_w, sigma_x, p, q, T)
        with torch.no_grad():
            x = x - omega * alpha * g
            history.append(true_cost(x, y, a, sigma_w, sigma_x, p, q, T).item())
    return x.detach(), history
