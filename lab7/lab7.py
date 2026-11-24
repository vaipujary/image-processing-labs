def neighbor_stats(x):
    """Return weighted neighbor sums and weights for the Lab 3 GMRF.

    x: (H, W) float64 PyTorch tensor. Returns (nbr_sum, c), both
    (H, W) float64 tensors. Missing neighbors are excluded by zero padding;
    g = [[1,2,1],[2,0,2],[1,2,1]] / 12, with no wraparound.
    """
    import torch
    import torch.nn.functional as F
    _check_image(x)
    padded = F.pad(x, (1, 1, 1, 1))
    mask = F.pad(torch.ones_like(x), (1, 1, 1, 1))
    nbr_sum, c = torch.zeros_like(x), torch.zeros_like(x)
    h, width = x.shape
    for dr, dc, weight in _neighbors():
        nbr_sum += weight * padded[1+dr:1+dr+h, 1+dc:1+dc+width]
        c += weight * mask[1+dr:1+dr+h, 1+dc:1+dc+width]
    return nbr_sum, c


def gibbs_sweep_gmrf(x, sigma_x):
    """Draw one four-color Gibbs sweep from the intrinsic GMRF prior.

    x: (H, W) float64 tensor with at least two pixels; sigma_x: positive
    prior scale. Returns a new tensor of the same shape, dtype, and device.
    No clipping or mean constraint is applied to the Markov chain.
    """
    import torch
    _check_image(x)
    _positive(sigma_x)
    out = x.clone()
    for row, col in ((0, 0), (0, 1), (1, 0), (1, 1)):
        sums, c = neighbor_stats(out)
        weights = c[row::2, col::2]
        if torch.any(weights == 0):
            raise ValueError("The unanchored one-pixel prior has no conditional density")
        block = out[row::2, col::2]
        block.copy_(sums[row::2, col::2] / weights
                    + sigma_x / weights.sqrt() * torch.randn_like(block))
    return out


def qggmrf_delta_energy(x, w, sigma_x, p, q, T):
    """Return each single-pixel replacement energy change in equation 7.

    x, w: matching (H, W) float64 tensors of current/proposed pixel values.
    sigma_x, T: positive scales; p, q: exponents with 1 <= p <= q <= 2.
    Returns an (H, W) float64 tensor, holding all other pixels fixed at x.
    """
    _check_image(x)
    _check_image(w)
    if w.shape != x.shape or w.device != x.device:
        raise ValueError("x and w must have the same shape and device")
    _check_potential(sigma_x, p, q, T)
    return _delta_block(x, w, sigma_x, p, q, T, 0, 0, 1)


def metropolis_sweep_qggmrf(x, sigma_x, p, q, T, sigma_prop):
    """Run a coordinatewise random-walk Metropolis sweep in four colors.

    x: (H, W) float64 tensor; sigma_x, T, sigma_prop: positive scales;
    p, q: exponents, 1 <= p <= q <= 2. Returns (x_new, accept_rate),
    a new float64 tensor and the fraction of all pixel proposals accepted.
    Neighbor values are refreshed after each color, and log acceptance
    avoids exponential overflow. The input tensor is not modified.
    """
    import torch
    _check_image(x)
    _check_potential(sigma_x, p, q, T)
    _positive(sigma_prop)
    out, count = x.clone(), x.new_zeros(())
    for row, col in ((0, 0), (0, 1), (1, 0), (1, 1)):
        block = out[row::2, col::2]
        proposal = block + sigma_prop * torch.randn_like(block)
        delta = _delta_block(out, proposal, sigma_x, p, q, T, row, col, 2)
        accept = torch.log(torch.rand_like(block)) < -delta
        block.copy_(torch.where(accept, proposal, block))
        count += accept.sum()
    return out, float(count / x.numel())


def gibbs_sweep_posterior(x, y, sigma_w, sigma_x):
    """Draw one four-color Gibbs sweep for the denoising posterior.

    x, y: matching (H, W) float64 tensors, current state and noisy image.
    sigma_w, sigma_x: positive noise/prior scales. Returns a new (H, W)
    float64 tensor drawn using equation 9, without clipping intensities.
    """
    import torch
    _check_image(x)
    _check_image(y)
    if x.shape != y.shape or x.device != y.device:
        raise ValueError("x and y must have the same shape and device")
    _positive(sigma_w, sigma_x)
    out = x.clone()
    for row, col in ((0, 0), (0, 1), (1, 0), (1, 1)):
        sums, c = neighbor_stats(out)
        precision = 1 / sigma_w**2 + c[row::2, col::2] / sigma_x**2
        mean = (y[row::2, col::2] / sigma_w**2
                + sums[row::2, col::2] / sigma_x**2) / precision
        block = out[row::2, col::2]
        block.copy_(mean + torch.randn_like(block) / precision.sqrt())
    return out


def posterior_statistics(y, sigma_w, sigma_x, num_burn, num_keep):
    """Estimate posterior moments from num_keep post-burn-in sweeps.

    y: (H, W) float64 noisy image; sigma_w, sigma_x: positive scales;
    num_burn: nonnegative integer; num_keep: positive integer. Starts at y.
    Returns (mean, std, energy_history): two (H, W) float64 tensors and
    a list of num_burn + num_keep floats, one energy after each full sweep.
    Uses Welford moments and divisor num_keep for the empirical variance.
    """
    import torch
    _check_image(y)
    _positive(sigma_w, sigma_x)
    if not isinstance(num_burn, int) or num_burn < 0:
        raise ValueError("num_burn must be a nonnegative integer")
    if not isinstance(num_keep, int) or num_keep < 1:
        raise ValueError("num_keep must be a positive integer")
    x, mean, m2 = y.clone(), torch.zeros_like(y), torch.zeros_like(y)
    history = []
    for sweep in range(num_burn + num_keep):
        x = gibbs_sweep_posterior(x, y, sigma_w, sigma_x)
        history.append(float(posterior_energy(x, y, sigma_w, sigma_x)))
        if sweep >= num_burn:
            n = sweep - num_burn + 1
            delta = x - mean
            mean += delta / n
            m2 += delta * (x - mean)
    return mean, (m2 / num_keep).clamp_min(0).sqrt(), history


def gmrf_energy(x, sigma_x):
    """Return scalar GMRF energy for float64 image x and positive sigma_x.

    Each unordered in-image neighboring pair contributes exactly once.
    """
    _positive(sigma_x)
    sums, c = neighbor_stats(x)
    return (x * (c * x - sums)).sum() / (2 * sigma_x**2)


def posterior_energy(x, y, sigma_w, sigma_x):
    """Return scalar equation 8 energy for float64 images x, y and scales.

    x and y have identical (H, W) shape; both scales must be positive.
    """
    _positive(sigma_w, sigma_x)
    return (x - y).square().sum() / (2 * sigma_w**2) + gmrf_energy(x, sigma_x)


def qggmrf_energy(x, sigma_x, p, q, T):
    """Return scalar equation 3 energy, counting unordered pairs once.

    x: (H, W) float64 tensor; sigma_x, T > 0; 1 <= p <= q <= 2.
    Returns a scalar float64 tensor with no penalty for missing neighbors.
    """
    _check_image(x)
    _check_potential(sigma_x, p, q, T)
    energy = x.new_zeros(())
    for a, b, weight in ((x[:, :-1], x[:, 1:], 2/12),
                         (x[:-1, :], x[1:, :], 2/12),
                         (x[:-1, :-1], x[1:, 1:], 1/12),
                         (x[:-1, 1:], x[1:, :-1], 1/12)):
        energy += weight * _rho(a-b, sigma_x, p, q, T).sum()
    return energy


def _rho(delta, sigma_x, p, q, T):
    """Return elementwise equation 4 potential for tensor delta and scales.

    Inputs are float64 differences, positive scales, and 1 <= p <= q <= 2;
    returns a float64 tensor of the same shape (including rho(0) = 0).
    """
    a = delta.abs() / sigma_x
    power = (a / T).pow(q-p)
    return a.pow(p) / p * (power / (1 + power))


def _delta_block(x, proposal, sigma_x, p, q, T, row, col, step):
    """Score replacements for x[row::step,col::step] against current x.

    proposal has the selected block's shape; model parameters match _rho.
    row, col, step are integer slice parameters. Returns block-shaped
    float64 local changes. Loops over eight offsets, never over pixels.
    """
    import torch
    import torch.nn.functional as F
    h, width = x.shape
    padded = F.pad(x, (1, 1, 1, 1))
    mask = F.pad(torch.ones_like(x), (1, 1, 1, 1))
    current = x[row::step, col::step]
    result = torch.zeros_like(current)
    for dr, dc, weight in _neighbors():
        neighbor = padded[1+dr+row:1+dr+h:step, 1+dc+col:1+dc+width:step]
        valid = mask[1+dr+row:1+dr+h:step, 1+dc+col:1+dc+width:step]
        result += weight * valid * (_rho(proposal-neighbor, sigma_x, p, q, T)
                                    - _rho(current-neighbor, sigma_x, p, q, T))
    return result


def _neighbors():
    """Take no inputs; return the eight (row offset, column offset, weight) tuples."""
    return ((-1, -1, 1/12), (-1, 0, 2/12), (-1, 1, 1/12),
            (0, -1, 2/12), (0, 1, 2/12),
            (1, -1, 1/12), (1, 0, 2/12), (1, 1, 1/12))


def _check_image(x):
    """Validate nonempty 2D float64 tensor x; return None or raise ValueError."""
    import torch
    if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.numel() == 0 or x.dtype != torch.float64:
        raise ValueError("Expected a nonempty (H, W) torch.float64 tensor")


def _positive(*values):
    """Validate finite positive scalar inputs; return None or raise ValueError."""
    import math
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Scales must be finite and positive")


def _check_potential(sigma_x, p, q, T):
    """Validate QGGMRF scales and exponents; return None or raise ValueError."""
    _positive(sigma_x, T)
    if not 1 <= p <= q <= 2:
        raise ValueError("Expected 1 <= p <= q <= 2")
