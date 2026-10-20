import torch
from functools import lru_cache


def kernel_dft(a, shape):
    """Embed odd float64 kernel a at its circular origin; return complex128 DFT of shape (H,W)."""
    if a.ndim != 2 or any(k % 2 != 1 for k in a.shape):
        raise ValueError('The kernel must be two-dimensional and odd-sized')
    ac = a.new_zeros(shape)
    ka, kb = a.shape
    ac[:ka, :kb] = a
    return torch.fft.fft2(torch.roll(ac, (-(ka // 2), -(kb // 2)), (0, 1)))


def prox_data(v, y, a_hat, sigma_w, sigma_p):
    """Return F(v) for float64 (H,W) anchor v and data y, complex DFT a_hat, and positive noise/prox SDs."""
    if sigma_w <= 0 or sigma_p <= 0:
        raise ValueError('Noise and proximal scales must be positive')
    v_hat = torch.fft.fft2(v)
    residual = torch.fft.fft2(y) - a_hat * v_hat
    denominator = a_hat.abs().square() / sigma_w**2 + 1 / sigma_p**2
    return v + torch.fft.ifft2(a_hat.conj() * residual / sigma_w**2 / denominator).real


def _pairs():
    """Return four unordered neighbor slice pairs and weights; boundary-crossing pairs are absent."""
    return (( (slice(None,-1),slice(None)), (slice(1,None),slice(None)), 1/6),
            ( (slice(None),slice(None,-1)), (slice(None),slice(1,None)), 1/6),
            ( (slice(None,-1),slice(None,-1)), (slice(1,None),slice(1,None)), 1/12),
            ( (slice(None,-1),slice(1,None)), (slice(1,None),slice(None,-1)), 1/12))


def _prior_stats(x, sigma_x, p, q, T):
    """Return prior gradient and four frozen curvature arrays for float64 image x and QGGMRF parameters."""
    if sigma_x <= 0 or T <= 0 or p != 2 or not 1 <= q < p:
        raise ValueError('Require positive scales, p=2 and 1<=q<p')
    grad = torch.zeros_like(x)
    curvatures = []
    for left, right, b in _pairs():
        d = x[left] - x[right]
        u = (d.abs().clamp_min(1e-8) / (T * sigma_x)).pow(p-q)
        factor = (1+(q/p)*u) / (1+u).square()
        exact_factor = torch.where(d.abs() < 1e-8, 1/(1+u), factor)
        influence = b * d * exact_factor / sigma_x**2
        grad[left] += influence
        grad[right] -= influence
        curvatures.append(b * torch.where(d.abs() <= 1e-8, 1., factor) / sigma_x**2)
    return grad, curvatures


def _prox_qggmrf(v, start, sigma_n, sigma_x, p, q, T, num_iters):
    """Solve QGGMRF prox at anchor v from float64 start; return image after num_iters surrogate steepest-descent steps."""
    if sigma_n <= 0:
        raise ValueError('sigma_n must be positive')
    x = start.detach().clone()
    with torch.no_grad():
        for _ in range(num_iters):
            prior_grad, curvature = _prior_stats(x, sigma_x, p, q, T)
            g = (x-v) / sigma_n**2 + prior_grad
            numerator = g.square().sum()
            if numerator == 0:
                continue
            denominator = numerator / sigma_n**2
            for (left, right, _), c in zip(_pairs(), curvature):
                denominator += (c * (g[left]-g[right]).square()).sum()
            x -= (numerator / denominator) * g
    return x


def denoise_qggmrf(v, sigma_n, sigma_x, p, q, T, num_iters):
    """Return QGGMRF MAP denoising of float64 (H,W) v with noise SD sigma_n, prior scale sigma_x, p,q,T and num_iters steps."""
    return _prox_qggmrf(v, v, sigma_n, sigma_x, p, q, T, num_iters)


def _relative(a, b):
    """Return ||a-b||/||a|| for image tensors, defining zero over zero as zero."""
    numerator, denominator = torch.linalg.vector_norm(a-b), torch.linalg.vector_norm(a)
    return float(numerator / denominator) if denominator > 0 else (0. if numerator == 0 else float('inf'))


def pnp_admm(y, a_hat, denoiser, sigma_w, sigma_p, num_iters):
    """Run num_iters ADMM steps from v=y,u=0; return float64 image x and ||x-v||/||x|| history.

    y is (H,W) data, a_hat its complex blur DFT, denoiser maps one image to
    one image, sigma_w is measurement noise SD and sigma_p the F prox SD.
    """
    v, u, x = y.clone(), torch.zeros_like(y), y.clone()
    history = []
    with torch.no_grad():
        for _ in range(num_iters):
            x = prox_data(v-u, y, a_hat, sigma_w, sigma_p)
            v = denoiser(x+u)
            u = u + x-v
            history.append(_relative(x, v))
    return x, history


def pnp_qggmrf_map(y, a_hat, sigma_w, sigma_p, sigma_n, sigma_x, p, q, T, num_iters, inner_iters):
    """Return QGGMRF ADMM image and residual history, warm-starting each inner solve at previous v.

    y is float64 (H,W), a_hat complex blur DFT; sigma_w/p/n/x are data,
    forward-prox, denoiser, and prior SDs. p,q,T define the potential.
    num_iters is the outer count and inner_iters the denoiser step count.
    Equal sigma_p and sigma_n target the unweighted Lab 4 MAP objective.
    """
    v, u, x = y.clone(), torch.zeros_like(y), y.clone()
    history = []
    with torch.no_grad():
        for _ in range(num_iters):
            x = prox_data(v-u, y, a_hat, sigma_w, sigma_p)
            v = _prox_qggmrf(x+u, v, sigma_n, sigma_x, p, q, T, inner_iters)
            u = u + x-v
            history.append(_relative(x, v))
    return x, history


@lru_cache(maxsize=1)
def _drunet_model():
    """Return cached float32 pretrained grayscale DRUNet; LAB5_DNN_DEVICE optionally selects an accelerator."""
    import os
    import deepinv as dinv
    return dinv.models.DRUNet(in_channels=1, out_channels=1, pretrained='download').float().eval().to(os.environ.get('LAB5_DNN_DEVICE','cpu'))


def denoise_dnn(v, sigma_n):
    """Denoise (H,W) float64 v in [0,1] units at SD sigma_n; return float64 image using float32 DRUNet."""
    with torch.no_grad():
        model = _drunet_model()
        device = next(model.parameters()).device
        # Separate dtype conversion and transfer: combined MPS/float64 transfers
        # produce invalid values in the installed PyTorch backend.
        out = model(v.to(dtype=torch.float32).to(device)[None, None], float(sigma_n))
    return out[0, 0].to(v.device).to(dtype=torch.float64)


def pnp_prox_grad(y, a_hat, denoiser, sigma_w, sigma_p, num_iters):
    """Return PnP-PG image and ||x-x_prev||/||x|| history after num_iters steps from y.

    y is float64 (H,W), a_hat complex blur DFT, denoiser maps images to
    images; sigma_w is data noise SD and sigma_p**2 the gradient step size.
    """
    x, history = y.clone(), []
    with torch.no_grad():
        for _ in range(num_iters):
            residual_hat = a_hat * torch.fft.fft2(x) - torch.fft.fft2(y)
            grad = torch.fft.ifft2(a_hat.conj()*residual_hat).real / sigma_w**2
            new = denoiser(x-sigma_p**2*grad)
            history.append(_relative(new, x))
            x = new
    return x, history
