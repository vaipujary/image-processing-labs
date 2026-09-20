import numpy as np
import torch
import torch.nn.functional as F


def load_pgm(path):
    """Read a binary PGM image file.

    path     path to a P5 PGM file
    returns  (H, W) float64 tensor with values in [0, 1]
    """
    with open(path, "rb") as f:
        assert f.readline().strip() == b"P5"
        width, height = map(int, f.readline().split())
        assert int(f.readline()) == 255
        raw = np.frombuffer(f.read(), dtype=np.uint8).reshape(height, width)
    return torch.tensor(raw / 255.0, dtype=torch.float64)


def load_kernel(path):
    """Read a blur kernel from a text file.

    path     path to a kernel file, one row of the kernel per line
    returns  (Ka, Kb) float64 tensor, nonnegative and summing to 1
    """
    return torch.tensor(np.loadtxt(path), dtype=torch.float64)


def forward(x, a):
    """Apply the blur operator A of equation (2).

    x        (H, W) float64 tensor, the image
    a        (Ka, Kb) float64 tensor, the blur kernel, odd sized in both axes
    returns  (H, W) float64 tensor, A x
    """
    # conv2d is correlation, so flip the kernel to implement convolution.
    pad_h, pad_w = a.shape[0] // 2, a.shape[1] // 2
    padded = F.pad(x[None, None], (pad_w, pad_w, pad_h, pad_h), mode="circular")
    return F.conv2d(padded, a.flip((0, 1))[None, None])[0, 0]


def adjoint_autograd(v, a):
    """Return A^t v by differentiating <A x, v> with respect to x.

    v is an (H, W) float64 tensor; a is the odd-sized float64 blur kernel.
    Returns an (H, W) float64 tensor.
    """
    with torch.enable_grad():
        x = torch.zeros_like(v, requires_grad=True)
        return torch.autograd.grad((forward(x, a) * v).sum(), x)[0]


def adjoint(v, a):
    """Apply A^t by circular correlation with the unflipped kernel, Eq. (4).

    v is an (H, W) float64 tensor; a is the odd-sized float64 blur kernel.
    Returns an (H, W) float64 tensor.
    """
    pad_h, pad_w = a.shape[0] // 2, a.shape[1] // 2
    padded = F.pad(v[None, None], (pad_w, pad_w, pad_h, pad_h), mode="circular")
    return F.conv2d(padded, a[None, None])[0, 0]


def prediction_kernel(x):
    """Return the (3, 3) neighbor weights g on x's dtype and device.

    x is a floating-point tensor used only to select dtype and device.
    """
    return x.new_tensor([[1, 2, 1], [2, 0, 2], [1, 2, 1]]) / 12.0


def prediction_error(x):
    """Return e = x - g*x with zero padding for an (H, W) float64 image.

    The result has shape (H, W); use [1:-1, 1:-1] for the interior statistic.
    This raw prediction residual differs from Bx at the image boundary.
    """
    g = prediction_kernel(x)[None, None]
    return x - F.conv2d(x[None, None], g, padding=1)[0, 0]


def prior_cost(x, sigma_x):
    """Return the scalar float64 prior cost (1/2) x^t B x, Eq. (7).

    x is an (H, W) float64 tensor; sigma_x is a positive float.
    """
    return 0.5 * (x * prior_grad(x, sigma_x)).sum()


def prior_grad(x, sigma_x):
    """Return Bx, shape (H, W), using only neighbors inside the image.

    x is an (H, W) float64 tensor; sigma_x is a positive float.
    """
    if sigma_x <= 0:
        raise ValueError("sigma_x must be positive")
    g = prediction_kernel(x)[None, None]
    neighbors = F.conv2d(x[None, None], g, padding=1)[0, 0]
    # The mask removes nonexistent neighbors; do not penalize the boundary
    # against zero or connect opposite edges as in the measurement model.
    c = F.conv2d(torch.ones_like(x)[None, None], g, padding=1)[0, 0]
    return (c * x - neighbors) / sigma_x**2


def cost(x, y, a, sigma_w, sigma_x):
    """Return scalar float64 MAP cost, Eq. (8), without clipping.

    x, y: (H, W) float64 tensors; a: odd-sized float64 kernel.
    sigma_w and sigma_x: positive noise and prior standard deviations.
    """
    if sigma_w <= 0:
        raise ValueError("sigma_w must be positive")
    residual = forward(x, a) - y
    return residual.square().sum() / (2 * sigma_w**2) + prior_cost(x, sigma_x)


def cost_grad(x, y, a, sigma_w, sigma_x):
    """Return the (H, W) float64 MAP gradient A^t(Ax-y)/sigma_w^2 + Bx.

    x, y: (H, W) float64 tensors; a: odd-sized float64 kernel.
    sigma_w and sigma_x: positive noise and prior standard deviations.
    """
    if sigma_w <= 0:
        raise ValueError("sigma_w must be positive")
    return adjoint(forward(x, a) - y, a) / sigma_w**2 + prior_grad(x, sigma_x)


def gradient_descent(y, a, sigma_w, sigma_x, num_iters, alpha=None):
    """Minimize the MAP cost from x=y for exactly num_iters steps, Eq. (10).

    y: (H, W) float64 data; a: odd-sized float64 kernel; sigmas: positive.
    alpha defaults to Eq. (11). Returns (x_hat, cost_history), an (H, W)
    tensor and a list of num_iters+1 costs beginning with c(y).
    """
    x_hat, cost_history, _ = gradient_descent_diagnostics(
        y, a, sigma_w, sigma_x, num_iters, alpha
    )
    return x_hat, cost_history


def gradient_descent_diagnostics(y, a, sigma_w, sigma_x, num_iters, alpha=None):
    """Run the same descent and also return per-step relative image changes.

    Inputs match gradient_descent. Returns (x_hat, cost_history, nrmse_history),
    with list lengths num_iters+1 and num_iters; NRMSE starts at iteration 1.
    """
    if sigma_w <= 0 or sigma_x <= 0:
        raise ValueError("sigma_w and sigma_x must be positive")
    if not isinstance(num_iters, int) or num_iters < 0:
        raise ValueError("num_iters must be a nonnegative integer")
    if alpha is None:
        alpha = 1.0 / (1.0 / sigma_w**2 + 2.0 / sigma_x**2)
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    # Keep autograd out of the iterative reconstruction. The separate cost
    # and gradient functions remain differentiable for the required checks.
    with torch.no_grad():
        x = y.clone()
        residual = forward(x, a) - y
        bx = prior_grad(x, sigma_x)
        history = [float(residual.square().sum() / (2 * sigma_w**2)
                         + 0.5 * (x * bx).sum())]
        changes = []
        for _ in range(num_iters):
            grad = adjoint(residual, a) / sigma_w**2 + bx
            x_next = x - alpha * grad
            denominator = torch.linalg.vector_norm(x)
            numerator = torch.linalg.vector_norm(x_next - x)
            if denominator == 0:
                change = 0.0 if numerator == 0 else float("inf")
            else:
                change = float(numerator / denominator)
            changes.append(change)
            x = x_next
            # Reuse these arrays for the next gradient and the current cost.
            residual = forward(x, a) - y
            bx = prior_grad(x, sigma_x)
            history.append(float(residual.square().sum() / (2 * sigma_w**2)
                                 + 0.5 * (x * bx).sum()))
    return x, history, changes
