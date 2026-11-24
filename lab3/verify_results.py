"""Audit saved lab results with independent FFT blur and pairwise prior formulas.

Run with the labs environment. Does not overwrite the original experiment data.
"""

import hashlib
import json
from pathlib import Path

import torch

import lab3
from run_experiments import run_checks


def pair_prior(x, sigma_x):
    """Return energy and gradient by counting the four unique neighbor directions."""
    grad = torch.zeros_like(x)
    energy = x.new_zeros(())
    pairs = [
        ((slice(None), slice(1, None)), (slice(None), slice(None, -1)), 1 / 6),
        ((slice(1, None), slice(None)), (slice(None, -1), slice(None)), 1 / 6),
        ((slice(1, None), slice(1, None)), (slice(None, -1), slice(None, -1)), 1 / 12),
        ((slice(1, None), slice(None, -1)), (slice(None, -1), slice(1, None)), 1 / 12),
    ]
    for first, second, weight in pairs:  # Four neighbor types, never pixels.
        difference = x[first] - x[second]
        energy += weight * difference.square().sum() / (2 * sigma_x**2)
        derivative = weight * difference / sigma_x**2
        grad[first] += derivative
        grad[second] -= derivative
    return energy, grad


def independent_descent(y, a, sigma_w, sigma_x, iterations, alpha=None):
    """Compute the same iterates using FFT convolution and a pairwise prior."""
    psf = torch.zeros_like(y)
    psf[:a.shape[0], :a.shape[1]] = a
    psf = psf.roll((-(a.shape[0] // 2), -(a.shape[1] // 2)), (0, 1))
    transfer = torch.fft.rfft2(psf)

    def blur(x):
        return torch.fft.irfft2(torch.fft.rfft2(x) * transfer, s=x.shape)

    def transpose(x):
        return torch.fft.irfft2(torch.fft.rfft2(x) * transfer.conj(), s=x.shape)

    if alpha is None:
        alpha = 1 / (1 / sigma_w**2 + 2 / sigma_x**2)
    x = y.clone()
    costs, changes = [], []
    for n in range(iterations + 1):
        residual = blur(x) - y
        prior, gradient = pair_prior(x, sigma_x)
        costs.append(float(residual.square().sum() / (2 * sigma_w**2) + prior))
        if n == iterations:
            break
        updated = x - alpha * (transpose(residual) / sigma_w**2 + gradient)
        changes.append(float(torch.linalg.vector_norm(updated - x) / torch.linalg.vector_norm(x)))
        x = updated
    return x, costs, changes


def main():
    """Verify input hashes, required checks, memory, statistics, and every saved run."""
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    base = Path(__file__).resolve().parent
    stored = json.loads((base / "results/summary.json").read_text())
    x = lab3.load_pgm(base / "kodim23.pgm")
    a = lab3.load_kernel(base / "levin09_kernels/levin09_kernel1.txt")
    assert hashlib.sha256((base / "kodim23.pgm").read_bytes()).hexdigest() == stored["image_sha256"]
    assert hashlib.sha256((base / "levin09_kernels/levin09_kernel1.txt").read_bytes()).hexdigest() == stored["kernel_sha256"]
    checks = run_checks(a)
    for key in checks:
        assert abs(checks[key] - stored["checks"][key]) <= 1e-10, key
    n, nonzeros = x.numel(), int(torch.count_nonzero(a))
    assert stored["memory"] == {"pixels": n, "dense_bytes": n*n*8,
        "sparse_bytes": n*nonzeros*12, "kernel_bytes": a.numel()*8}

    # Direct neighbor average on the interior, independently of conv2d.
    middle = x[1:-1, 1:-1]
    neighbors = (x[:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, :-2] + x[1:-1, 2:]) / 6
    neighbors += (x[:-2, :-2] + x[:-2, 2:] + x[2:, :-2] + x[2:, 2:]) / 12
    prediction_std = float((middle - neighbors).std(correction=0))
    assert abs(prediction_std - stored["prediction_error_std"]) < 1e-14
    audit = {"required_checks": "passed", "input_hashes": "match", "memory": "match",
             "prediction_error_std": prediction_std, "runs": []}

    for name, kernel in (("denoising", torch.ones(1, 1, dtype=torch.float64)), ("deconvolution", a)):
        experiment = stored[name]
        arrays = torch.load(base / f"results/{name}_arrays.pt", weights_only=True)
        generator = torch.Generator().manual_seed(stored["seed"])
        y = lab3.forward(x, kernel) + experiment["sigma_w"] * torch.randn(x.shape, generator=generator, dtype=torch.float64)
        torch.testing.assert_close(y, arrays["y"], rtol=0, atol=0)
        assert abs(float((y-x).square().mean().sqrt()) - experiment["input_rmse"]) < 1e-14
        for run in experiment["runs"]:
            sigma_x = run["sigma_x"]
            result, costs, changes = independent_descent(y, kernel, experiment["sigma_w"], sigma_x, 50)
            expected = arrays[f"x_hat_{sigma_x}"]
            max_error = float((result - expected).abs().max())
            torch.testing.assert_close(result, expected, rtol=0, atol=2e-12)
            torch.testing.assert_close(torch.tensor(costs, dtype=torch.float64),
                torch.tensor(run["cost_history"], dtype=torch.float64), rtol=1e-12, atol=1e-8)
            torch.testing.assert_close(torch.tensor(changes, dtype=torch.float64),
                torch.tensor(run["nrmse_history"], dtype=torch.float64), rtol=0, atol=2e-14)
            error = float((result-x).square().mean().sqrt())
            assert abs(error - run["rmse"]) < 1e-13
            alpha = 1 / (1 / experiment["sigma_w"]**2 + 2 / sigma_x**2)
            assert abs(alpha - run["alpha"]) < 1e-16
            crossing = next((n for n, value in enumerate(changes, 1) if value < 0.002), None)
            assert crossing == run["first_nrmse_below_0_002"]
            row = {"experiment": name, "sigma_x": sigma_x, "max_pixel_error": max_error,
                   "rmse": error, "threshold_iteration": crossing}
            audit["runs"].append(row)
            print(json.dumps(row), flush=True)
    y = torch.load(base / "results/denoising_arrays.pt", weights_only=True)["y"]
    unstable = stored["unstable"]
    _, costs, _ = independent_descent(y, torch.ones(1, 1, dtype=torch.float64),
        0.05, 0.05, 50, unstable["alpha"])
    torch.testing.assert_close(torch.tensor(costs, dtype=torch.float64),
        torch.tensor(unstable["cost_history"], dtype=torch.float64), rtol=2e-12, atol=1e-8)
    audit["unstable_cost_history"] = "matches independent calculation"
    (base / "results/verification.json").write_text(json.dumps(audit, indent=2) + "\n")
    print("PASS: all saved results match the independent implementation.", flush=True)


if __name__ == "__main__":
    main()
