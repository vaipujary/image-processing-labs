"""Run Lab 3 in order and save all numerical results and required figures.

Usage from the repository root: python lab3/run_experiments.py
No calculations run when this module is imported.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import lab3


def run_checks(a):
    """Run D2 and D6 with seed 0, returning the actual measured errors."""
    torch.manual_seed(0)
    u = torch.randn(512, 768, dtype=torch.float64)
    v = torch.randn(512, 768, dtype=torch.float64)
    au, atv = lab3.forward(u, a), lab3.adjoint(v, a)
    auto_atv = lab3.adjoint_autograd(v, a)
    lhs, rhs = (au * v).sum(), (u * atv).sum()
    sigma_x, sigma_w = 0.05, 0.05
    b_lhs = (u * lab3.prior_grad(v, sigma_x)).sum()
    b_rhs = (v * lab3.prior_grad(u, sigma_x)).sum()
    z = torch.randn(32, 32, dtype=torch.float64, requires_grad=True)
    y_small = torch.randn(32, 32, dtype=torch.float64)
    auto = torch.autograd.grad(lab3.cost(z, y_small, a, sigma_w, sigma_x), z)[0]
    manual = lab3.cost_grad(z.detach(), y_small, a, sigma_w, sigma_x)
    checks = {
        "check1_adjoint_max_abs": float((atv - auto_atv).abs().max()),
        "check2_transpose_relative": float((lhs - rhs).abs() / lhs.abs()),
        "check3_forward_adjoint_max_abs": float((au - lab3.adjoint(u, a)).abs().max()),
        "forward_max_abs": float(au.abs().max()),
        "transpose_lhs": float(lhs), "transpose_rhs": float(rhs),
        "check4_prior_symmetry_abs": float((b_lhs - b_rhs).abs()),
        "prior_symmetry_relative": float((b_lhs - b_rhs).abs() / torch.maximum(b_lhs.abs(), b_rhs.abs())),
        "check5_gradient_max_abs": float((auto - manual).abs().max()),
        "gradient_max_abs": float(auto.abs().max()),
        "sigma_x": sigma_x, "sigma_w": sigma_w,
    }
    torch.testing.assert_close(atv, auto_atv, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(lhs, rhs, atol=1e-10, rtol=1e-12)
    torch.testing.assert_close(b_lhs, b_rhs, atol=1e-8, rtol=1e-12)
    torch.testing.assert_close(auto, manual, atol=1e-10, rtol=1e-12)
    assert checks["check3_forward_adjoint_max_abs"] > 0.1
    print("D2 / D6: required checks passed", json.dumps(checks, indent=2), flush=True)
    return checks


def save_images(path, images, titles):
    """Save a labeled grid, clipping for display only with fixed [0,1] scaling."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.1), layout="constrained")
    for axis, image, title in zip(axes.flat, images, titles):
        axis.imshow(image.detach().cpu().clamp(0, 1).numpy(), cmap="gray", vmin=0, vmax=1)
        axis.set_title(title, fontsize=12)
        axis.axis("off")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_convergence(path, history, changes, title):
    """Save cost and relative image change versus iteration, including the threshold."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.1), layout="constrained")
    axes[0].plot(range(len(history)), history, color="#176d8d")
    axes[0].set(xlabel="Iteration n", ylabel="MAP cost", title="(a) Cost")
    # Zero changes cannot be plotted on a log axis; do not invent a floor.
    values = [value if value > 0 else float("nan") for value in changes]
    axes[1].semilogy(range(1, len(changes) + 1), values, color="#176d8d")
    axes[1].axhline(0.002, color="#ad4f29", linestyle="--", label="0.2% threshold")
    axes[1].set(xlabel="Iteration n", ylabel="NRMSE (fraction)", title="(b) Relative image change")
    axes[1].legend(fontsize=9)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def rmse(estimate, reference):
    """Return float RMSE on unmodified float64 arrays, before display clipping."""
    return float((estimate - reference).square().mean().sqrt())


def run_restorations(x, a, sigma_w, name, output, seed):
    """Run the three specified 50-step experiments on one fixed noise realization."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    y = lab3.forward(x, a) + sigma_w * torch.randn(x.shape, dtype=torch.float64, generator=generator)
    result = {"sigma_w": sigma_w, "num_iters": 50, "seed": seed,
              "input_rmse": rmse(y, x), "input_min": float(y.min()),
              "input_max": float(y.max()), "runs": []}
    images, titles = [y], [f"{'Noisy' if name == 'denoising' else 'Blurred noisy'} input y"]
    arrays = {"y": y}
    for sigma_x in (0.01, 0.05, 0.25):
        alpha = 1 / (1 / sigma_w**2 + 2 / sigma_x**2)
        start = time.perf_counter()
        x_hat, history, changes = lab3.gradient_descent_diagnostics(y, a, sigma_w, sigma_x, 50)
        seconds = time.perf_counter() - start
        deltas = torch.diff(torch.tensor(history, dtype=torch.float64))
        # Round-off at a stationary point is not mathematical divergence.
        tolerance = 1e-12 * max(1, abs(history[0]))
        crossing = next((n for n, value in enumerate(changes, 1) if value < 0.002), None)
        row = {"sigma_x": sigma_x, "alpha": alpha, "rmse": rmse(x_hat, x),
               "final_cost": history[-1], "seconds": seconds,
               "first_nrmse_below_0_002": crossing,
               "strictly_decreasing": bool((deltas < 0).all()),
               "nonincreasing_with_roundoff": bool((deltas <= tolerance).all()),
               "max_cost_increase": max(0, float(deltas.max())),
               "cost_history": history, "nrmse_history": changes}
        assert row["nonincreasing_with_roundoff"], f"Cost increased: {name}, sigma_x={sigma_x}"
        assert torch.isfinite(x_hat).all()
        torch.testing.assert_close(torch.tensor(history[-1], dtype=torch.float64),
                                   lab3.cost(x_hat, y, a, sigma_w, sigma_x))
        result["runs"].append(row)
        images.append(x_hat)
        titles.append(rf"$\sigma_x={sigma_x}$; RMSE = {row['rmse']:.5f}")
        arrays[f"x_hat_{sigma_x}"] = x_hat
        print(f"{name}: sigma_x={sigma_x}, RMSE={row['rmse']:.8f}, "
              f"threshold n={crossing}, time={seconds:.2f}s", flush=True)
        if sigma_x == 0.05:
            save_convergence(output / f"{name}_convergence.png", history, changes,
                             rf"{name.capitalize()}: $\sigma_x=0.05$, $\sigma_w={sigma_w}$")
    result["best_sigma_x"] = min(result["runs"], key=lambda row: row["rmse"])["sigma_x"]
    save_images(output / f"{name}_images.png", images, titles)
    torch.save(arrays, output / f"{name}_arrays.pt")
    return result, y


def main():
    """Execute checks first, then D1-D4, D8-D9, and the D10 stability experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checks-only", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    base = Path(__file__).resolve().parent
    output = base / "results"
    output.mkdir(exist_ok=True)
    x = lab3.load_pgm(base / "kodim23.pgm")
    a = lab3.load_kernel(base / "levin09_kernels/levin09_kernel1.txt")
    assert x.shape == (512, 768) and x.dtype == torch.float64
    assert a.shape == (19, 19) and bool((a >= 0).all())
    torch.testing.assert_close(a.sum(), a.new_tensor(1.0), atol=1e-14, rtol=0)
    checks = run_checks(a)
    if args.checks_only:
        return

    ax, atx = lab3.forward(x, a), lab3.adjoint(x, a)
    save_images(output / "operators.png", [x, ax, atx, ax - atx + 0.5],
                ["Clean image x", "Forward blur Ax", r"Adjoint blur $A^t x$",
                 r"Difference $Ax-A^t x$ (offset by 0.5)"])
    e = lab3.prediction_error(x)
    fig, axis = plt.subplots(figsize=(8, 5.5), layout="constrained")
    axis.imshow((e + 0.5).clamp(0, 1).numpy(), cmap="gray", vmin=0, vmax=1)
    axis.set_title("Noncausal prediction error e (offset by 0.5)")
    axis.axis("off")
    fig.savefig(output / "prediction_error.png", dpi=180)
    plt.close(fig)
    n, nnz = x.numel(), int(torch.count_nonzero(a))
    summary = {
        "source_url": "https://cabouman.github.io/grad_labs/labs-60141/lab3/index.html",
        "seed": args.seed, "torch_version": str(torch.__version__),
        "python_version": platform.python_version(), "platform": platform.platform(),
        "device": "cpu", "threads": torch.get_num_threads(), "dtype": "torch.float64",
        "image_shape": list(x.shape), "kernel_shape": list(a.shape),
        "kernel_sum": float(a.sum()), "kernel_nonzeros": nnz,
        "image_sha256": hashlib.sha256((base / "kodim23.pgm").read_bytes()).hexdigest(),
        "kernel_sha256": hashlib.sha256((base / "levin09_kernels/levin09_kernel1.txt").read_bytes()).hexdigest(),
        "checks": checks,
        "memory": {"pixels": n, "dense_bytes": n * n * 8,
                   "sparse_bytes": n * nnz * (8 + 4), "kernel_bytes": a.numel() * 8},
        "prediction_error_std": float(e[1:-1, 1:-1].std(correction=0)),
        "prediction_error_std_sample": float(e[1:-1, 1:-1].std(correction=1)),
        "prediction_std_convention": "population standard deviation, correction=0, interior [1:-1,1:-1]",
    }
    print("D1 / D3 / D4:", json.dumps({k: summary[k] for k in
          ("kernel_shape", "kernel_sum", "kernel_nonzeros", "memory", "prediction_error_std")}, indent=2), flush=True)
    identity = torch.ones((1, 1), dtype=torch.float64)
    summary["denoising"], y_denoise = run_restorations(x, identity, 0.05, "denoising", output, args.seed)
    summary["deconvolution"], _ = run_restorations(x, a, 0.02, "deconvolution", output, args.seed)

    stable_alpha = 1 / (1 / 0.05**2 + 2 / 0.05**2)
    _, unstable_cost = lab3.gradient_descent(y_denoise, identity, 0.05, 0.05, 50, 4 * stable_alpha)
    summary["unstable"] = {"sigma_x": 0.05, "sigma_w": 0.05, "alpha": 4 * stable_alpha,
                           "cost_history": unstable_cost, "growth_factor": unstable_cost[-1] / unstable_cost[0]}
    stable = next(row for row in summary["denoising"]["runs"] if row["sigma_x"] == 0.05)
    fig, axis = plt.subplots(figsize=(8, 3.5), layout="constrained")
    axis.semilogy(range(51), unstable_cost, color="#ad4f29", label=r"$4\alpha$: unstable")
    axis.semilogy(range(51), stable["cost_history"], color="#176d8d", label=r"$\alpha$: prescribed")
    axis.set(xlabel="Iteration n", ylabel="MAP cost (log scale)", title=r"Denoising stability: $\sigma_x=\sigma_w=0.05$")
    axis.legend()
    axis.grid(alpha=0.2)
    fig.savefig(output / "unstable_step.png", dpi=180)
    plt.close(fig)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"All experiments complete. Results: {output}", flush=True)


if __name__ == "__main__":
    main()
