# Lab 3: Linear Forward Models and MAP Reconstruction

The completed numerical work follows the [live Lab 3 page](https://cabouman.github.io/grad_labs/labs-60141/lab3/index.html)
as accessed September 19, 2026: **50 iterations**, with
`sigma_x = 0.01, 0.05, 0.25` for both denoising and deconvolution.
The supplied textbook was checked against the implementation and discussion.

## Run the lab

From the repository root:

```bash
conda activate labs
python -m unittest discover -s lab3 -v
python lab3/run_experiments.py
```

The experiments also run from another working directory because input/output
paths are resolved relative to the script. On this machine the environment's
Python is `/opt/miniconda3/envs/labs/bin/python`. Select that interpreter in
your IDE if its default Python cannot import PyTorch.

`lab3.py` deliberately contains only imports and functions, as required by the
assignment. Running it directly produces no output; run `run_experiments.py`
to execute the lab. To rerun only the five specified checks:

```bash
python lab3/run_experiments.py --checks-only
```

The seed defaults to 0 and the CPU thread count to 4. Optional flags are
`--seed N` and `--threads N`. The saved report describes the default run.

## Step-by-step guide

1. **Measurement model.** The data are `y = forward(x,a) + sigma_w*noise`.
   Images are scaled by 255 into `[0,1]`. Noise is Gaussian and is never
   clipped. Every numerical image calculation uses PyTorch `float64`.
2. **Forward operator.** `forward` uses circular padding and `conv2d` with
   the kernel reversed in both axes. PyTorch's operation is correlation,
   which is why reversing the kernel is necessary. A 1x1 kernel containing
   1 gives the identity for denoising.
3. **Adjoint.** `adjoint` uses the same circular padding and the unflipped
   kernel. `adjoint_autograd` independently obtains the transpose by
   differentiating `<Ax,v>`. D1-D3 cover the operator images, three checks,
   and storage arithmetic. No full image matrix is constructed.
4. **Prior.** `prior_grad` implements `Bx=(c*x-g*x)/sigma_x**2`, with the
   prescribed eight-neighbor weights. Here `g*x` uses **zero padding** and
   `c=g*ones` counts existing neighbors, so the prior includes only pairs
   inside the image. `prior_cost` returns `0.5*sum(x*Bx)`.
   D4 measures the standard deviation of `x-g*x` over `[1:-1,1:-1]`.
5. **MAP cost and optimizer.** `cost` adds the Gaussian data term to the
   prior; `cost_grad` returns `A^t(Ax-y)/sigma_w**2 + Bx`.
   `gradient_descent` starts at `y`, uses the lab's step-size formula, and
   returns `(x_hat, cost_history)` with exactly `num_iters+1` costs.
   `gradient_descent_diagnostics` runs the same algorithm and also returns
   the NRMSE at each update without changing the required interface.
6. **Validate the code.** Required checks test the adjoint, inner-product
   identity, prior symmetry, and analytic gradient. The six unit tests
   additionally check the convolution direction, circular wraparound,
   a directly summed pairwise prior, corner behavior, constant-image
   behavior, and convergence to a stationary point. D5 records the actual
   AI request; D6 records measured checks; D7 gives the convexity proof.
7. **Denoise.** Use the identity kernel, `sigma_w=0.05`, and the three prior
   strengths. D8 contains all four requested deliverables.
8. **Deconvolve.** Use kernel 1 and `sigma_w=0.02`. D9 contains all four
   requested deliverables. All 50 steps are executed; the first NRMSE
   below 0.002 is recorded separately to honor both instructions on the
   lab page. Final image and RMSE comparisons use iteration 50.
9. **Interpret.** D10 answers all five questions using the actual results,
   includes the four-times-step-size experiment, and refers to Chapter 6
   for the edge-preserving alternative to the Gaussian prior.

## Measured results

Seed 0, CPU, 4 threads, PyTorch 2.14.0, Python 3.14.7:

| Image | Denoising RMSE | Deconvolution RMSE |
| --- | ---: | ---: |
| Input `y` | 0.04991285 | 0.04753759 |
| `sigma_x=0.01` | 0.03983552 | 0.04246243 |
| `sigma_x=0.05` | **0.02971294** | **0.03599792** |
| `sigma_x=0.25` | 0.04801993 | 0.06838509 |

The interior prediction-error standard deviation is `0.0205966785`
(population convention). At `sigma_x=0.05`, NRMSE first falls below 0.2%
at iteration **4** for denoising and **12** for deconvolution. Measured
50-step times were approximately 0.543 s and 17.997 s, respectively;
timings and last-bit floating-point results can vary between machines.

The largest adjoint disagreement was `9.992e-16`; the largest gradient
disagreement was `2.558e-13`. All required checks and six unit tests passed.

## Files and outputs

- `lab3.py`: the functions-only file requested for submission.
- `run_experiments.py`: the reproducible experiment driver.
- `test_lab3.py`: small independent correctness tests.
- `AI_SPECIFICATION.md`: the actual user prompt and the assistant's
  separately labeled implementation specification.
- `results/summary.json`: full-precision results, all cost/NRMSE histories,
  timings, input hashes, and environment details.
- `results/*.png`: the seven required image/plot figures.
- `results/*_arrays.pt`: unmodified float64 inputs and reconstructions,
  regenerated by the driver and excluded from Git. Load with
  `torch.load(path, weights_only=True)`.
- `output/pdf/Lab_3_Report.pdf`: the eight-page D1-D10 report.
- `make_report.py`: rebuilds the report from the saved results.
- `output/docx/Lab_3_Report.docx`: the editable Word report with native
  text, tables, and display equations; figures are replaceable images.
- `make_word_report.py`: builds the Word version from the same structured
  report content, so measurements and explanations stay consistent.
- `verify_results.py`: independently recomputes the results using FFT blur
  and directly evaluated pairwise prior formulas.
- `results/verification.json`: the independent verification results.

To rebuild the report, install its optional dependencies into the active
environment if needed, then run:

```bash
python -m pip install reportlab pillow
python lab3/make_report.py
```

The report generator uses Arial on macOS or DejaVu Sans on Linux. `--name`
and `--repo` allow changing the report metadata. The current values were
read from this repository's Git configuration. The provided PDF has been
rendered and visually checked; rerunning experiments requires rebuilding
the report to incorporate new measurements.

To verify the saved results and rebuild the Word version:

```bash
python lab3/verify_results.py
python -m pip install python-docx reportlab pillow lxml
python lab3/make_word_report.py
```

The provided Word version was checked against the supplied 16-page printout
of the live webpage. All six reconstructions agree with an independent
implementation to within `1.34e-15` per pixel. RMSEs, full cost histories,
NRMSE histories and threshold crossings, memory arithmetic, and the
prediction-error standard deviation were confirmed. The Word document has
also been rendered and visually inspected.

## Textbook consistency

- Chapter 4, §§4.1-4.4 and §4.6, explains noncausal prediction and the
  precision matrix. Do not replace this prior with a squared prediction
  residual: that would give a different gradient involving another filter.
- Chapter 5, Eqs. (5.11)-(5.12), p. 67, gives this MAP cost and gradient.
  The lab's `alpha` corresponds to the book's unscaled `beta`. The book's
  later `alpha` in Eq. (5.13) uses a differently scaled update direction.
- The difference-only prior in this lab annihilates constant images and
  is positive semidefinite. Its Gaussian prior is intrinsic (improper in
  the constant-image direction), but its denoising posterior is proper and
  strictly convex because of the data term.
- Chapter 5, §§5.3.1-5.3.2, relates convergence to Hessian eigenvalues.
  Chapter 6, §§6.2-6.3, explains how nonquadratic potentials preserve edges.

## Before submitting

Review the answers and figures. **D7 explicitly asks for a handwritten
proof**; the report contains the typed derivation for you to transcribe
and replace with your handwritten version. The report identifies the AI
prompt honestly and does not claim the student wrote a more detailed one.

The assignment requests `Lab_3_Report.pdf` and the plain `lab3.py` file as
two separate Brightspace uploads. It also asks you to commit and push the
code. This work has not been committed, pushed, or submitted. Review the
existing staged changes before committing; unrelated user files were not
changed. Do not include the textbook PDF in the commit.
