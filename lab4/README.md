# Lab 4: MAP Reconstruction with Non-Gaussian Priors

Completed against the [assigned webpage](https://cabouman.github.io/grad_labs/labs-60141/lab4/index.html)
as accessed October 1, 2026. The local textbook was checked for the QGGMRF
potential and symmetric-bound surrogate (Chapters 6 and 8).

## Editable report and submission files

- `output/docx/Lab_4_Report.docx`: editable Word report, D1–D11 in order.
  Text, tables, and display equations are native Word objects. Figures are
  replaceable PNG images, and their source arrays and scripts are included.
- `output/pdf/Lab_4_Report.pdf`: the matching PDF requested by the webpage.
- `lab4.py`: the required functions-only code file. It reuses the four
  permitted functions from `../lab3/lab3.py` rather than copying them.

The webpage requests separate PDF and plain `.py` uploads to Brightspace,
not a zip. Review the report before uploading. Step 4a calls the D2 work a
paper-and-pencil derivation; the report supplies the complete editable typed
derivation. Transcribe it if your instructor requires a handwritten version.
No Brightspace submission is performed by the scripts.

## Run or reproduce

From the repository root, using the existing labs environment:

```bash
conda activate labs
python lab4/test_lab4.py
python lab4/run_experiments.py
python lab4/make_figures.py
python lab4/verify_results.py
```

On this machine the numerical interpreter is
`/opt/miniconda3/envs/labs/bin/python`. Each script also resolves its input
paths correctly when launched from a different working directory.
The data are the existing `lab3/kodim23.pgm` and
`lab3/levin09_kernels/levin09_kernel1.txt`.

The experiment driver defaults to `--seed 0 --threads 4`. It performs all
50 iterations for each of R1–R6; it does not terminate early. Only display
rendering saturates values outside [0,1]. The observation, reconstructions,
and cost/RMSE calculations are not clipped.

To rebuild the Word report after regenerating results and figures:

```bash
python lab4/make_word_report.py
```

This requires `python-docx`; the existing Codex document runtime provides
it. Its Python on this machine is
`/Users/vaidehipujary/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python`.
Rebuilding overwrites the generated report, so save manual Word edits in a
separate copy if you also want to regenerate it from Python. Export the
edited Word file to PDF before submitting an updated version.

The delivered document was rendered with the documents skill's
`render_docx.py` using the bundled LibreOffice, and every page was visually
checked. Rendering intermediates are held outside the project.

## Implementation and checks

- `rho` uses the equivalent stable expression
  `abs(delta)**p / (p*sigma_x**p*(1+u))`, with
  `u=(abs(delta)/(T*sigma_x))**(p-q)`. Only the power argument is floored
  at `1e-8`. `rho_prime` differentiates this implemented function, including
  the tiny floored region. The numerator keeps the true difference.
- The eight planes are NW, N, NE, W, E, SW, S, SE. Invalid pairs are masked
  after the potential, including at image corners. Loops run over eight
  offsets or iterations, never individual pixels.
- The true cost uses half of the directed neighbor sum. The surrogate
  weight already includes its factor of one half; the directional Hessian
  uses the full directed sum with no extra half. The threshold weights use
  `b/(2*sigma_x**2)` at and below `1e-8` as required.
- `true_gradient` uses PyTorch autograd. The analytic Eq. (16) appears only
  in validation. The reconstruction uses one surrogate step per iteration,
  with no line search or inner loop.
- Tests cover the potential, influence and zero differences; independent
  unordered-pair cost; neighbor signs and boundary masks; gradient;
  surrogate Hessian; scalar upper bounds; monotonicity; and zero-gradient
  stationary inputs.
- `verify_results.py` independently recomputes costs using FFT convolution
  and four unordered-pair slice sums. Its potential does not use the floor,
  so this also checks that the numerical threshold has negligible effect.
- The T=100 runs use the prescribed QGGMRF potential, which approximates
  the Gaussian regime. D1 separately uses the exact quadratic comparison.
  The book and lab reverse the names of p and q; the implementation follows
  the lab equations and normalization.

## Measured results

Seed 0; sigma_w=0.02; p=2; q=1.2; omega=1; 50 updates; CPU with 4 threads:

| Run | Prior | T | sigma_x | RMSE | Final true cost | Seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| R1 | Gaussian comparison | 100 | 0.036 | 0.03497541 | 200096.981 | 44.66 |
| R2 | QGGMRF | 1 | 0.020 | **0.02990709** | 204148.294 | 45.69 |
| R3 | QGGMRF | 1 | 0.010 | 0.03399855 | 248489.096 | 45.31 |
| R4 | QGGMRF | 1 | 0.040 | 0.03801039 | 159746.717 | 45.23 |
| R5 | Gaussian comparison | 100 | 0.018 | 0.03862095 | 253843.513 | 44.91 |
| R6 | Gaussian comparison | 100 | 0.072 | 0.04064256 | 150423.363 | 44.68 |

All 300 cost changes are negative. The smallest R2 decrease is 0.703300.
The maximum gradient-check discrepancy is 2.27e-12, and the maximum
independent final-cost error is 2.91e-11. All RMSEs agree. The noisy
observation is bit-for-bit identical to the saved Lab 3 deconvolution data,
with input RMSE 0.04753759. Timings can vary with machine load.

The shared crop is `[160:288, 384:512]` (zero-based, half-open), and the
edge profile uses row 232. D9 uses a stated display range of [0,1250] for
the weights recomputed at R2 iteration 50.

## Output map

- `results/summary.json`: parameters, hashes, environment, full histories,
  RMSEs, timings and figure coordinates.
- `results/verification.json`: independent numerical verification.
- `results/D*.png`: eight required figure panels.
- `results/reconstructions.pt`: float64 ground truth, observation, kernel,
  all six estimates, and final R2 weight sum; load with
  `torch.load(path, weights_only=True)`. Regenerated and excluded from Git.
- `AI_SPECIFICATION.md`: actual user request, the assistant's separately
  identified specification, and the first-run import correction.
- `references/lab4_instructions.html`: local snapshot of the assigned page,
  excluded from Git. The textbook and unrelated files are not part of Lab 4.

The report and large numerical tensors remain local; source code, figure
PNGs, and compact measurement records can be committed under `lab4/`.
