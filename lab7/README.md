# Lab 7 Stochastic Sampling

Assignment: https://cabouman.github.io/grad_labs/labs-60141/lab7/index.html

The editable report is **`output/Lab_7_Report.docx`**. It contains D1–D9
in the requested order, the author's name, repository link, and all random
seeds. Text, headings, numerical results, and the equation remain editable in
Word. Figures are embedded PNGs; their editable Python source is
`make_figures.py`. No report PDF is created. The assignment's PDF submission
format is intentionally superseded by the user's explicit Word-only request.

## Files and reproduction

The required submission code is `lab7.py`, a standalone file containing only
function definitions, including all six exact required signatures. PyTorch
imports occur inside the functions. Samplers use float64 and four parity
classes; loops traverse sweeps, colors, and neighbor offsets, never pixels.

From the repository root, using the existing `labs` environment:

```bash
python lab7/run_experiments.py
python lab7/verify_results.py
python lab7/make_figures.py
```

This uses the existing `lab3/kodim23.pgm` and imports the existing Lab 3 MAP
implementation from `lab3/lab3.py`. The image is divided by 255 and cropped
at zero-based rows `[128:384)` and columns `[256:512)`. Its source SHA-256
is recorded in `results/summary.json`. No observation or sample is clipped.

Build the report with Python and `python-docx`:

```bash
python lab7/make_word_report.py
python lab7/audit_report.py
```

The authoring run used the bundled document Python at
`/Users/vaidehipujary/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python`.
The numerical experiments used `/opt/miniconda3/envs/labs/bin/python`.
Rebuilding overwrites the Word document, so save manual revisions under a
different filename before regenerating it. At the user's request, the report
is checked through its files only, without screenshots, GUI layout review,
or PDF conversion. Exact page rendering has not been visually inspected.

## Required work

| Item | Implementation and result |
| --- | --- |
| D1 | Three zero-start 256 × 256 GMRF chains, 200 sweeps, scales 0.01, 0.05, 0.2; individual centered display ranges |
| D2 | 500-sweep mean and centered standard deviation traces at scale 0.05; intrinsic brightness drift explained |
| D3 | Three 300-sweep zero-start QGGMRF chains, scales 0.002, 0.02, 0.2; acceptance shown under each image |
| D4 | Energy after every sweep for all proposal widths; measured tuning comparison |
| D5 | Selected QGGMRF sample beside the D1 GMRF sample at prior scale 0.05; common display range |
| D6 | Required clean, noisy, single sample, posterior mean and MAP images over [0,1]; last three RMSEs |
| D7 | Pixelwise posterior standard deviation from 400 kept samples; boundary interpretation and display range |
| D8 | Two additional 500-sweep chains initialized at y and zero; two-start energy plot and burn-in assessment |
| D9 | All five discussion answers, including the full Hastings ratio for Gibbs |

The initial 100 sweeps are discarded when accumulating posterior moments.
The single sample in D6 is the final sample from the additional y-start chain
in D8. Variance is accumulated with Welford's algorithm and divisor 400.
Four colors are updated in order (even, even), (even, odd), (odd, even),
(odd, odd). Missing neighbors are excluded rather than treated as observations
of zero. All prior images are centered for display only.

Seeds are 7 for every prior run and for observation noise, 8 for posterior
moment estimation, 9 for the additional y-start chain, and 10 for the
zero-start chain. Resetting seed 7 for D1 holds the normal draws fixed across
prior scales, so self-scaled GMRF images have identical textures. PyTorch
2.14.0, Python 3.14.7, CPU, and one thread were used.

## Results and interpretation

- RMSE: single posterior sample 0.047166713, posterior mean 0.030259825,
  MAP 0.030202773. The exact Gaussian mean equals MAP; the sample mean
  differs from MAP by RMSE 0.001911500 due to finite correlated sampling.
- Width 0.2 was best among the tested proposals, with overall acceptance
  40.53%; widths 0.002 and 0.02 accepted 99.28% and 90.97%. Mean squared
  movement and energy correlation support the selection. A high acceptance
  percentage alone does not establish efficient exploration.
- The D8 chains reach the same energy band in about 4–5 sweeps. The required
  100-sweep burn-in is conservative for the measured statistics.
- Boundary uncertainty is higher because there are fewer neighbors.
  The covariance of this fixed Gaussian posterior does not depend on y;
  uncertainty should not be interpreted as a detector of image edges.
- The unconstrained difference prior is improper in its constant mode.
  Its brightness cannot converge to a stationary marginal. Centered variation
  starts to stabilize, but slow spatial modes are not proven mixed at 500 sweeps.
- D9 question 1's suggested RMSE ordering is not observed. D9 question 3's
  symmetric Metropolis formula requires the general Hastings proposal ratio
  for Gibbs. The report explicitly explains both points rather than changing
  the measured data or using an incorrect acceptance proof.

`results/verification.json` records independent checks, including exact local
replacement energies, additive-shift invariance, no input mutation,
reproducibility, explicit moment accumulation, full-scale MAP residuals, and
comparison of a small sampled Gaussian posterior with its exact covariance.
`results/images.pt` stores all image tensors; `histories.json` stores all
required sweep histories; `summary.json` stores the numerical results.

The report is an AI-assisted editable draft. Brightspace submission remains
with the student so that the report can be reviewed and updated first.
Source files and compact result records are versioned; the Word report,
image tensors, plots, and full histories remain local and can be regenerated.
