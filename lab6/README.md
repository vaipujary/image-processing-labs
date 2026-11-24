# Lab 6 The EM Algorithm

Source: https://cabouman.github.io/grad_labs/labs-60141/lab6/index.html

The editable report is `output/Lab_6_Report.docx`, with D1–D13 in order. Text,
equations, matrices, and tables remain native Word objects. Plots are embedded
images with editable Python source in `make_figures.py`. No PDF is delivered.

## Reproduce

Use the existing `labs` conda environment, with NumPy, SciPy, and Matplotlib:

```bash
python lab6/generate_data.py
python lab6/run_experiments.py
python lab6/verify_results.py
python lab6/make_figures.py
```

Data generation reuses `results/data.npz` if it already exists. The label draw
and Gaussian draw are interleaved for each of 500 observations using
`np.random.default_rng(0)`. Random starts use seeds 0 through 4 exactly as in
the assignment. No data are regenerated between fits.

The required seven functions are in `lab6.py`. `run_experiments.py` records the
four required numerical checks, all fits, and the complete merge search.
`verify_results.py` independently verifies fits using explicit 2D Gaussian
densities and raw second moments, and verifies all chosen merge pairs.

Build the report with the bundled document Python:

```bash
/Users/vaidehipujary/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python lab6/make_word_report.py
```

Rebuilding overwrites the Word file, so preserve manual edits in another copy.
Render and visually inspect the document after changing it. Preview PNGs are
kept outside the repository in `/private/tmp/lab6-docx-preview-final`.

## Results and interpretation

- Saved class counts: 187, 198, 115.
- Baseline cost: 1906.272853961; label disagreement: 305/500, or 61%.
- Best random start: seed 3, cost 1858.242279342.
- Merge-search cost at M = 3: 1858.242261588.
- Selected order: 3; MDL: 1916.958181460.
- Largest signed likelihood decrease: -2.274e-13, a rounding-level value.

The prescribed first-three-samples initialization splits the negative cluster
and merges the two positive clusters. This differs from the page's qualitative
description of a reasonable baseline; independent calculations confirm the
reported outcome under the explicit sample-by-sample random draw order.

The specifications were recorded in `SPECIFICATIONS.json` before coding and
reproduced verbatim in the report. The pre-run prediction and specification
clarifications are recorded in `WORK_LOG.md`. AI-authored material is identified
in the report. D2 must be replaced with the student's independently completed
derivations to meet the assignment's unaided-work requirement.

The assignment eventually requests a PDF and separate plain Python files, plus
a repository commit and push. This task creates the requested editable report
and local supporting code; no external submission or Git push is performed.
