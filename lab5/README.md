# Lab 5 MAP Reconstruction with Plug and Play

Completed from the [Lab 5 instructions](https://cabouman.github.io/grad_labs/labs-60141/lab5/index.html), accessed October 19, 2026.

## Reports and submission

- `output/docx/Lab_5_Report.docx`: editable report with D1–D9 in order, including optional D7 and D8. Text, equations and tables are native Word objects; figures are replaceable PNGs.
- `output/pdf/Lab_5_Report.pdf`: matching submission PDF.
- `lab5.py`: required functions-only implementation with the specified signatures.

The assignment requests the PDF and `lab5.py` as two separate Brightspace uploads, not a zip. Read the report, make any personal edits, and export the edited Word file to PDF before submitting. No Brightspace submission is performed. Regenerating the Word document overwrites it, so preserve manual edits in another copy if rebuilding.

## Reproduce

Use the existing labs Python (`/opt/miniconda3/envs/labs/bin/python` on this machine). The installed DeepInv dependencies are local to `lab5/.deps`; the numerical scripts automatically select them except the test script, which can use `PYTHONPATH` as shown. No earlier lab file is modified.

```bash
PYTHONPATH=lab5/.deps python lab5/test_lab5.py
python lab5/run_experiments.py --part qggmrf
LAB5_DNN_DEVICE=mps python lab5/run_experiments.py --part dnn
python lab5/make_figures.py
python lab5/verify_results.py
```

Omit `LAB5_DNN_DEVICE=mps` to use CPU. MPS requires a compatible Apple GPU and GPU access; CPU is the default. The model uses float32; outer algorithms and QGGMRF use float64. `check_dnn_device.py` compares CPU and MPS inference on the entire observation. Device transfer and dtype conversion deliberately occur separately to avoid a verified backend issue.

For a fresh environment install `requirements.txt` and ensure the prior Lab 3 data exist. Weights download on first use into `lab5/models` under the experiment driver. QGGMRF and DNN runs checkpoint separately and can run concurrently. Existing complete runs are reused: move the appropriate summary and tensor checkpoint files aside to force a fresh run. `--part all` runs the same experiments sequentially in one checkpoint.

Build the editable report with the bundled document Python:

```bash
/Users/vaidehipujary/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python lab5/make_word_report.py
```

The report must be rendered and inspected after rebuilding. The delivered report was rendered with the documents skill's `render_docx.py`, using bundled LibreOffice, and all pages were visually inspected. Rendering previews stay outside the repository.

## Experiment details and instruction conflicts

- Seed 0; unchanged Lab 3 image and kernel 1; circular blur; sigma_w=0.02. The observation is identical to saved Lab 4 data.
- Prior: p=2, q=1.2, T=1, sigma_x=0.020; unordered valid neighbor pairs with axial weight 1/6 and diagonal weight 1/12.
- QGGMRF: 100 outer × 20 inner steps, warm-started from the previous denoiser output. The equal-penalty case uses sigma_p=sigma_n=0.05. D4 fixes sigma_n=0.05 and sweeps sigma_p through 0.025, 0.05, 0.10.
- DRUNet ADMM: 50 steps, sigma_n=0.03, the same three sigma_p settings. Optional PG: 50 steps for (sigma_p,sigma_n)=(0.05,0.03) and (0.025,0.015).
- Task 10b's fixed-denoiser sweep changes prior weights to 4, 1, 0.25. The final checklist's request for identical RMSEs contradicts this. The explicit Task 10b experiment takes precedence; the report explains it.
- The original Lab 4 estimate has only 50 iterations. A direct 1000-step solve of that same objective is also included to distinguish numerical convergence from a change of objective. Its 50-step checkpoint matches saved R2 to round-off.
- A relative iterate-change diagnostic can plateau in a diverging PG run. D7 includes image RMS as additional evidence of divergence; no values are clipped to make the plot agree with a prediction.
- All crops: `[160:288, 384:512]`, zero-based and half-open, 128 by 128. All image panels use [0,1]; RMSEs and algorithms use unclipped arrays.

## Validation and output map

- `test_lab5.py`: FFT/spatial agreement; dense-system proximal check; Lab 4 gradient/curvature comparison; denoiser descent; closed-form quadratic ADMM solution; zero fixed point; signatures and functions-only structure.
- `verify_results.py`: independent RMSE recomputation, all expected history lengths and finite arrays, original Lab 4 autograd stationarity, baseline identity, mean-only denoiser check, and checkpoint hash.
- `results/summary.json` and `results/dnn_summary.json`: raw parameters and iteration histories.
- `results/combined_summary.json`: all results used by the report.
- `results/*reconstructions.pt`: raw float64 arrays, load with `torch.load(..., weights_only=True)`.
- `results/tests.json`, `verification.json`, `dnn_device_check.json`: numerical audit records.
- `results/D*.png`: all required figures; `Q_inner_comparison.png` supports D9 question 2 with an actual 10-inner-step run.
- `AI_SPECIFICATION.md`: actual request, assistant's identified specification and corrections.

Dependencies, model weights, raw tensors and report binaries remain local and are excluded from Git; code, figure PNGs and compact result records are versioned under `lab5/`.
