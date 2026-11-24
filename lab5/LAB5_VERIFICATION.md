# Lab 5 verification

Audit date: October 23, 2026. Reviewed the current edited report `output/docx/ECE 60141 Lab 5 Report.docx`, the implementation, original tensors and histories, the saved assignment snapshot, the live assignment, and the local Bouman textbook. The edited Word report and reconstruction implementation were not changed.

**Task 10b is implemented correctly: its three measured RMSEs are different.** The final checklist's request to explain equal RMSEs is inconsistent with Task 10b. Equality of the converged solution applies when the ratio of the two proximal variances stays fixed. It does not apply when only sigma_p changes and sigma_n stays fixed.

The numerical results and D9 mathematical answers check out. Report wording, attribution, pagination, and submission-file issues are listed below.

## Textbook evidence and derivation

The source is `../Foundations_of_Computational_Imaging_-_Charles_A_Bouman copy.pdf`. Page numbers below are the printed textbook pages; the PDF page numbers identify the corresponding physical pages in this local copy.

| Material inspected | What matters here |
| --- | --- |
| Section 9.5, Figure 9.5, equations (9.46)-(9.48), p. 141 (PDF 145) | The two ADMM minimizations use the same augmented-Lagrangian coefficient a. |
| Section 10.1.1, equations (10.2)-(10.3), p. 153 (PDF 156) | F and H both use the same sigma squared. |
| Section 10.3, equations (10.5)-(10.6), p. 158 (PDF 161) | At equilibrium, x = F(x-u) = H(x+u). |
| Section 10.3.1, Theorem 10.2, p. 159 (PDF 162) | For convex proximal agents defined with the shared parameter, CE and MAP solutions agree. |
| Section 10.4, p. 165 (PDF 168) | The selectable parameter does not affect the final solution when it is shared by the proximal agents of fixed objectives. |
| Section 10.5, problem 1, p. 168 (PDF 171) | The MAP/CE equivalence exercise explicitly assumes the same parameter in F and H. |

Write the lab's two maps as

\[
F(z)=\arg\min_x\left[f(x)+\frac{\|x-z\|^2}{2\sigma_p^2}\right],\qquad
H(z)=\arg\min_x\left[h(x)+\frac{\|x-z\|^2}{2\sigma_n^2}\right].
\]

At a fixed point, the dual update implies x = v. The optimality conditions for the first two updates are therefore

\[
\nabla f(x^*)+\frac{u^*}{\sigma_p^2}=0,\qquad
\nabla h(x^*)-\frac{u^*}{\sigma_n^2}=0.
\]

Multiply the second equation by sigma_n squared / sigma_p squared and add it to the first:

\[
\nabla f(x^*)+\frac{\sigma_n^2}{\sigma_p^2}\nabla h(x^*)=0.
\]

Thus the target is

\[
x^*=\arg\min_x\{f(x)+\lambda h(x)\},\qquad
\lambda=\frac{\sigma_n^2}{\sigma_p^2}.
\]

The same conclusion follows by multiplying the denoiser's objective by the positive number lambda: it becomes lambda h plus a squared-distance penalty with variance sigma_p squared. The pair is ordinary ADMM for f + lambda h with one shared penalty.

With sigma_n fixed at 0.05, the prescribed sweep changes lambda to 4, 1, and 0.25. Smaller sigma_p increases smoothing; larger sigma_p reduces it and admits more noise. Holding sigma_n = sigma_p instead keeps lambda = 1, so all sufficiently converged runs target the same MAP image. Finite iteration counts and approximate inner solves can still produce small discrepancies in that latter experiment.

The lab potential is convex. For p = 2, q = 1.2, T = 1, put t = (abs(delta)/sigma_x)^0.8. Direct differentiation of the lab potential gives

\[
\rho''(\delta)=\frac{1+0.48t+0.12t^2}{\sigma_x^2(1+t)^3}>0.
\]

The neighbor graph is connected. Its prior has curvature in every nonconstant direction, while the normalized blur observes the constant direction. The mathematical weighted objectives therefore have unique minimizers. The tiny 1e-8 safeguard inherited from Lab 4 was also checked against an independently implemented, unfloored potential; it does not explain the observed differences.

There is a notation trap in Section 6.3, p. 93 (PDF 98): the textbook describes its p/q convention and flags a swap relative to its reference [85]. The audit uses the lab's explicit potential and prescribed p = 2, q = 1.2, rather than substituting the textbook's parameter labels into the lab. The surrogate curvature was checked against Section 8.3, equations (8.16)-(8.20), pp. 114-115 (PDF 119-120).

**Precision of the conclusion:** changing an objective does not logically force unequal scalar RMSEs for every possible image. Different estimates can lie at equal distances from the truth. For example, minimizing (x-1)^2/2 + lambda*x^2/2 gives x = 1/(1+lambda); lambda = 1 and 3 produce 0.5 and 0.25, both at distance 0.125 from truth 0.375. The proof establishes the changing target; fresh computations establish unequal RMSEs for this lab's actual image and noise.

The [live assignment](https://cabouman.github.io/grad_labs/labs-60141/lab5/index.html), checked October 23, has the same contradiction as the saved October 19 snapshot. Task 10b, the parameter table, and its adjacent D4 hand-in instructions all describe the fixed-denoiser experiment. The final checklist asks for equality instead. Task 12b also has a comparison with Module 3 that is inconsistent with the changed Module 3 sweep. These inconsistencies do not alter the equations.

## Independent numerical evidence

Every QGGMRF sweep was rerun from the original observation. All three reconstructed arrays and all three 100-entry histories matched the saved results exactly. A separately written implementation of the unfloored prior, its gradient, and direct weighted MAP descent independently reached the following references. Its gradient was checked against automatic differentiation; it uses neither Lab 4 nor Lab 5 prior helpers.

| sigma_p | lambda | Prescribed PnP RMSE | Independent converged MAP RMSE | RMS difference between images |
| --- | --- | --- | --- | --- |
| 0.025 | 4 | 0.03681756367350207 | 0.03681755425558719 | 9.5029e-8 |
| 0.050 | 1 | 0.02963931855671826 | 0.02963929613271914 | 1.6377e-7 |
| 0.100 | 0.25 | 0.04807676426664232 | 0.04807676424298353 | 3.6779e-10 |

Each independent reference reached gradient RMS below 1e-9. At the saved sweep images, the correct weighted-objective gradient RMS values are 1.5070e-5, 6.5400e-6, and 7.2209e-9. Evaluating the *unweighted* objective at the two outer sweep points gives gradient RMS 8.4085 and 12.7625. Those two images are demonstrably not stationary points of the original unweighted MAP objective.

A separate counterfactual experiment changes both sigmas together, keeping the original prior and lambda = 1. It confirms the textbook's parameter invariance at convergence:

| sigma_p = sigma_n | Outer iterations, 20 inner each | RMSE | RMS difference from original converged MAP |
| --- | --- | --- | --- |
| 0.025 | 100 | 0.02971153042715829 | 2.4531205e-4 |
| 0.025 | 500 | 0.02963929733856767 | 1.0072101e-8 |
| 0.050 | 100 | 0.02963931855671826 | 1.6379236e-7 |
| 0.100 | 100 | 0.02963929613053404 | 4.1695373e-14 |

The 0.025 control also shows why even a shared-parameter experiment should not demand exactly identical RMSEs at an arbitrary fixed iteration count. Its target is unchanged, but 100 iterations are insufficient for the same numerical accuracy. These controls are diagnostic additions, not replacements for the prescribed fixed-sigma_n D4 sweep.

The image, seed-0 noise realization, kernel normalization, and equality with the Lab 4 observation were checked. Full-image RMSE was recomputed from every raw float64 reconstruction without clipping. All arrays are finite; all histories have the required lengths.

All five DRUNet experiments were rerun for 50 iterations with the same pretrained grayscale weights on MPS. The replay independently implements the ADMM loop using the equivalent direct Fourier quotient and independently implements the PG loop. The largest RMS difference from a saved neural reconstruction is 1.881e-16; the largest difference in a saved history is 3.731e-14. This is a reproducibility check of these runs, not a proof of global convergence of DRUNet. Existing CPU/MPS single-inference agreement remains documented in `results/dnn_device_check.json`; a full CPU trajectory was not rerun in this audit.

## Verification of the remaining deliverables

| Item | Verified result and interpretation |
| --- | --- |
| D1 | Full-image FFT/spatial maximum error 7.21645e-16, adjoint error 6.66134e-16. Data residual RMS decreases from 0.02344504 for y to 0.02280504 and 0.00229167. The large-prox output fits data better but amplifies noise. The two full-image normal-equation errors are below 1.7e-12. |
| D2 | The quoted technical specification describes the implemented functions accurately, and the correction record agrees with the files. Attribution needs clarification in the edited report; see below. |
| D3 | PnP RMSE 0.02963931856; original converged direct MAP RMSE 0.02963929613; image RMS difference 1.6379236e-7. The 50-step Lab 4 baseline has RMSE 0.02990709424 and is not fully converged. Its reproduced array matches the original R2 within 1.38778e-15 maximum absolute error. |
| D5 | Input RMSE 0.04753759210; DRUNet RMSE 0.01939073224; original Lab 4 baseline RMSE 0.02990709424. The reported 35.2% RMSE reduction is correct. Final consensus residual 7.6689734e-5. |
| D6 | RMSEs 0.02463010373, 0.01939073224, and 0.14844127065. The last large value is reproduced and is not a transcription error. The crops support the smoothing/noise explanation. At sigma_p = 0.10 the final consensus residual is 9.1138e-5, while relative image change is still 0.0011394; describe this as the prescribed 50-step result rather than a proved exact equilibrium. |
| D7 | The linear data-step ceiling is 0.02828427125. Sigma_p = 0.05 violates it. Image RMS increases from about 0.472 to 10711.1193; final RMSE is 10711.0217. The relative-change plateau at 1.42986 is compatible with divergence. The report correctly avoids assuming a global nonexpansiveness guarantee for DRUNet. |
| D8 | Rescaled PG RMSE 0.02282338016, compared with ADMM 0.01939073224. The recorded diagnostics stay below 1e-3 from iterations 29 and 7 respectively. These are different diagnostics, not proof of identical convergence accuracy or a wall-clock speed ratio. Keeping the parameter ratio fixed does not prove that two DRUNet noise levels have identical fixed points. |

All 13 embedded Word figures are byte-for-byte copies of the saved figure PNGs. The figures use the same 128 by 128 crop, rows 160:288 and columns 384:512 with Python's zero-based, end-exclusive indexing, and display range [0,1]. The computations retain out-of-range values. The required signatures, docstrings, functions-only source organization, Fourier centering, dense-system proximal agreement, QGGMRF gradient and surrogate update, quadratic ADMM reference, and zero fixed point all pass the existing numerical tests.

The five D9 answers are mathematically consistent:

1. Shared sigma_n = sigma_p and the same prior determine the unweighted MAP objective. Warm starting improves the finite inner solve's accuracy; it is not what makes an exact proximal map mathematically valid. Relaxing equality of the sigmas gives the two-parameter setting. The report handles this correctly.
2. Halving inner iterations can worsen finite-iteration accuracy, but a persistent residual floor is not inevitable. The additional 10-step experiment was rerun and matches exactly: RMSE 0.02963932040 and final residual 2.8448292e-9, compared with 2.6552459e-9 for 20 steps. The two RMSEs agree when rounded to seven decimal places.
3. sigma_n controls the noise level requested from the network; sigma_p controls the data-proximal correction. There is no explicit sigma_x in this DRUNet wrapper. Their effects should be tuned and evaluated empirically; an exact scalar prior-weight interpretation has not been proved for this network.
4. For the mean projection P, v = P(x+u) and u_new = (I-P)(x+u). Thus v is constant and u has zero mean. At consensus, minimizing the data error over constants yields x = mean(y)*1 for the normalized circular blur. The independently rerun mean-denoiser check gives mean(y) = 0.4302206535, residual 1.26305e-7, and RMSE 0.1860681891. Agreement between agents does not imply a useful image.
5. An indicator of independently calibrated bounds [0,1]^N is a valid third agent. Its proximal map is projection onto that box. Including it in a three-agent CE formulation is a coherent proposal; the submitted experiments correctly do not silently add this constraint.

## Report changes to make before submission

The audit rendered and inspected all ten pages of the current edited Word file, whose SHA256 is `d56d3d13f68dcce6ed0f5dbbcaf15cb8de2bf65cbb155ef5a36654c2d5a6d747`.

1. **D4, page 4:** the numerical explanation is right, but the explicit checklist-conflict paragraph was removed. Page 2 still promises that the conflict will be addressed below, and page 10 only mentions it briefly. A useful replacement sentence is: “Following Task 10b, sigma_n remains 0.05, so changing sigma_p changes the prior weight to 4, 1, and 0.25 and gives different measured RMSEs; equality of the converged MAP solution would instead require holding sigma_n squared / sigma_p squared fixed.”
2. **D2, page 2:** `AI_SPECIFICATION.md` identifies the quoted detailed specification as the assistant's translation of the assignment, not the user's verbatim original prompt. The edited report removed that provenance label. Restore the distinction and include the actual request if the intention is to quote what was originally given to the assistant.
3. **Title, page 1:** the date reads October 25, 2026, while this audit is October 23. Use the intended report/submission date.
4. **D7, pages 6-7:** the final two lines spill onto an otherwise empty page in the rendered document. Keep the short final paragraph together or recover a little space on page 6 before exporting.
5. **Submission output:** there was no final submission PDF under `lab5/output` at audit time. The README and old `deliverable_checks.json` refer to earlier paths/hashes and a nine-page document; they do not certify the current edited ten-page file. Export and inspect the final edited document before submitting it.

Audit records are in `results/audit_qggmrf.json`, `results/audit_dnn.json`, `results/audit_additional.json`, the rerun `results/tests.json`, and the rerun `results/verification.json`. `audit_lab5.py` reproduces the new weighted-MAP and neural checks. No publication, repository push, or submission was performed.
