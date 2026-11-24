# Lab 6 work record

The assistant prepared the specifications in SPECIFICATIONS.json before implementation. They are reproduced verbatim in the report; they are not claimed to be independently written student specifications.

Pre-experiment prediction for Step 9, recorded before running the five starts: the five random starts will not all reach the same final cost as the first-three-samples start after 20 iterations. Different starting means can occupy different cluster regions and lead to different likelihood basins or rates of convergence. This is an assistant-generated prediction, not a claim about a prior student prediction.

Clarifications made when translating the lab into specifications, before code: include iteration zero in the returned history while plotting iterations 1 through 20; compare hard-label ML covariances with the same 1e-6 identity addition; specify array shapes, empty-component behavior, deterministic merge ordering, and duplicate cumulative iteration coordinates for post-merge MDL values.

All required numerical checks passed. An independent explicit two-dimensional Gaussian and raw-second-moment implementation reproduced all six M = 3 fits to within 4.548e-13, and independent determinant calculations selected all eight of the same merge pairs. No saved specification needed a post-failure change. A plot annotation placed outside the data range was corrected during visual document QA; this was a figure-layout correction, not an EM numerical failure.
