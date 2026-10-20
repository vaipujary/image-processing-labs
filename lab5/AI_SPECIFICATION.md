# Lab 5 AI specification and correction record

## Actual user request

> can you look at this lab and follow all the instructions to do it in the lab 5 folder I've created? like the previous labs, create a word document for the report so that I can make updates to it once you've populated the report. here is the link: https://cabouman.github.io/grad_labs/labs-60141/lab5/index.html. make no mistakes.

## Assistant implementation specification

The following is the assistant's translation of the linked requirements, not a separately authored student prompt:

> Implement denoise_qggmrf(v, sigma_n, sigma_x, p, q, T, num_iters) as the unconstrained QGGMRF MAP denoiser with the Lab 4 potential, valid eight-neighbor pairs, axial weights 1/6 and diagonal weights 1/12. Use one steepest-descent update of a freshly touching quadratic surrogate per inner iteration, p=2, q=1.2, T=1, and float64. Implement pnp_admm(y, a_hat, denoiser, sigma_w, sigma_p, num_iters) with v=y and u=0, then x=F(v-u), v=denoiser(x+u), u=u+x-v. Return the final x and one ||x-v||/||x|| value per iteration. Keep the denoiser independent of the generic loop. In pnp_qggmrf_map, warm-start every inner solve from the previous denoiser output; use exactly 100 outer and 20 inner iterations. Preserve all prescribed function signatures in a functions-only lab5.py; keep experiments separate. Use seed 0, the unchanged Lab 3 image and kernel, sigma_w=0.02, and the same observation in all experiments. Do not clip numerical values. Use pretrained grayscale DeepInv DRUNet in float32, with float64 inputs and outputs at the wrapper boundary, and run the stated sweeps and both optional 50-step PnP-PG settings. Report measured values and explain any disagreement with the webpage.

## Decisions and corrections

- Task 10b and the parameter table explicitly fix sigma_n=0.05 while varying sigma_p. Their prior weights are 4, 1 and 0.25. The final checklist's phrase asking why the RMSEs are the same is inconsistent with that experiment; the task and equation determine the implementation.
- The 50-step Lab 4 R2 baseline is not exactly converged. It is reproduced with mathematically equivalent FFT data operations, compared against the saved Lab 4 result, and retained for the D5 comparison. A 1000-step direct solve of that same objective separately establishes the MAP agreement in D3.
- The first quadratic reference test used alpha=0.7 and only 600 ADMM steps, insufficient to reach the chosen 1e-5 tolerance in weakly observed modes. The test now uses alpha=100, keeping the independent closed-form reference and tolerance unchanged. No algorithm formula was changed to pass it.
- CPU DRUNet was stopped before its first completed experiment because sandbox restrictions hid the available GPU. The complete neural-network experiments restart from the same y using float32 MPS inference; all FFTs and outer updates remain float64 on CPU. CPU remains the default device in lab5.py.
- An initial GPU trial exposed invalid values from combined dtype/device transfers in PyTorch. It was discarded. The wrapper now casts on CPU before transferring to MPS, and transfers back before casting to float64. A full-image comparison against CPU measured maximum error 1.430511474609375e-6 and RMS error 1.9865320507943666e-7 before the accepted GPU runs began.
- The D7 normalized change is a relative diagnostic; divergence can make it plateau rather than grow without bound. Its plot also records the image RMS to distinguish this plateau from convergence.
- The QGGMRF proximal solve is approximate at finite inner iteration count. A warm start improves its accuracy; it does not mathematically make 20 steps an exact proximal map.
