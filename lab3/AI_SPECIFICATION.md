# Lab 3: instruction record

## User's actual request

> I want you to help me complete this whole lab: https://cabouman.github.io/grad_labs/labs-60141/lab3/index.html. I've created a lab3.py file already in the lab3 folder to complete all the code and I included the picture and folder in there as well. complete the lab step by step. I've also placed the textbook in the directory so you can access it for reference and make sure everything is consistent with the information given there. now go ahead and do lab 3.

## Implementation specification extracted by the assistant

This is the assistant's interpretation of the linked instructions, not a second
prompt written by the student. The live page was read on September 19, 2026.

1. Preserve the supplied `load_pgm(path)` and `load_kernel(path)` readers.
   Implement the lab's exact function signatures in `lab3.py`, with short
   docstrings. Keep experiments in a separate importing script. Use PyTorch
   `float64`, `(H, W)` images, odd `(Ka, Kb)` kernels, and no pixel loops.
2. Implement `forward(x, a)` as circular convolution, using circular padding
   followed by `F.conv2d` with the kernel flipped along both axes. Implement
   `adjoint(v, a)` with the same padding and the unflipped kernel. Include
   `adjoint_autograd(v, a)` as the gradient of `<forward(x,a),v>` with respect
   to a zero image `x` with `requires_grad=True`.
3. Use the symmetric neighbor kernel
   `g = [[1,2,1],[2,0,2],[1,2,1]] / 12`. In the prior only, use zero-padded
   convolution and `c = conv2d(ones, g)` to count existing neighbors.
   `prior_grad(x,sigma_x) = (c*x - conv2d(x,g)) / sigma_x**2`.
   `prior_cost(x,sigma_x) = 0.5*sum(x*prior_grad(x,sigma_x))`.
   Each unordered pair occurs once in the equivalent pairwise energy.
4. Implement `cost(x,y,a,sigma_w,sigma_x)` as
   `sum((y-forward(x,a))**2)/(2*sigma_w**2) + prior_cost(x,sigma_x)`.
   Implement `cost_grad(x,y,a,sigma_w,sigma_x)` as
   `adjoint(forward(x,a)-y,a)/sigma_w**2 + prior_grad(x,sigma_x)`.
5. Implement `gradient_descent(y,a,sigma_w,sigma_x,num_iters,alpha=None)`
   starting at `y`, with default
   `alpha = 1/(1/sigma_w**2 + 2/sigma_x**2)` and exactly `num_iters` updates.
   Return the restored tensor and `num_iters+1` costs, including `c(y)`.
   A separate diagnostic wrapper may additionally expose per-step NRMSE
   without changing this required return signature.
6. Run all five required checks before plotting results. Also independently
   check the direction of a shifted impulse kernel, circular wraparound,
   the unordered-pair prior energy, constant-image behavior, corner
   behavior, and convergence to a stationary point on a small image.
7. Use the supplied full-resolution `kodim23.pgm` and kernel 1. Use seed 0
   with a fresh CPU `torch.Generator` for each experiment, so all three
   prior strengths within an experiment receive identical noisy data.
   Denoising uses the 1x1 identity kernel and `sigma_w=0.05`;
   deconvolution uses kernel 1 and `sigma_w=0.02`.
8. For both experiments, use `sigma_x` in `{0.01,0.05,0.25}` and run all
   50 requested iterations. Record the first iteration with
   `norm(x_new-x_old)/norm(x_old) < 0.002`. Report 50-step images and RMSEs
   and explicitly distinguish them from the first stopping-rule crossing.
   Never clip data, reconstructions, costs, or RMSEs; clip only for display.
9. Produce D1-D10 in order, with arithmetic for memory use, the interior
   prediction-error standard deviation, the algebraic strict-convexity
   proof, both convergence panels, timing, and the four-times-step-size
   experiment. Cite the supplied textbook and real measured results.

## Clarifications during implementation

- The search engine's cached `labs/lab3` page described older 500/2000-step
  experiments and different parameter sets. The live URL supplied by the
  user specifies 50 steps and `{0.01,0.05,0.25}`; the live specification was
  used before running the experiments.
- The live page specifies both 50-step runs and a 0.2% stopping rule. The
  implementation retains all 50 steps and records the first threshold
  crossing separately. Its final checklist also says "five items", although
  each current experiment actually enumerates four. All current items and
  the numerical monotonicity check are included.
- The book's Eq. (5.13) scales its update direction by the noise variance.
  The lab's alpha multiplies the unscaled gradient, so it corresponds to
  beta in the textbook discussion on p. 67, not the book's scaled alpha.
- The lab's difference-only prior has a constant-image nullspace. It is
  an intrinsic GMRF energy rather than a normalized, nonsingular Gaussian
  density on all of R^N. Its posterior is strictly convex for denoising.
- All required and additional correctness checks passed on the first run.
  No user-issued implementation corrections occurred.
