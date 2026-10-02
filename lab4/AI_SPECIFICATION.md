# Lab 4 instruction and correction record

## Actual user request

> can you do this whole lab: https://cabouman.github.io/grad_labs/labs-60141/lab4/index.html? follow all the instructions on the webpage. for the report, create a word document and put the contents in so that I can also make updates to it after you're done. the textbook for the course is included in this directory if you want to refer to it. the lab3 files are also in this directory

## Specification extracted and implemented by the assistant

The following is the assistant's implementation specification derived from the
linked webpage, not an additional prompt written by the student. The page was
downloaded on October 1, 2026, local time; its snapshot is in `references/`.

> Implement the exact eight public function signatures on the Lab 4 page in a
> functions-only `lab4.py`. Reuse `forward`, `adjoint`, `load_pgm` and
> `load_kernel` from Lab 3. Use PyTorch float64, with no pixel loops and no
> nonnegativity constraint. Evaluate the QGGMRF potential and influence in
> Eqs. (3)-(4), and the cost in Eq. (1). Store neighbor differences and
> validity in (8,H,W) tensors ordered NW, N, NE, W, E, SW, S, SE. Axial
> weights are 1/6 and diagonal weights are 1/12; exclude neighbors outside
> the image. Keep the measurement convolution circular. Use a 1e-8 floor
> for the fractional-power argument, and Eq. (15) at or below that floor
> for the surrogate weights. Implement Eq. (14) through its algebraically
> equivalent positive-power form to avoid infinities. Compute the true
> gradient by autograd, checking it against Eq. (16). Build the directional
> Hessian from Eq. (18), including its full double sum with no extra 1/2,
> and use alpha = (g^T g)/(g^T H g) from Eq. (17). Start at y, update
> x = x - omega*alpha*g, and record N+1 true costs, including f(y).
> Use kodim23.pgm and Levin kernel 1, sigma_w=0.02, p=2, q=1.2, omega=1,
> N=50, and one shared noisy observation with seed 0. Run R1 through R6
> using (T,sigma_x)=(100,.036),(1,.020),(1,.010),(1,.040),(100,.018),
> (100,.072). Test the limit, boundaries, gradient, curvature and cost
> decrease before interpreting the results. Generate D1-D11, use one
> fixed 128-by-128 crop, and report unclipped costs/RMSEs and timed runs.

## What went wrong and how it was found

- The first test command failed before executing tests: Python resolved
  `lab4.py` as a module instead of treating the `lab4` directory as a package.
  The exception was `ModuleNotFoundError: 'lab4' is not a package`.
  Adding a package `__init__.py` fixed script/package imports.
- The first numerical implementation passed all six independent test groups
  after that import fix. No mathematical correction was needed to obtain
  passing gradients, pair counts, surrogate curvature, or cost decrease.
- Document-building or figure-formatting revisions do not alter the numerical
  implementation or its measured results. This record does not claim that
  the student supplied a detailed mathematical prompt that they did not write.
