"""Independent deterministic and distributional checks for the lab samplers."""
import ast
import inspect
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import lab7 as lab

HERE = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(83)
    checks = {}
    tree = ast.parse((HERE/'lab7.py').read_text())
    assert all(isinstance(node, ast.FunctionDef) for node in tree.body)
    signatures = {'neighbor_stats': ['x'], 'gibbs_sweep_gmrf': ['x', 'sigma_x'],
        'qggmrf_delta_energy': ['x', 'w', 'sigma_x', 'p', 'q', 'T'],
        'metropolis_sweep_qggmrf': ['x', 'sigma_x', 'p', 'q', 'T', 'sigma_prop'],
        'gibbs_sweep_posterior': ['x', 'y', 'sigma_w', 'sigma_x'],
        'posterior_statistics': ['y', 'sigma_w', 'sigma_x', 'num_burn', 'num_keep']}
    for name, expected in signatures.items():
        assert list(inspect.signature(getattr(lab, name)).parameters) == expected
    assert all(ast.get_docstring(node) for node in tree.body)
    checks['signatures_docstrings_functions_only'] = True
    x = torch.randn((7, 9))*.05
    kernel = x.new_tensor([[1,2,1],[2,0,2],[1,2,1]])[None,None]/12
    sums, weights = lab.neighbor_stats(x)
    ref_sums = F.conv2d(x[None,None], kernel, padding=1)[0,0]
    ref_weights = F.conv2d(torch.ones_like(x)[None,None], kernel, padding=1)[0,0]
    assert torch.allclose(sums, ref_sums, atol=1e-15, rtol=0)
    assert torch.allclose(weights, ref_weights, atol=1e-15, rtol=0)
    checks['neighbor_max_error'] = float((sums-ref_sums).abs().max())
    assert abs(float(weights[0,0])-5/12) < 1e-15
    assert abs(float(weights[0,1])-8/12) < 1e-15
    # Check all single-pixel replacements simultaneously with vmap, not a pixel loop.
    w = x + .02*torch.randn_like(x)
    replacements = x[None] + torch.eye(x.numel()).reshape(-1, *x.shape)*(w-x)[None]
    reference = torch.vmap(lambda z: lab.qggmrf_energy(z,.05,1.2,2.,.1))(replacements)
    reference = (reference-lab.qggmrf_energy(x,.05,1.2,2.,.1)).reshape(x.shape)
    delta = lab.qggmrf_delta_energy(x,w,.05,1.2,2.,.1)
    checks['delta_energy_max_error'] = float((delta-reference).abs().max())
    assert torch.allclose(delta, reference, atol=5e-13, rtol=0)
    z = x.clone(); z[::2,::2] = w[::2,::2]
    block_error = abs(float(lab.qggmrf_energy(z,.05,1.2,2.,.1)
                           - lab.qggmrf_energy(x,.05,1.2,2.,.1)-delta[::2,::2].sum()))
    assert block_error < 5e-13
    checks['independent_color_energy_error'] = block_error
    # Intrinsic prior must be invariant to an additive brightness shift.
    for energy in (lambda z: lab.gmrf_energy(z,.05), lambda z: lab.qggmrf_energy(z,.05,1.2,2.,.1)):
        assert torch.allclose(energy(x), energy(x+1.7), atol=2e-10, rtol=0)
    checks['prior_shift_invariance'] = True
    before = x.clone()
    torch.manual_seed(11); first, a = lab.metropolis_sweep_qggmrf(x,.05,1.2,2.,.1,.2)
    torch.manual_seed(11); second, b = lab.metropolis_sweep_qggmrf(x,.05,1.2,2.,.1,.2)
    assert torch.equal(x,before) and torch.equal(first,second) and a==b and 0<=a<=1
    checks['seed_reproducibility_and_no_input_mutation'] = True
    # Independently form graph Laplacian from pairwise row/column distances.
    rows, cols = torch.meshgrid(torch.arange(3), torch.arange(4), indexing='ij')
    dr = (rows.flatten()[:,None]-rows.flatten()[None,:]).abs()
    dc = (cols.flatten()[:,None]-cols.flatten()[None,:]).abs()
    adjacent = (dr<=1)&(dc<=1)&((dr+dc)>0)
    weights_matrix = torch.where(adjacent, torch.where((dr+dc)==1,2.,1.)/12,0.)
    laplacian = torch.diag(weights_matrix.sum(1))-weights_matrix
    precision = (torch.eye(12)+laplacian)/.05**2
    covariance = torch.linalg.inv(precision)
    y = torch.randn((3,4))*.1
    exact_mean = torch.linalg.solve(precision,y.flatten()/.05**2).reshape(3,4)
    torch.manual_seed(13)
    state = y.clone()
    draws = []
    for sweep in range(12100):
        state = lab.gibbs_sweep_posterior(state,y,.05,.05)
        if sweep>=100: draws.append(state.flatten())
    samples = torch.stack(draws)
    sample_mean = samples.mean(0)
    sample_cov = torch.cov(samples.T)
    checks['small_posterior_mean_max_error'] = float((sample_mean-exact_mean.flatten()).abs().max())
    checks['small_posterior_covariance_relative_frobenius_error'] = float(torch.linalg.vector_norm(sample_cov-covariance)/torch.linalg.vector_norm(covariance))
    assert checks['small_posterior_mean_max_error'] < .0025
    assert checks['small_posterior_covariance_relative_frobenius_error'] < .07
    # Verify accumulator against the explicit same-seed stack.
    torch.manual_seed(19)
    mean,std,history = lab.posterior_statistics(y,.05,.05,3,12)
    torch.manual_seed(19); state=y.clone(); kept=[]; explicit=[]
    for sweep in range(15):
        state=lab.gibbs_sweep_posterior(state,y,.05,.05)
        explicit.append(float(lab.posterior_energy(state,y,.05,.05)))
        if sweep>=3:kept.append(state)
    stacked = torch.stack(kept)
    assert torch.allclose(mean,stacked.mean(0),atol=1e-15,rtol=0)
    assert torch.allclose(std,stacked.std(0,correction=0),atol=1e-15,rtol=0)
    assert history==explicit
    checks['posterior_moments_and_history'] = True
    if (HERE/'results/images.pt').exists():
        imgs=torch.load(HERE/'results/images.pt',weights_only=True)
        for value in imgs.values():assert value.dtype==torch.float64 and value.shape==(256,256) and torch.isfinite(value).all()
        n,c=lab.neighbor_stats(imgs['map'])
        gradient=(imgs['map']-imgs['noisy']+c*imgs['map']-n)/.05**2
        checks['full_map_gradient_max']=float(gradient.abs().max())
        assert checks['full_map_gradient_max']<1e-9
        saved=json.loads((HERE/'results/summary.json').read_text())
        for key in ('single_sample','posterior_mean','map'):
            actual=float((imgs[key]-imgs['clean']).square().mean().sqrt())
            assert abs(actual-saved['posterior']['rmse'][key])<1e-14
        histories=json.loads((HERE/'results/histories.json').read_text())
        assert len(histories['prior_mean'])==500
        assert len(histories['posterior_statistics_energy'])==500
        assert all(len(histories[f'qggmrf_{w}']['energy'])==300 for w in (.002,.02,.2))
        checks['full_results_shapes_rmse_history_lengths']=True
    (HERE/'results').mkdir(exist_ok=True)
    (HERE/'results/verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
