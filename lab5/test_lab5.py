"""Independent numerical checks for Lab 5; no model download required."""
import ast
import inspect
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import lab5.lab5 as m
from lab3.lab3 import forward, adjoint
from lab4.lab4 import true_cost, true_gradient, optimal_step_size


def main():
    torch.set_num_threads(4)
    gen = torch.Generator().manual_seed(123)
    x = torch.randn((19,23),generator=gen,dtype=torch.float64)
    a = torch.rand((5,7),generator=gen,dtype=torch.float64); a /= a.sum()
    y = torch.randn(x.shape,generator=gen,dtype=torch.float64)
    ah = m.kernel_dft(a,x.shape)
    checks = {}
    checks['fft_forward_max_error'] = float((forward(x,a)-torch.fft.ifft2(ah*torch.fft.fft2(x)).real).abs().max())
    checks['fft_adjoint_max_error'] = float((adjoint(x,a)-torch.fft.ifft2(ah.conj()*torch.fft.fft2(x)).real).abs().max())
    z = m.prox_data(x,y,ah,.02,.05)
    normal = adjoint(forward(z,a)-y,a)/.02**2+(z-x)/.05**2
    checks['prox_normal_equation_max_error'] = float(normal.abs().max())
    identity = torch.ones((1,1),dtype=torch.float64)
    pg,c = m._prior_stats(x,.02,2.,1.2,1.)
    ag = true_gradient(x.requires_grad_(),x.detach(),identity,.05,.02,2.,1.2,1.)
    checks['prior_gradient_vs_lab4_max_error'] = float((pg-ag).abs().max())
    x = x.detach()
    g = (x-y)/.05**2+pg
    denom = g.square().sum()/.05**2
    for (l,r,_),w in zip(m._pairs(),c): denom += (w*(g[l]-g[r]).square()).sum()
    step = g.square().sum()/denom
    checks['surrogate_step_vs_lab4_error'] = float(abs(step-optimal_step_size(x,g,identity,.05,.02,2.,1.2,1.)))
    den = m.denoise_qggmrf(y,.05,.02,2.,1.2,1.,100)
    checks['denoising_cost_decrease'] = float(true_cost(y,y,identity,.05,.02,2.,1.2,1.)-true_cost(den,y,identity,.05,.02,2.,1.2,1.))
    fixed,h = m.pnp_admm(torch.zeros_like(y),ah,lambda z:z,.02,.05,5)
    assert h == [0.]*5 and fixed.count_nonzero()==0
    # Exact quadratic denoiser gives an independently known MAP solution.
    alpha=100.
    out,h=m.pnp_admm(y,ah,lambda z:z/(1+.05**2*alpha),.02,.05,600)
    exact=torch.fft.ifft2(ah.conj()*torch.fft.fft2(y)/(ah.abs().square()+.02**2*alpha)).real
    checks['quadratic_map_rms_error']=float((out-exact).square().mean().sqrt())
    # Explicit dense solve independently validates FFT centering and proximal formula.
    small=torch.randn((5,6),generator=gen,dtype=torch.float64)
    ak=a[:3,:3];ak=ak/ak.sum(); ahk=m.kernel_dft(ak,small.shape)
    basis=torch.eye(30,dtype=torch.float64)
    matrix=torch.stack([forward(e.reshape(5,6),ak).flatten() for e in basis],dim=1)
    dense=torch.linalg.solve(matrix.T@matrix/.02**2+basis/.05**2,
                            matrix.T@small.flatten()/.02**2+small.flatten()/.05**2)
    checks['prox_vs_dense_max_error']=float((m.prox_data(small,small,ahk,.02,.05).flatten()-dense).abs().max())
    for key,value in checks.items():
        if 'max_error' in key or 'step_vs' in key: assert value<1e-9,(key,value)
    assert checks['denoising_cost_decrease']>0
    assert checks['quadratic_map_rms_error']<1e-5
    tree=ast.parse(Path(m.__file__).read_text())
    assert all(isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef)) for n in tree.body)
    checks['functions_only']=True
    checks['signatures']={n:str(inspect.signature(getattr(m,n))) for n in
        ['kernel_dft','prox_data','denoise_qggmrf','pnp_admm','pnp_qggmrf_map','denoise_dnn','pnp_prox_grad']}
    assert checks['signatures']=={
        'kernel_dft':'(a, shape)',
        'prox_data':'(v, y, a_hat, sigma_w, sigma_p)',
        'denoise_qggmrf':'(v, sigma_n, sigma_x, p, q, T, num_iters)',
        'pnp_admm':'(y, a_hat, denoiser, sigma_w, sigma_p, num_iters)',
        'pnp_qggmrf_map':'(y, a_hat, sigma_w, sigma_p, sigma_n, sigma_x, p, q, T, num_iters, inner_iters)',
        'denoise_dnn':'(v, sigma_n)',
        'pnp_prox_grad':'(y, a_hat, denoiser, sigma_w, sigma_p, num_iters)'}
    assert all(inspect.getdoc(getattr(m,n)) for n in checks['signatures'])
    Path(__file__).with_name('results').mkdir(exist_ok=True)
    Path(__file__).with_name('results').joinpath('tests.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__': main()
