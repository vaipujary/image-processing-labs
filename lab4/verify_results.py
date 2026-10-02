"""Verify saved outputs with an independent FFT blur and four-pair prior."""
import json
from pathlib import Path
import sys

import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from lab4.lab4 import true_gradient, forward, adjoint, neighbor_diffs, rho_prime


def main():
    torch.set_num_threads(4)
    arrays=torch.load(HERE/'results/reconstructions.pt',weights_only=True)
    s=json.loads((HERE/'results/summary.json').read_text())
    x,y,k=arrays['x_true'],arrays['y'],arrays['a']
    psf=torch.zeros_like(x)
    kh,kw=k.shape
    psf[:kh,:kw]=k
    psf=torch.roll(psf,shifts=(-(kh//2),-(kw//2)),dims=(0,1))
    spectrum=torch.fft.rfft2(psf)
    blur=lambda z:torch.fft.irfft2(torch.fft.rfft2(z)*spectrum,s=z.shape)
    result={'independent_blur_max_abs_error':float((blur(x)-forward(x,k)).abs().max()),'runs':{}}
    for name,run in s['runs'].items():
        z=arrays[name]
        sx,T=run['sigma_x'],run['T']
        # Exact positive-power formula, without the implementation's floor.
        potential=lambda d:d.abs().square()/(2*sx**2*(1+(d.abs()/(T*sx)).pow(.8)))
        prior=(potential(z[:,1:]-z[:,:-1]).sum()/6
              +potential(z[1:,:]-z[:-1,:]).sum()/6
              +potential(z[1:,1:]-z[:-1,:-1]).sum()/12
              +potential(z[1:,:-1]-z[:-1,1:]).sum()/12)
        independent_cost=float((blur(z)-y).square().sum()/(2*.02**2)+prior)
        independent_rmse=float((z-x).square().mean().sqrt())
        cost_error=abs(independent_cost-run['final_cost'])
        assert cost_error < 1e-6,(name,cost_error)
        assert abs(independent_rmse-run['rmse']) < 1e-14
        assert len(run['cost_history'])==51
        assert all(v<u for u,v in zip(run['cost_history'],run['cost_history'][1:]))
        result['runs'][name]={'independent_final_cost':independent_cost,'cost_abs_error':cost_error,
                              'independent_rmse':independent_rmse,'all_50_cost_changes_negative':True}
    gen=torch.Generator().manual_seed(41)
    xx=torch.randn((31,37),generator=gen,dtype=torch.float64,requires_grad=True)
    yy=torch.randn((31,37),generator=gen,dtype=torch.float64)
    grad=true_gradient(xx,yy,k,.02,.02,2.,1.2,1.)
    diffs,valid=neighbor_diffs(xx.detach())
    weights=xx.new_tensor([1/12,1/6,1/12,1/6,1/6,1/12,1/6,1/12])[:,None,None]
    reference=adjoint(forward(xx.detach(),k)-yy,k)/.02**2+(valid*weights*rho_prime(diffs,.02,2.,1.2,1.)).sum(0)
    result['autograd_vs_equation16_max_abs_error']=float((grad-reference).abs().max())
    previous_path=HERE.parent/'lab3/results/deconvolution_arrays.pt'
    if previous_path.exists():
        previous=torch.load(previous_path,weights_only=True)
        result['lab3_array_keys']=list(previous.keys())
        if 'y' in previous:
            result['measurement_vs_lab3_max_abs_error']=float((previous['y']-y).abs().max())
    result['all_passed']=True
    (HERE/'results/verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
