"""Check saved numbers with the original Lab 4 autograd implementation."""
import hashlib
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE/'.deps'))
import torch
from lab4.lab4 import true_gradient, true_cost
from lab3.lab3 import forward
from lab5.lab5 import kernel_dft, pnp_admm


def main():
    torch.set_num_threads(4)
    s=json.loads((HERE/'results/combined_summary.json').read_text())
    a=torch.load(HERE/'results/combined_reconstructions.pt',weights_only=True)
    v={'runs':{}}
    for name,r in s['runs'].items():
        x=a[name];rmse=float((x-a['x_true']).square().mean().sqrt())
        assert abs(rmse-r['rmse'])<1e-14
        assert x.dtype==torch.float64 and torch.isfinite(x).all()
        expected=100 if name.startswith('Q') else 50 if name.startswith(('D','PG')) else 0
        assert len(r['history'])==expected
        v['runs'][name]={'rmse_recomputed':rmse,'finite':True,'history_length':expected}
    for name in ['MAP_converged','MAP50','Q05','Q025','Q10']:
        x=a[name].detach().requires_grad_()
        lam=(.05/s['runs'][name]['sigma_p'])**2 if name.startswith('Q') else 1.
        # Scaling sigma_w by sqrt(lambda) lets the existing cost represent f/lambda+h.
        grad=true_gradient(x,a['y'],a['a'],.02*lam**.5,.02,2.,1.2,1.)*lam
        v['runs'][name]['weighted_map_gradient_rms']=float(grad.square().mean().sqrt())
        v['runs'][name]['map_cost']=float(true_cost(x,a['y'],a['a'],.02,.02,2.,1.2,1.).detach())
    assert v['runs']['MAP_converged']['weighted_map_gradient_rms']<1e-7
    assert s['qggmrf_vs_converged_map_rms']<1e-5
    assert s['map50_vs_saved_lab4_max_error']<1e-12
    assert s['lab4_observation_identical']
    assert s['runs']['D05']['rmse']<s['runs']['MAP50']['rmse']
    ah=kernel_dft(a['a'],a['y'].shape)
    out,h=pnp_admm(a['y'],ah,lambda z:torch.full_like(z,z.mean()),.02,.05,100)
    v['mean_denoiser']={'target_mean':float(a['y'].mean()),'output_mean':float(out.mean()),
                       'max_difference_from_mean':float((out-a['y'].mean()).abs().max()),
                       'rmse':float((out-a['x_true']).square().mean().sqrt()),'final_residual':h[-1]}
    v['weights_sha256']=hashlib.sha256((HERE/'models/hub/checkpoints/drunet_deepinv_gray_finetune_26k.pth').read_bytes()).hexdigest()
    v['stability_ceiling']=float((2*.02**2/ah.abs().square().max()).sqrt())
    def settle(history):
        return next((i+1 for i in range(len(history)) if max(history[i:])<1e-3),None)
    v['below_1e_3_through_end']={k:settle(s['runs'][k]['history']) for k in ['D05','PG_stable','Q05','Q05_inner10']}
    (HERE/'results/verification.json').write_text(json.dumps(v,indent=2)+'\n')
    print(json.dumps(v,indent=2))


if __name__=='__main__': main()
