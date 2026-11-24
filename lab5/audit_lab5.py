"""Independent October 23 audit. Writes new audit records, never report/results inputs.

Run with the labs Python; use --part dnn for fresh neural-network replays.
The independent weighted MAP reference implements the unfloored equation (8)
directly and uses its own surrogate descent, without Lab 4/5 prior helpers.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE / '.deps'))
os.environ.setdefault('TORCH_HOME', str(HERE / 'models'))
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.mplconfig'))
import torch
from lab5 import lab5 as m


def rms(x):
    return float(x.square().mean().sqrt())


def pairs(x):
    return [(x[:-1, :], x[1:, :], 1/6),
            (x[:, :-1], x[:, 1:], 1/6),
            (x[:-1, :-1], x[1:, 1:], 1/12),
            (x[:-1, 1:], x[1:, :-1], 1/12)]


def prior(x):
    return sum(b * ((a-c).square() / (.0008 * (1 + ((a-c).abs()/.02)**.8))).sum()
               for a, c, b in pairs(x))


def stats(x):
    g = torch.zeros_like(x)
    weights = []
    destinations = [(g[:-1, :], g[1:, :]), (g[:, :-1], g[:, 1:]),
                    (g[:-1, :-1], g[1:, 1:]), (g[:-1, 1:], g[1:, :-1])]
    for (a, c, b), (ga, gc) in zip(pairs(x), destinations):
        d = a-c
        t = (d.abs()/.02)**.8
        w = b * (1+.6*t) / (.0004 * (1+t)**2)
        ga += w*d
        gc -= w*d
        weights.append(w)
    return g, weights


def independent_map(y, ah, lam):
    x = y.clone()
    yh = torch.fft.fft2(y)
    for i in range(3000):
        pg, w = stats(x)
        g = torch.fft.ifft2(ah.conj()*(ah*torch.fft.fft2(x)-yh)).real/.0004 + lam*pg
        if rms(g) < 1e-9:
            break
        ag = torch.fft.ifft2(ah*torch.fft.fft2(g)).real
        den = ag.square().sum()/.0004
        den += lam*sum((wi*(a-c).square()).sum() for (a,c,_),wi in zip(pairs(g),w))
        x -= g.square().sum()/den*g
        if (i+1) % 500 == 0:
            print('independent MAP',lam,i+1,rms(g),flush=True)
    return x, i+1, rms(g)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--part', choices=['qggmrf','dnn'], default='qggmrf')
    args = p.parse_args()
    torch.set_num_threads(4)
    a = torch.load(HERE/'results/combined_reconstructions.pt', weights_only=True)
    s = json.loads((HERE/'results/combined_summary.json').read_text())
    y, truth = a['y'], a['x_true']
    ah = m.kernel_dft(a['a'], y.shape)
    result = {'part':args.part, 'runs':{}, 'torch':torch.__version__,
              'report_sha256':hashlib.sha256((HERE/'output/docx/ECE 60141 Lab 5 Report.docx').read_bytes()).hexdigest()}
    path = HERE/'results'/('audit_'+args.part+'.json')
    def save():
        path.write_text(json.dumps(result,indent=2)+'\n')
    if args.part == 'qggmrf':
        gen = torch.Generator().manual_seed(17)
        small = torch.rand((13,17),generator=gen,dtype=torch.float64,requires_grad=True)
        auto = torch.autograd.grad(prior(small),small)[0]
        result['independent_gradient_max_error'] = float((auto-stats(small.detach())[0]).abs().max())
        assert result['independent_gradient_max_error'] < 1e-10
        from lab3.lab3 import load_pgm, forward
        clean = load_pgm(HERE.parent/'lab3/kodim23.pgm')
        observation = forward(clean,a['a'])+.02*torch.randn(clean.shape,generator=torch.Generator().manual_seed(0),dtype=torch.float64)
        assert torch.equal(clean,truth) and torch.equal(observation,y)
        result['fresh_input_matches'] = True
        result['kernel_sum'] = float(a['a'].sum())
        result['kernel_minimum'] = float(a['a'].min())
        result['saved_rmse_checks'] = {k:rms(a[k]-truth) for k in s['runs']}
        for key,sp in [('Q025',.025),('Q05',.05),('Q10',.1)]:
            t = time.perf_counter()
            out,history = m.pnp_qggmrf_map(y,ah,.02,sp,.05,.02,2.,1.2,1.,100,20)
            lam = (.05/sp)**2
            direct,steps,grad = independent_map(y,ah,lam)
            pg,_ = stats(a[key])
            fg = torch.fft.ifft2(ah.conj()*(ah*torch.fft.fft2(a[key])-torch.fft.fft2(y))).real/.0004
            r = {'lambda':lam,'rmse':rms(out-truth),'replay_rms_difference':rms(out-a[key]),
                 'history_max_error':max(abs(u-v) for u,v in zip(history,s['runs'][key]['history'])),
                 'reference_rmse':rms(direct-truth),'reference_steps':steps,
                 'reference_gradient_rms':grad,'rms_difference_to_reference':rms(out-direct),
                 'correct_objective_gradient_rms':rms(fg+lam*pg),
                 'unweighted_objective_gradient_rms':rms(fg+pg),'seconds':time.perf_counter()-t}
            assert r['replay_rms_difference'] < 1e-12
            assert r['rms_difference_to_reference'] < 1e-5
            result['runs'][key] = r
            save(); print(key,r,flush=True)
        result['tied_parameter_control'] = {}
        for sp in [.025,.1]:
            out,h = m.pnp_qggmrf_map(y,ah,.02,sp,sp,.02,2.,1.2,1.,100,20)
            r = {'rmse_at_100':rms(out-truth),'rms_to_shared_MAP_at_100':rms(out-a['MAP_converged']),
                 'residual_at_100':h[-1]}
            # A lower common penalty can require more iterations; record it honestly.
            if r['rms_to_shared_MAP_at_100'] > 1e-5:
                out,h = m.pnp_qggmrf_map(y,ah,.02,sp,sp,.02,2.,1.2,1.,500,20)
                r.update(rmse_at_500=rms(out-truth),rms_to_shared_MAP_at_500=rms(out-a['MAP_converged']),residual_at_500=h[-1])
            result['tied_parameter_control'][str(sp)] = r
            save(); print('Tied',sp,r,flush=True)
    else:
        device = os.environ.get('LAB5_DNN_DEVICE','cpu')
        result['device'] = device
        print('DNN device',device,flush=True)
        for key,method,sp,sn in [('D05','ADMM',.05,.03),('D025','ADMM',.025,.03),
                                 ('D10','ADMM',.1,.03),('PG_stable','PG',.025,.015),('PG_unstable','PG',.05,.03)]:
            t = time.perf_counter()
            # Independently coded three-line loop using the equivalent FFT quotient.
            x,v,u = y.clone(),y.clone(),torch.zeros_like(y)
            h,changes = [],[]
            with torch.no_grad():
                for i in range(50):
                    old = x
                    if method == 'ADMM':
                        anchor = v-u
                        x = torch.fft.ifft2((ah.conj()*torch.fft.fft2(y)/.0004+torch.fft.fft2(anchor)/sp**2)/(ah.abs().square()/.0004+1/sp**2)).real
                        v = m.denoise_dnn(x+u,sn)
                        u += x-v
                        h.append(rms(x-v)/rms(x))
                    else:
                        g = torch.fft.ifft2(ah.conj()*(ah*torch.fft.fft2(x)-torch.fft.fft2(y))).real/.0004
                        x = m.denoise_dnn(x-sp**2*g,sn)
                        h.append(rms(x-old)/rms(x))
                    changes.append(rms(x-old)/rms(x))
                    if (i+1)%10 == 0: print(key,i+1,flush=True)
            r = {'rmse':rms(x-truth),'saved_rmse':s['runs'][key]['rmse'],
                 'replay_rms_difference':rms(x-a[key]),'relative_replay_difference':rms(x-a[key])/rms(a[key]),
                 'history_max_error':max(abs(v-w) for v,w in zip(h,s['runs'][key]['history'])),
                 'final_residual':h[-1],'final_relative_iterate_change':changes[-1],
                 'image_rms':rms(x),'seconds':time.perf_counter()-t}
            result['runs'][key] = r
            assert torch.isfinite(x).all()
            save(); print(key,r,flush=True)
    save()


if __name__ == '__main__':
    main()
