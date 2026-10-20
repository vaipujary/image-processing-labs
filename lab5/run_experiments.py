"""Reproduce Lab 5, checkpointing each finished experiment. Run from any directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
if (HERE/'.deps').exists(): sys.path.insert(0,str(HERE/'.deps'))
os.environ.setdefault('TORCH_HOME',str(HERE/'models'))
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.mplconfig'))
import torch
import lab5.lab5 as m
from lab3.lab3 import load_pgm, forward, adjoint


def direct_map(y, ah, iterations):
    """Lab 4 surrogate descent with mathematically equivalent FFT data operations."""
    x=y.clone(); history=[]
    yh=torch.fft.fft2(y)
    for i in range(iterations):
        pg,c=m._prior_stats(x,.02,2.,1.2,1.)
        g=torch.fft.ifft2(ah.conj()*(ah*torch.fft.fft2(x)-yh)).real/.02**2+pg
        ag=torch.fft.ifft2(ah*torch.fft.fft2(g)).real
        den=ag.square().sum()/.02**2
        for (l,r,_),w in zip(m._pairs(),c): den+=(w*(g[l]-g[r]).square()).sum()
        x-=g.square().sum()/den*g
        if i==49: first50=x.clone()
        if (i+1)%100==0: print(f'direct MAP {i+1}/{iterations}',flush=True)
    return x,first50


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--part',choices=['qggmrf','dnn','all'],default='all')
    args=parser.parse_args();torch.set_num_threads(4)
    result=HERE/'results';result.mkdir(exist_ok=True)
    prefix='dnn_' if args.part=='dnn' else ''
    summary_path=result/(prefix+'summary.json');array_path=result/(prefix+'reconstructions.pt')
    if summary_path.exists():
        s=json.loads(summary_path.read_text());arr=torch.load(array_path,weights_only=True)
    else:
        image=ROOT/'lab3/kodim23.pgm';kernel=ROOT/'lab3/levin09_kernels/levin09_kernel1.txt'
        x=load_pgm(image)
        a=torch.tensor([[float(v) for v in line.split()] for line in kernel.read_text().splitlines() if line.strip()],dtype=torch.float64)
        y=forward(x,a)+.02*torch.randn(x.shape,generator=torch.Generator().manual_seed(0),dtype=torch.float64)
        arr={'x_true':x,'a':a,'y':y}
        s={'seed':0,'torch':torch.__version__,'threads':4,'shape':list(x.shape),'crop':[160,288,384,512],
           'sigma_w':.02,'sigma_x':.02,'p':2.,'q':1.2,'T':1.,'runs':{},
           'input_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [image,kernel]}}
    ah=m.kernel_dft(arr['a'],arr['y'].shape);y=arr['y'];truth=arr['x_true']
    def save():
        torch.save(arr,array_path);summary_path.write_text(json.dumps(s,indent=2)+'\n')
    def record(name,out,history,elapsed,**params):
        arr[name]=out
        s['runs'][name]={'rmse':float((out-truth).square().mean().sqrt()),'history':history,'seconds':elapsed,
                        'minimum':float(out.min()),'maximum':float(out.max()),**params}
        print(name,{k:v for k,v in s['runs'][name].items() if k not in ['history','output_rms_history']},flush=True);save()
    s['input_rmse']=float((y-truth).square().mean().sqrt())
    u=torch.randn(y.shape,generator=torch.Generator().manual_seed(1),dtype=torch.float64)
    s['fft_forward_max_error']=float((torch.fft.ifft2(ah*torch.fft.fft2(u)).real-forward(u,arr['a'])).abs().max())
    s['fft_adjoint_max_error']=float((torch.fft.ifft2(ah.conj()*torch.fft.fft2(u)).real-adjoint(u,arr['a'])).abs().max())
    for name,sp in [('F_small',.01),('F_large',1.)]:
        out=m.prox_data(y,y,ah,.02,sp)
        record(name,out,[],0,sigma_p=sp,data_residual_rms=float((forward(out,arr['a'])-y).square().mean().sqrt()))
    s['y_data_residual_rms']=float((forward(y,arr['a'])-y).square().mean().sqrt())
    old=torch.load(ROOT/'lab4/results/reconstructions.pt',weights_only=True)
    s['lab4_observation_identical']=torch.equal(old['y'],y)
    if args.part in ['qggmrf','all']:
        if 'MAP_converged' not in arr:
            t=time.perf_counter();out,out50=direct_map(y,ah,1000)
            s['map50_vs_saved_lab4_max_error']=float((out50-old['R2']).abs().max())
            record('MAP50',out50,[],0,iterations=50)
            record('MAP_converged',out,[],time.perf_counter()-t,iterations=1000)
        for name,sp in [('Q05',.05),('Q025',.025),('Q10',.1)]:
            if name in arr:continue
            print(f'{name}: 100 outer by 20 inner steps',flush=True);t=time.perf_counter()
            out,h=m.pnp_qggmrf_map(y,ah,.02,sp,.05,.02,2.,1.2,1.,100,20)
            record(name,out,h,time.perf_counter()-t,sigma_p=sp,sigma_n=.05,iterations=100,inner_iters=20)
        if 'Q05_inner10' not in arr:
            print('Testing halved inner iterations for D9 question 2',flush=True);t=time.perf_counter()
            out,h=m.pnp_qggmrf_map(y,ah,.02,.05,.05,.02,2.,1.2,1.,100,10)
            record('Q05_inner10',out,h,time.perf_counter()-t,sigma_p=.05,sigma_n=.05,iterations=100,inner_iters=10)
        s['qggmrf_vs_map50_rms']=float((arr['Q05']-arr['MAP50']).square().mean().sqrt())
        s['qggmrf_vs_converged_map_rms']=float((arr['Q05']-arr['MAP_converged']).square().mean().sqrt())
        save()
    if args.part in ['dnn','all']:
        import deepinv
        s['deepinv']=deepinv.__version__
        s['dnn_device']=os.environ.get('LAB5_DNN_DEVICE','cpu')
        m._drunet_model()
        for name,method,sp,sn in [('D05','ADMM',.05,.03),('D025','ADMM',.025,.03),('D10','ADMM',.1,.03),
                                   ('PG_unstable','PG',.05,.03),('PG_stable','PG',.025,.015)]:
            if name in arr:continue
            count=[0];norms=[]
            def denoiser(z):
                out=m.denoise_dnn(z,sn);count[0]+=1;norms.append(float(out.square().mean().sqrt()))
                if count[0]%5==0: print(f'{name} {count[0]}/50, RMS {norms[-1]:.6g}',flush=True)
                return out
            t=time.perf_counter()
            method_fn=m.pnp_admm if method=='ADMM' else m.pnp_prox_grad
            out,h=method_fn(y,ah,denoiser,.02,sp,50)
            record(name,out,h,time.perf_counter()-t,sigma_p=sp,sigma_n=sn,iterations=50,output_rms_history=norms)
        save()


if __name__=='__main__': main()
