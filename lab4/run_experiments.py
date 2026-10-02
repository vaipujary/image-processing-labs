"""Run the exact six Lab 4 experiments and save reproducible measurements."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.mplconfig'))
import torch
from lab4.lab4 import load_pgm, load_kernel, forward, reconstruct, surrogate_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--threads',type=int,default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    image_path = ROOT/'lab3/kodim23.pgm'
    kernel_path = ROOT/'lab3/levin09_kernels/levin09_kernel1.txt'
    x = load_pgm(image_path)
    a = load_kernel(kernel_path)
    y = forward(x,a)+.02*torch.randn(x.shape,generator=torch.Generator().manual_seed(args.seed),dtype=torch.float64)
    arrays = {'x_true':x,'y':y,'a':a}
    summary = {'seed':args.seed,'sigma_w':.02,'p':2.,'q':1.2,'omega':1.,'iterations':50,
               'image_shape':list(x.shape),'threads':args.threads,'python':platform.python_version(),
               'torch':torch.__version__,'platform':platform.platform(),'device':'CPU',
               'input_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [image_path,kernel_path]},
               'input_rmse':float((y-x).square().mean().sqrt()),'runs':{}}
    # Warm up the same forward/autograd/step operations before timing.
    reconstruct(y[:32,:32],a,.02,.02,2.,1.2,1.,1,1.)
    runs = [('R1','Gaussian MRF',100.,.036),('R2','QGGMRF',1.,.020),
            ('R3','QGGMRF',1.,.010),('R4','QGGMRF',1.,.040),
            ('R5','Gaussian MRF',100.,.018),('R6','Gaussian MRF',100.,.072)]
    for name,prior,T,sigma_x in runs:
        print(f'{name}: {prior}, T={T:g}, sigma_x={sigma_x:g}; 50 iterations',flush=True)
        begin = time.perf_counter()
        estimate,history = reconstruct(y,a,.02,sigma_x,2.,1.2,T,50,1.)
        elapsed = time.perf_counter()-begin
        differences = [b-a for a,b in zip(history,history[1:])]
        assert all(d<0 for d in differences),f'{name}: cost did not decrease at every step'
        arrays[name] = estimate
        summary['runs'][name] = {'prior':prior,'T':T,'sigma_x':sigma_x,
                                 'final_cost':history[-1],'rmse':float((estimate-x).square().mean().sqrt()),
                                 'seconds':elapsed,'cost_history':history,'strictly_decreasing':True,
                                 'largest_cost_change':max(differences),'smallest_cost_change':min(differences),
                                 'min_pixel':float(estimate.min()),'max_pixel':float(estimate.max())}
        print(json.dumps({k:v for k,v in summary['runs'][name].items() if k!='cost_history'}),flush=True)
        (HERE/'results/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        torch.save(arrays,HERE/'results/reconstructions.pt')
    arrays['weight_sum'] = surrogate_weights(arrays['R2'],.02,2.,1.2,1.).sum(0)
    torch.save(arrays,HERE/'results/reconstructions.pt')
    print('All six runs completed; true cost strictly decreased at all 300 updates.',flush=True)


if __name__=='__main__':
    main()
