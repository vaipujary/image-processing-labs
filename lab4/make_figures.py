"""Generate D1, D3, D5-D9 figures from measured, unclipped Lab 4 arrays."""
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
from lab4.lab4 import rho, rho_prime

RESULTS = HERE/'results'
CROP = (160,288,384,512)
ROW = 232
COLORS = ['#202020','#386cb0','#c85c26']


def save(fig,name):
    fig.savefig(RESULTS/name,dpi=220,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.constrained_layout.use':True,'axes.titleweight':'semibold'})
    a = torch.load(RESULTS/'reconstructions.pt',weights_only=True)
    s = json.loads((RESULTS/'summary.json').read_text())
    d = torch.linspace(-.3,.3,1601,dtype=torch.float64)
    fig,ax = plt.subplots(1,2,figsize=(8,3))
    for axis,quad,nonquad,title,ylabel in zip(ax,[d.square()/(2*.2**2),d/.2**2],
             [rho(d,.2,2.,1.2,1.),rho_prime(d,.2,2.,1.2,1.)],
             ['Potential','Influence function'],[r'$\rho(\Delta)$',r"$\rho'(\Delta)$"]):
        axis.plot(d,quad,label='Quadratic',color=COLORS[1],lw=2)
        axis.plot(d,nonquad,label='QGGMRF',color=COLORS[2],lw=2)
        axis.set(xlabel=r'$\Delta$',ylabel=ylabel,title=title,xlim=(-.3,.3))
        axis.grid(alpha=.2);axis.legend(fontsize=9)
    save(fig,'D1_potential_influence.png')
    fig,ax=plt.subplots(figsize=(7.4,3.2))
    ax.plot(d,rho(d,.2,2.,1.2,1.),color=COLORS[0],lw=2,label='QGGMRF potential')
    for t,color in zip([.02,.15],COLORS[1:]):
        t=torch.tensor(t,dtype=torch.float64)
        surrogate=rho_prime(t,.2,2.,1.2,1.)/(2*t)*(d.square()-t.square())+rho(t,.2,2.,1.2,1.)
        ax.plot(d,surrogate,color=color,label=rf"Surrogate at $\Delta'={t.item():.2f}$",lw=1.8)
        ax.scatter([-t,t],[rho(t,.2,2.,1.2,1.)]*2,color=color,zorder=5,s=35)
    ax.set(xlabel=r'$\Delta$',ylabel='Potential and touching surrogate',xlim=(-.3,.3))
    ax.legend(fontsize=9);ax.grid(alpha=.2)
    save(fig,'D3_surrogates.png')
    h=s['runs']['R2']['cost_history']
    fig,ax=plt.subplots(figsize=(7.4,2.6))
    ax.plot(range(51),h,color=COLORS[2],lw=2)
    ax.set(xlabel='Iteration',ylabel='True MAP cost',xlim=(0,50))
    ax.grid(alpha=.2)
    save(fig,'D5_true_cost.png')
    r0,r1,c0,c1=CROP
    keys=['x_true','y','R1','R2']
    labels=['Original','Blurred and noisy','R1 Gaussian MRF','R2 QGGMRF']
    fig,axs=plt.subplots(2,2,figsize=(8,5.7))
    for ax,key,label in zip(axs.flat,keys,labels):
        ax.imshow(a[key],cmap='gray',vmin=0,vmax=1)
        ax.add_patch(Rectangle((c0-.5,r0-.5),128,128,fill=False,edgecolor='#ffa640',lw=1))
        ax.set_title(label);ax.axis('off')
    save(fig,'D6_full_images.png')
    fig,axs=plt.subplots(1,4,figsize=(8,2.15))
    for ax,key,label in zip(axs,keys,labels):
        ax.imshow(a[key][r0:r1,c0:c1],cmap='gray',vmin=0,vmax=1,interpolation='nearest')
        ax.set_title(label,fontsize=10);ax.axis('off')
    save(fig,'D6_crops.png')
    fig,axs=plt.subplots(1,2,figsize=(8,3.1),gridspec_kw={'width_ratios':[1.65,1]})
    for key,label,color in zip(['x_true','R1','R2'],['Original','R1 Gaussian MRF','R2 QGGMRF'],COLORS):
        axs[0].plot(range(c0,c1),a[key][ROW,c0:c1],color=color,label=label,lw=1.6)
        axs[1].plot(range(c0,c1),a[key][ROW,c0:c1],color=color,lw=1.7)
    axs[0].set(xlabel='Column index',ylabel='Pixel value',xlim=(c0,c1-1),title=f'Row {ROW} across the crop')
    axs[1].set(xlabel='Column index',xlim=(438,460),title='Detail at the beak edge')
    axs[0].legend(fontsize=8)
    for ax in axs: ax.grid(alpha=.2)
    save(fig,'D7_edge_profile.png')
    fig,axs=plt.subplots(2,3,figsize=(8,5.25))
    for ax,key in zip(axs.flat,['R3','R2','R4','R5','R1','R6']):
        run=s['runs'][key]
        ax.imshow(a[key][r0:r1,c0:c1],cmap='gray',vmin=0,vmax=1,interpolation='nearest')
        ax.set_title(f"{key}  {run['prior']}\n"+rf"$\sigma_x={run['sigma_x']:.3f}$"+f"   RMSE {run['rmse']:.5f}",fontsize=10)
        ax.axis('off')
    save(fig,'D8_scale_study.png')
    fig,ax=plt.subplots(figsize=(7.6,4.5))
    weights=a['weight_sum']
    im=ax.imshow(weights,cmap='gray',vmin=0,vmax=1250)
    ax.set(xlabel='Column index',ylabel='Row index')
    cb=fig.colorbar(im,ax=ax,fraction=.035,pad=.03)
    cb.set_label('Sum of surrogate weights')
    save(fig,'D9_weight_sum.png')
    edge_cols=(455,456)
    s['figures']={'crop_zero_based_half_open':list(CROP),'profile_row':ROW,
                  'weight_display_range':[0,1250],'weight_actual_range':[float(weights.min()),float(weights.max())],
                  'edge_columns':list(edge_cols),'edge_values':{key:[float(a[key][ROW,c]) for c in edge_cols] for key in ['x_true','R1','R2']}}
    (RESULTS/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
    print(json.dumps(s['figures'],indent=2))


if __name__=='__main__':
    main()
