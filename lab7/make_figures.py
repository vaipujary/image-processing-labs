"""Create the report's scientific figures as PNGs only, with no PDF output."""
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/lab7-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

HERE=Path(__file__).resolve().parent
RESULTS=HERE/'results'
plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':10,
                     'legend.fontsize':9,'savefig.dpi':220,'font.family':'DejaVu Sans',
                     'axes.spines.top':False,'axes.spines.right':False})
COLORS=['#135B8E','#C77611','#20815A']


def save(fig,name):
    fig.savefig(RESULTS/name,facecolor='white')
    plt.close(fig)


def picture(ax,tensor,title,vmin,vmax):
    im=ax.imshow(tensor.numpy(),cmap='gray',vmin=vmin,vmax=vmax,interpolation='nearest')
    ax.set_title(title,pad=6)
    ax.set_xticks([]);ax.set_yticks([])
    for spine in ax.spines.values():spine.set_visible(False)
    return im


def main():
    images=torch.load(RESULTS/'images.pt',weights_only=True)
    s=json.loads((RESULTS/'summary.json').read_text())
    h=json.loads((RESULTS/'histories.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(7.1,2.72))
    for ax,sigma in zip(axes,(.01,.05,.2)):
        x=images[f'gmrf_{sigma}']; x=x-x.mean()
        lo,hi=s['gmrf'][str(sigma)]['display_range']
        picture(ax,x,fr'$\sigma_x={sigma}$',lo,hi)
        ax.set_xlabel(f'Range [{lo:.4f}, {hi:.4f}]',fontsize=9)
    fig.subplots_adjust(left=.015,right=.985,top=.89,bottom=.14,wspace=.12)
    save(fig,'D1_gmrf_samples.png')

    fig,axes=plt.subplots(1,2,figsize=(7.1,2.35))
    for ax,key,title,color in zip(axes,('prior_mean','prior_std'),
                                  ('Image mean','Centered image standard deviation'),COLORS):
        ax.plot(range(1,501),h[key],color=color,lw=1.1)
        ax.set(xlabel='Sweep',ylabel=title,xlim=(1,500))
        ax.grid(alpha=.2)
    fig.tight_layout(pad=.7,w_pad=1.5)
    save(fig,'D2_prior_traces.png')

    fig,axes=plt.subplots(1,3,figsize=(7.1,2.95))
    for ax,width in zip(axes,(.002,.02,.2)):
        x=images[f'qggmrf_{width}'];x=x-x.mean()
        info=s['qggmrf'][str(width)];lo,hi=info['display_range']
        picture(ax,x,fr'$\sigma_{{prop}}={width}$',lo,hi)
        ax.set_xlabel(f"Acceptance {info['acceptance_all']:.2%}\nRange [{lo:.3f}, {hi:.3f}]",fontsize=9)
    fig.subplots_adjust(left=.015,right=.985,top=.89,bottom=.2,wspace=.12)
    save(fig,'D3_qggmrf_samples.png')

    fig,ax=plt.subplots(figsize=(7.1,3.05))
    for width,color in zip((.002,.02,.2),COLORS):
        ax.plot(range(1,301),h[f'qggmrf_{width}']['energy'],color=color,
                label=fr'$\sigma_{{prop}}={width}$',lw=1.25)
    ax.set(xlabel='Sweep',ylabel='QGGMRF energy',xlim=(1,300),ylim=(0,56000))
    ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    ax.legend(loc='lower right',frameon=False);ax.grid(alpha=.2)
    fig.tight_layout(pad=.7)
    save(fig,'D4_proposal_energy.png')

    fig,axes=plt.subplots(1,2,figsize=(5.7,2.55))
    best=s['best_proposal']
    a=images['gmrf_0.05'];a=a-a.mean()
    b=images[f'qggmrf_{best}'];b=b-b.mean()
    lo=min(float(a.min()),float(b.min()));hi=max(float(a.max()),float(b.max()))
    for ax,x,title in zip(axes,(a,b),('GMRF after 200 sweeps','QGGMRF after 300 sweeps')):
        picture(ax,x,title,lo,hi)
    fig.subplots_adjust(left=.01,right=.99,top=.88,bottom=.05,wspace=.05)
    save(fig,'D5_prior_comparison.png')
    s['comparison']={'display_range':[lo,hi],
                     'gmrf_difference_std':float((a[:,1:]-a[:,:-1]).std(correction=0)),
                     'qggmrf_difference_std':float((b[:,1:]-b[:,:-1]).std(correction=0))}

    fig,axes=plt.subplots(2,3,figsize=(7.1,4.25))
    entries=[('clean','Clean crop'),('noisy','Noisy measurement'),
             ('single_sample','Single posterior sample'),('posterior_mean','Posterior mean'),('map','MAP estimate')]
    # Put the two inputs on the top row and the three estimates on the bottom.
    positions=[axes[0,0],axes[0,1],axes[1,0],axes[1,1],axes[1,2]]
    for ax,(key,title) in zip(positions,entries):
        picture(ax,images[key],title,0,1)
    axes[0,2].axis('off')
    fig.subplots_adjust(left=.01,right=.99,top=.94,bottom=.015,wspace=.05,hspace=.22)
    save(fig,'D6_denoising.png')

    fig,ax=plt.subplots(figsize=(3.9,2.9))
    lo,hi=s['posterior']['std_range']
    im=picture(ax,images['posterior_std'],'',lo,hi)
    bar=fig.colorbar(im,ax=ax,fraction=.05,pad=.04)
    bar.set_label('Posterior standard deviation')
    fig.tight_layout(pad=.5)
    save(fig,'D7_uncertainty.png')

    fig,axes=plt.subplots(1,2,figsize=(7.1,2.9))
    for label,color in zip(('y','zero'),COLORS):
        values=h[f'start_{label}']['energy']
        legend='Start at y' if label=='y' else 'Start at zero'
        axes[0].semilogy(range(1,21),values[:20],color=color,label=legend,lw=1.3)
        axes[1].plot(range(1,501),values,color=color,label=legend,lw=.8,alpha=.9)
    axes[0].set(xlabel='Sweep',ylabel='Posterior energy',title='Initial approach',xlim=(1,20))
    axes[0].legend(frameon=False)
    axes[1].set(xlabel='Sweep',ylabel='Posterior energy',title='Stationary band detail',xlim=(1,500),ylim=(55800,57900))
    axes[1].axvline(100,color='black',lw=.8,ls='--',label='Burn-in ends')
    axes[1].legend(frameon=False,loc='lower right')
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout(pad=.7,w_pad=1)
    save(fig,'D8_burnin.png')
    (RESULTS/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
    print('Created D1–D8 PNG figures; no PDF output.')


if __name__=='__main__':main()
