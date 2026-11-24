"""Produce every required lab figure from saved numerical results."""
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/lab6-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
COLORS=['#0072B2','#D55E00','#009E73']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
                     'axes.labelsize':10,'legend.fontsize':8.5,'figure.dpi':160,
                     'axes.spines.top':False,'axes.spines.right':False})

def save(fig,name):
    fig.savefig(OUT/name,dpi=220,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def scatter(ax,Y,labels,means=None,title=None,legend=True):
    for k in range(3):
        pts=Y[labels==k]
        ax.scatter(pts[:,0],pts[:,1],s=12,c=COLORS[k],alpha=.7,edgecolors='none',label=f'Class {k}')
    if means is not None:
        ax.scatter(means[:,0],means[:,1],marker='X',s=100,c=COLORS,edgecolors='black',linewidths=.9,label='Estimated means',zorder=5)
    ax.set(xlabel='$y_1$',ylabel='$y_2$',xlim=(-5.8,8.8),ylim=(-5.5,5.8))
    ax.set_aspect('equal',adjustable='box')
    if title: ax.set_title(title,pad=10)
    ax.grid(alpha=.15)
    if legend: ax.legend(loc='upper left',framealpha=.95,markerscale=1.25)

def main():
    d=np.load(OUT/'data.npz');Y,x=d['y'],d['x']
    s=json.loads((OUT/'summary.json').read_text());runs=s['runs']
    fig,ax=plt.subplots(figsize=(6.8,4.8),layout='constrained')
    scatter(ax,Y,x,title='Generated data with true labels')
    save(fig,'D1_true_data.png')
    fig,ax=plt.subplots(figsize=(6.8,3.65),layout='constrained')
    cost=-np.array(runs['baseline']['history'][1:])
    ax.plot(np.arange(1,21),cost,'o-',color=COLORS[0],ms=3)
    ax.set(xlabel='EM iteration',ylabel='Negative log likelihood',xticks=[1,5,10,15,20],title='First three samples initialization')
    ax.grid(alpha=.2)
    ax.annotate(f'Iteration 20: {cost[-1]:.3f}',(20,cost[-1]),xytext=(.54,.45),textcoords='axes fraction',arrowprops={'arrowstyle':'->','color':'#555555'})
    save(fig,'D5_baseline_cost.png')
    fig,ax=plt.subplots(figsize=(6.8,4.8),layout='constrained')
    r=runs['baseline'];scatter(ax,Y,np.array(r['labels']),np.array(r['matched_means']),title=f'M = 3 after 20 iterations | cost {r["cost"]:.3f}')
    save(fig,'D7_baseline_labels.png')
    fig,ax=plt.subplots(figsize=(6.8,3.8),layout='constrained')
    for k,r in runs.items():
        label='First three samples' if k=='baseline' else k.replace('_',' ')
        ax.plot(range(1,21),-np.array(r['history'][1:]),label=label,lw=2 if k=='baseline' else 1.4,ls='--' if k=='baseline' else '-')
    ax.set(xlabel='EM iteration',ylabel='Negative log likelihood',xticks=[1,5,10,15,20])
    ax.grid(alpha=.2);ax.legend(ncol=2)
    save(fig,'D8_all_costs.png')
    for seed in range(5):
        r=runs[f'seed_{seed}']
        fig,ax=plt.subplots(figsize=(6.8,4.25),layout='constrained')
        scatter(ax,Y,np.array(r['labels']),np.array(r['matched_means']),title=f'Seed {seed} | cost {r["cost"]:.3f} | disagreements {r["errors"]}/500')
        save(fig,f'D9_seed_{seed}.png')
    # Multipanels retain an individual scatter plot for every random start.
    for block,seeds in [('a',[0,1,2]),('b',[3,4])]:
        fig,axes=plt.subplots(len(seeds),1,figsize=(6.8,3.35*len(seeds)),layout='constrained')
        for ax,seed in zip(np.atleast_1d(axes),seeds):
            r=runs[f'seed_{seed}']
            scatter(ax,Y,np.array(r['labels']),np.array(r['matched_means']),title=f'Seed {seed} | cost {r["cost"]:.3f} | disagreements {r["errors"]}/500')
        save(fig,f'D9_group_{block}.png')
    fig,axes=plt.subplots(3,2,figsize=(7.0,8.5))
    fig.subplots_adjust(left=.08,right=.99,bottom=.06,top=.95,hspace=.46,wspace=.27)
    for seed,ax in enumerate(axes.flat):
        if seed==5:
            ax.axis('off')
            ax.text(.06,.77,'Colors follow the one-to-one\nmean matching to true classes.\n\nX markers show estimated means.\nAll panels use equal axis scaling.\n\nSeed 2 splits the negative cluster\nand combines both positive clusters.\nThe other four starts recover the\nthree main cluster regions.',transform=ax.transAxes,fontsize=9,linespacing=1.6,va='top')
            continue
        r=runs[f'seed_{seed}']
        scatter(ax,Y,np.array(r['labels']),np.array(r['matched_means']),title=f'Seed {seed} | cost {r["cost"]:.3f}',legend=False)
        ax.title.set_fontsize(10)
        ax.tick_params(labelsize=8)
        ax.set_xticks([-5,0,5]);ax.set_yticks([-4,0,4])
    save(fig,'D9_five_starts.png')
    trace=s['order']['trace']
    fig,ax=plt.subplots(figsize=(6.8,4.05),layout='constrained')
    ax.plot([t['iteration'] for t in trace],[t['mdl'] for t in trace],color=COLORS[0],lw=1.6)
    for t in trace:
        if t['event']=='merge':
            ax.axvline(t['iteration'],color='#888888',ls=':',lw=.8)
            ax.scatter(t['iteration'],t['mdl'],marker='D',s=22,c=COLORS[1],zorder=5)
    for i,M in enumerate(range(9,0,-1)):
        ax.text(i*20+10,.98,f'M={M}',transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=8)
    ax.plot([],[],marker='D',c=COLORS[1],ls='none',label='Immediately after merge')
    ax.set(xlabel='Cumulative EM iterations',ylabel='MDL',xlim=(0,182),xticks=np.arange(0,181,20))
    ax.grid(axis='y',alpha=.2);ax.legend(loc='center right',fontsize=8)
    save(fig,'D11_mdl_trace.png')
    models=s['order']['models']; Ms=np.arange(1,10); scores=np.array([models[str(M)]['mdl'] for M in Ms]);best=s['order']['best_order']
    fig,ax=plt.subplots(figsize=(6.8,3.8),layout='constrained')
    ax.plot(Ms,scores,'o-',color=COLORS[0]);ax.scatter(best,scores[best-1],s=95,facecolors='none',edgecolors=COLORS[1],lw=2,zorder=5)
    ax.annotate(f'Selected M = {best}\nMDL = {scores[best-1]:.3f}',(best,scores[best-1]),xytext=(4.2,scores[best-1]+90),arrowprops={'arrowstyle':'->'})
    ax.set(xlabel='Number of components M',ylabel='Final MDL after 20 EM iterations',xticks=Ms);ax.grid(alpha=.2)
    save(fig,'D12_mdl_order.png')
    print('Saved all deliverable figures')

if __name__=='__main__':main()
