"""Merge independently checkpointed experiments and generate report figures."""
import json
import os
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'.deps'))
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.mplconfig'))
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
R=HERE/'results'


def main():
    s=json.loads((R/'summary.json').read_text());a=torch.load(R/'reconstructions.pt',weights_only=True)
    if (R/'dnn_summary.json').exists():
        ds=json.loads((R/'dnn_summary.json').read_text());da=torch.load(R/'dnn_reconstructions.pt',weights_only=True)
        assert torch.equal(a['y'],da['y'])
        s['runs'].update(ds['runs']);s['deepinv']=ds['deepinv'];s['dnn_device']=ds['dnn_device'];a.update(da)
    (R/'combined_summary.json').write_text(json.dumps(s,indent=2)+'\n')
    torch.save(a,R/'combined_reconstructions.pt')
    plt.rcParams.update({'font.size':11,'axes.titlesize':11,'figure.dpi':160,'savefig.dpi':220})
    def panels(filename,keys,labels,crop=False):
        fig,axes=plt.subplots(1,len(keys),figsize=(10.5,3.6 if crop else 2.9),squeeze=False)
        for ax,key,label in zip(axes[0],keys,labels):
            im=a[key];im=im[160:288,384:512] if crop else im
            ax.imshow(im.numpy(),cmap='gray',vmin=0,vmax=1,interpolation='nearest');ax.axis('off')
            rmse=s['input_rmse'] if key=='y' else s['runs'].get(key,{}).get('rmse')
            title=label+(f'\nRMSE {rmse:.5f}' if rmse is not None else '')
            ax.set_title(title)
            if not crop:ax.add_patch(Rectangle((384,160),128,128,fill=False,edgecolor='#df8a24',linewidth=.9))
        fig.tight_layout(pad=.3);fig.savefig(R/filename,bbox_inches='tight');plt.close(fig)
    def history(filename,keys,labels,ylabel,second=None):
        fig,ax=plt.subplots(figsize=(8,3))
        for k,label in zip(keys,labels):
            h=s['runs'][k]['history'];ax.semilogy(range(1,len(h)+1),h,label=label)
        ax.set(xlabel='Iteration',ylabel=ylabel);ax.grid(True,which='both',alpha=.2);ax.legend(fontsize=9)
        if second:
            twin=ax.twinx();h=s['runs'][second]['output_rms_history']
            twin.semilogy(range(1,len(h)+1),h,color='#ad4333',linestyle=':',label='Image RMS')
            twin.set_ylabel('Image RMS',color='#ad4333');twin.legend(loc='center right',fontsize=9)
        fig.tight_layout();fig.savefig(R/filename,bbox_inches='tight');plt.close(fig)
    panels('D1_forward.png',['y','F_small','F_large'],['Blurred noisy data y','F(y), σp = 0.01','F(y), σp = 1.0'])
    panels('D3_full.png',['Q05','MAP_converged'],['QGGMRF PnP','Direct MAP, 1000 steps'])
    panels('D3_crops.png',['Q05','MAP_converged','MAP50'],['QGGMRF PnP','Direct MAP, 1000 steps','Lab 4 baseline, 50 steps'],True)
    panels('D4_sweep.png',['Q025','Q05','Q10'],['σp = 0.025, weight = 4','σp = 0.05, weight = 1','σp = 0.10, weight = 0.25'],True)
    history('Q_inner_comparison.png',['Q05','Q05_inner10'],['20 inner steps','10 inner steps'],'Relative equilibrium residual')
    panels('D5_full.png',['y','D05','MAP50'],['Blurred noisy data y','DRUNet PnP ADMM','Lab 4 MAP, 50 steps'])
    panels('D5_crops.png',['x_true','y','D05','MAP50'],['Ground truth','Blurred noisy data','DRUNet PnP ADMM','Lab 4 MAP'],True)
    history('D5_convergence.png',['D05'],['DRUNet PnP ADMM'],'||x − v|| / ||x||')
    panels('D6_sweep.png',['D025','D05','D10'],['σp = 0.025','σp = 0.05','σp = 0.10'],True)
    history('D7_unstable.png',['PG_unstable'],['PnP PG relative change'],'||x − x previous|| / ||x||',second='PG_unstable')
    panels('D8_full.png',['PG_stable','D05'],['PnP PG, rescaled','PnP ADMM, D5'])
    panels('D8_crops.png',['PG_stable','D05'],['PnP PG, rescaled','PnP ADMM, D5'],True)
    history('D8_convergence.png',['PG_stable','D05'],['PnP PG: relative iterate change','PnP ADMM: equilibrium residual'],'Relative diagnostic')
    print('All report figures generated.')


if __name__=='__main__': main()
