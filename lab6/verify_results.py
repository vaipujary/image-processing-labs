"""Independent numerical audit using explicit 2D densities and raw moments."""
import json
from itertools import combinations
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'

def log_joint(Y,w,mu,R):
    # Explicit 2x2 inverse and determinant, independent of the implementation's Cholesky.
    diff=Y[:,None,:]-mu[None,:,:]
    a,b,c=R[:,0,0],R[:,0,1],R[:,1,1]
    det=a*c-b*b
    q=(c*diff[:,:,0]**2-2*b*diff[:,:,0]*diff[:,:,1]+a*diff[:,:,1]**2)/det
    return np.log(w)-np.log(2*np.pi)-.5*np.log(det)-.5*q

def ll(Y,w,mu,R):
    return float(np.logaddexp.reduce(log_joint(Y,w,mu,R),axis=1).sum())

def independent_em(Y,w,mu,R):
    hist=[ll(Y,w,mu,R)]
    for _ in range(20):
        joint=log_joint(Y,w,mu,R)
        P=np.exp(joint-np.logaddexp.reduce(joint,axis=1)[:,None])
        counts=P.sum(0)
        w=counts/len(Y);mu=P.T@Y/counts[:,None]
        raw=np.einsum('nm,ni,nj->mij',P,Y,Y)/counts[:,None,None]
        R=raw-np.einsum('mi,mj->mij',mu,mu)+1e-6*np.eye(2)
        hist.append(ll(Y,w,mu,R))
    return w,mu,R,np.array(hist)

def main():
    s=json.loads((OUT/'summary.json').read_text());Y=np.load(OUT/'data.npz')['y']
    errors={}
    for name,r in s['runs'].items():
        independent=independent_em(Y,np.ones(3)/3,np.array(r['initial_means']),np.tile(np.eye(2),(3,1,1)))
        errors[name]=max(float(np.max(abs(a-np.array(r[k])))) for a,k in zip(independent,['weights','means','covariances','history']))
    assert max(errors.values())<1e-9
    score_errors={}
    for M,r in s['order']['models'].items():
        w,mu,R=[np.array(r[k]) for k in ['weights','means','covariances']]
        score=-ll(Y,w,mu,R)+.5*(6*int(M)-1)*np.log(1000)
        score_errors[M]=abs(score-r['mdl'])
    assert max(score_errors.values())<1e-9
    pair_checks=[]
    for merge in s['order']['merges']:
        r=s['order']['models'][str(merge['from_order'])]
        w,mu,R=[np.array(r[k]) for k in ['weights','means','covariances']]
        distances={}
        for l,m in combinations(range(len(w)),2):
            total=w[l]+w[m];a=w[l]/total;b=w[m]/total
            diff=mu[l]-mu[m]
            cov=a*R[l]+b*R[m]+a*b*np.outer(diff,diff)
            det=np.linalg.det(cov)
            distances[l,m]=250*(w[l]*np.log(det/np.linalg.det(R[l]))+w[m]*np.log(det/np.linalg.det(R[m])))
        pair=min(distances,key=distances.get)
        assert list(pair)==merge['pair']
        pair_checks.append(dict(order=len(w),pair=list(pair),distance_error=abs(distances[pair]-merge['distance'])))
    out=dict(independent_em_max_errors=errors,independent_mdl_errors=score_errors,merge_checks=pair_checks,
             passed=True)
    (OUT/'verification.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
