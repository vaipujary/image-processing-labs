"""Run the specified lab experiments and numerical checks on saved data."""
from itertools import permutations
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
from scipy.stats import multivariate_normal
from generate_data import TRUE_WEIGHTS, TRUE_MEANS, TRUE_COVARIANCES
from lab6 import e_step, m_step, log_likelihood, run_em, mdl, merge_closest, estimate_order

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'

def serial(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)

def matched(w, mu, R):
    # perm[t] is the estimated component corresponding to true component t.
    perm = min(permutations(range(3)), key=lambda perm:
               np.linalg.norm(mu[list(perm)] - TRUE_MEANS, axis=1).sum())
    perm = np.array(perm)
    return perm, w[perm], mu[perm], R[perm]

def pack_run(Y, x, initial, indices):
    w, mu, R, history = run_em(Y, np.full(3, 1/3), initial, np.tile(np.eye(2), (3,1,1)), 20)
    perm, mw, mm, mc = matched(w, mu, R)
    labels = e_step(Y, mw, mm, mc).argmax(axis=1)
    confusion = np.zeros((3,3), dtype=int)
    np.add.at(confusion, (x, labels), 1)
    return dict(weights=w, means=mu, covariances=R, history=history, cost=-history[-1],
                initial_indices=indices, initial_means=initial, initial_true_labels=x[indices],
                permutation=perm, matched_weights=mw, matched_means=mm, matched_covariances=mc,
                labels=labels, errors=int(np.count_nonzero(labels != x)),
                error_fraction=float(np.mean(labels != x)), confusion=confusion)

def main():
    data = np.load(OUT / 'data.npz')
    Y, x = data['y'], data['x']
    truth = (TRUE_WEIGHTS, TRUE_MEANS, TRUE_COVARIANCES)
    P = e_step(Y, *truth)
    far = np.array([[1000.,1000.]])
    far_P = e_step(far, *truth)
    raw_far = np.array([multivariate_normal.pdf(far, m, R) for m, R in zip(TRUE_MEANS,TRUE_COVARIANCES)])
    assert np.isfinite(P).all() and np.isfinite(far_P).all()
    assert np.max(abs(P.sum(1)-1)) < 1e-12 and abs(far_P.sum()-1) < 1e-12
    assert np.all(raw_far == 0)
    hard = m_step(Y, np.eye(3)[x])
    direct = (np.bincount(x)/len(Y), np.array([Y[x==k].mean(0) for k in range(3)]),
              np.array([np.cov(Y[x==k].T, bias=True)+1e-6*np.eye(2) for k in range(3)]))
    hard_errors = [float(np.max(abs(a-b))) for a,b in zip(hard,direct)]
    assert max(hard_errors) < 1e-12
    one_mu = Y.mean(0); one_cov = np.cov(Y.T,bias=True)+1e-6*np.eye(2)
    ll = log_likelihood(Y,np.ones(1),one_mu[None],one_cov[None])
    ll_direct = float(multivariate_normal.logpdf(Y,one_mu,one_cov).sum())
    assert abs(ll-ll_direct) < 1e-10
    true_run = run_em(Y,*truth,20)
    perm, tw, tm, tc = matched(*true_run[:3])
    runs = {'baseline': pack_run(Y,x,Y[:3],np.arange(3))}
    for s in range(5):
        idx = np.random.default_rng(s).choice(500,3,replace=False)
        runs[f'seed_{s}'] = pack_run(Y,x,Y[idx],idx)
    order = estimate_order(Y,9,Y[:9],20)
    histories = {'true_start':true_run[3], **{k:v['history'] for k,v in runs.items()},
                 **{f'order_{M}':v['history'] for M,v in order['models'].items()}}
    min_changes = {k:float(np.min(np.diff(v))) for k,v in histories.items()}
    assert min(min_changes.values()) >= -1e-8
    assert all(len(v)==21 for v in histories.values())
    assert len(order['trace'])==188 and len(order['merges'])==8 and order['trace'][-1]['iteration']==180
    assert sum(t['event']=='merge' for t in order['trace']) == 8
    assert all(np.linalg.eigvalsh(v['covariances']).min() >= 1e-6-1e-12 for v in order['models'].values())
    # Moment-preserving merge, checked on the initial nine-component model.
    w0=np.full(9,1/9); mu0=Y[:9].copy(); R0=np.tile(np.eye(2),(9,1,1))
    before=[a.copy() for a in (w0,mu0,R0)]
    merged=merge_closest(Y,w0,mu0,R0)
    def moments(w,mu,R):
        return w@mu,np.einsum('m,mij->ij',w,R+np.einsum('mi,mj->mij',mu,mu))
    moment_errors=[float(np.max(abs(a-b))) for a,b in zip(moments(*before),moments(*merged))]
    assert max(moment_errors)<1e-12
    assert all(np.array_equal(a,b) for a,b in zip(before,(w0,mu0,R0)))
    penalty_error=abs(mdl(Y,*truth)+log_likelihood(Y,*truth)-.5*17*np.log(1000))
    assert penalty_error<1e-10
    final3=order['models'][3]
    perm,ow,om,oc=matched(final3['weights'],final3['means'],final3['covariances'])
    out=dict(environment=dict(python=platform.python_version(),numpy=np.__version__,
                              rng='numpy.random.Generator with PCG64',seed=0),
             data=dict(counts=np.bincount(x), first_nine=Y[:9],first_nine_labels=x[:9],
                       sha256=hashlib.sha256((OUT/'data.npz').read_bytes()).hexdigest()),
             truth=dict(weights=TRUE_WEIGHTS,means=TRUE_MEANS,covariances=TRUE_COVARIANCES),
             checks=dict(e_step_max_row_error=float(np.max(abs(P.sum(1)-1))),far_point=far[0],
                         far_posterior=far_P[0],raw_far_densities=raw_far,hard_errors=hard_errors,
                         labeled_weights=hard[0],labeled_means=hard[1],labeled_covariances=hard[2],
                         single_gaussian_ll=ll,direct_gaussian_ll=ll_direct,gaussian_error=abs(ll-ll_direct),
                         min_changes=min_changes,largest_decrease=min(0.,min(min_changes.values())),
                         true_start_weights=tw,true_start_means=tm,true_start_covariances=tc,
                         true_start_weight_error=float(np.max(abs(tw-TRUE_WEIGHTS))),
                         true_start_mean_errors=np.linalg.norm(tm-TRUE_MEANS,axis=1),
                         true_start_cov_errors=np.linalg.norm(tc-TRUE_COVARIANCES,axis=(1,2)),
                         true_start_cov_entry_error=float(np.max(abs(tc-TRUE_COVARIANCES))),
                         merge_moment_errors=moment_errors,penalty_error=penalty_error),
             runs=runs,order=order,merge3=dict(weights=ow,means=om,covariances=oc,
                  cost=-final3['history'][-1],errors=int(np.count_nonzero(e_step(Y,ow,om,oc).argmax(1)!=x))))
    (OUT/'summary.json').write_text(json.dumps(out,default=serial,indent=2)+'\n')
    print(json.dumps(dict(checks=out['checks'],final_costs={k:v['cost'] for k,v in runs.items()},
                         random_details={k:dict(initial=v['initial_true_labels'],means=v['matched_means'],confusion=v['confusion']) for k,v in runs.items()},
                         best_order=order['best_order'],mdl={M:v['mdl'] for M,v in order['models'].items()},
                         merge3=out['merge3']),default=serial,indent=2))

if __name__=='__main__':
    main()
