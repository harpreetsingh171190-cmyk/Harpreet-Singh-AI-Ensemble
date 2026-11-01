"""Native NumPy FP32 routing plus a frozen FP64 Gaussian; no fitting."""
import numpy as np
from center_readout import source_logsoftmax,add_logprob
from gaussian_core import scores

def smol_score(Z,ro,groups,eligible,prm,gaussian_fn=scores):
    assert Z.dtype==np.float32 and Z.ndim==2 and Z.shape[1]==12288 and np.isfinite(Z).all()
    n=len(Z);groups=np.asarray(groups);eligible=np.asarray(eligible)
    assert groups.shape==eligible.shape==(n,) and groups.dtype.kind in 'iu' and eligible.dtype==bool
    counts=np.unique(groups,return_inverse=True,return_counts=True)
    sizes=counts[2][counts[1]]
    np.testing.assert_array_equal(eligible,(sizes>=2)&(sizes<=18))
    for route in ('plain','center'):
        assert ro[route+'_coef'].dtype==ro[route+'_intercept'].dtype==np.float32
        assert ro[route+'_coef'].shape==(18,12288) and ro[route+'_intercept'].shape==(18,)
    # Exactly the validated CPU plain-head order; clipping belongs ONLY here.
    lg=Z@ro['plain_coef'].T+ro['plain_intercept']
    lg-=lg.max(1,keepdims=True)
    p=np.exp(lg)
    plain=add_logprob(p/p.sum(1,keepdims=True))
    routed=plain.copy()
    local=np.flatnonzero(eligible)
    if len(local):
        # One query pool; all members of every eligible original group are present.
        X=np.asarray(Z[local],dtype=np.float32).copy()
        local_groups=groups[local]
        for group in np.unique(local_groups):
            i=local_groups==group
            X[i]-=X[i].mean(0)
        lc=X@ro['center_coef'].T+ro['center_intercept']
        routed[local]=source_logsoftmax(lc)
    assert routed.dtype==np.float32 and np.isfinite(routed).all()
    result=routed.astype(np.float64)
    if len(local):
        group_labels=np.unique(groups[local],return_inverse=True)[1]
        gaussian=gaussian_fn(Z[local],group_labels,prm)
        assert gaussian.dtype==np.float64 and gaussian.shape==(len(local),18) and np.isfinite(gaussian).all()
        np.testing.assert_allclose(np.exp(gaussian).sum(1),1,rtol=0,atol=1e-10)
        result[local]+=0.10*gaussian
    assert result.dtype==np.float64 and np.isfinite(result).all()
    # No member renormalization, exponentiation or clipping after Gaussian addition.
    return result,routed
