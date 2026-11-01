"""Read-only, no-pickle loading of the frozen owned final450 Gaussian. No fitting."""
import json
import numpy as np

def load_gaussian(path,expected_classes,model_hashes):
    keys={'mx','sx','W','m','lam','classes','pool_layers','model_adapter_sha256','heads_sha256','readouts_sha256','T1','Tg','cfg_json'}
    with np.load(path,allow_pickle=False) as saved:
        assert set(saved.files)==keys
        data={k:saved[k].copy() for k in saved.files}
    assert all(not v.dtype.hasobject for v in data.values())
    for key,shape in [('mx',(12288,)),('sx',(12288,)),('W',(12288,128)),('m',(18,128)),('lam',(128,))]:
        assert data[key].shape==shape and data[key].dtype==np.float64 and np.isfinite(data[key]).all()
    assert (data['sx']>0).all() and (data['lam']>=1e-4).all()
    assert data['classes'].shape==(18,) and data['classes'].dtype.kind=='U' and len(set(data['classes']))==18
    np.testing.assert_array_equal(data['classes'],np.asarray(expected_classes,dtype=str))
    np.testing.assert_array_equal(data['pool_layers'],[6,12,18,24,30,36])
    for key in ['model_adapter_sha256','heads_sha256','readouts_sha256']:
        assert data[key].shape==() and str(data[key])==model_hashes[key]
    cfg=json.loads(str(data['cfg_json']))
    assert cfg==dict(red='wpca',d=128,se_shrink='lw',sp_shrink='lw',lam_floor=1e-4,n_exact=3,inj=True,iters=5,repel=True,sp_scale=1.0)
    assert float(data['T1'])==float(data['Tg'])==3.0
    prm={k:data[k] for k in ['mx','sx','W','m','lam']}
    prm.update(cfg=cfg,T1=3.0,Tg=3.0)
    return prm,data['classes']
