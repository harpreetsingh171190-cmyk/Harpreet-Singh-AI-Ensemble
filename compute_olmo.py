"""Frozen native24 OLMo2 plain readout; one query pass, no fitting."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
import pandas as pd

from olmo_component import OLMoFinalPlain

here=Path(__file__).resolve().parent;config=json.loads((here/'candidate-config.json').read_text())
assert config['source_ready'] and config['OLMo_plain_raw_coefficient']==.03
spec=json.loads((here/'olmo-spec.json').read_text())
assert spec['inference_authorized'] and spec['root_production_admitted'] and spec['public_asset_release_audited']
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assets=[p for p in Path('/kaggle/input').rglob('asset_manifest.json') if sha(p)==spec['compact_manifest_sha256']]
base_config=next(r for r in spec['base_files'] if r['name']=='config.json')
bases=[p for p in Path('/kaggle/input').rglob('config.json') if sha(p)==base_config['sha256']]

tests=[p/'test.csv' for p in (Path('/kaggle/input/competitions/name-that-llm'),Path('/kaggle/input/name-that-llm')) if (p/'test.csv').is_file()]
if os.environ.get('NTL_QUERY_CSV'): tests=[Path(os.environ['NTL_QUERY_CSV'])]
assert len(assets)==len(bases)==len(tests)==1
cache=Path(os.environ['NTL_CANDIDATE_TEMP']).resolve();assert cache.is_relative_to(Path('/kaggle/temp').resolve())
assert not (cache/'olmo_plain.npz').exists(),'No automatic repeat/reuse'
query=pd.read_csv(tests[0],usecols=['id','text'],dtype=str)
query['text']=query['text'].fillna('').astype(str)
ids=query.id.to_numpy(dtype=str);labels=np.asarray(spec['classes'],dtype=str)
np.testing.assert_array_equal(query.id.to_numpy(dtype=str),ids)
assert os.environ['CUDA_VISIBLE_DEVICES'].startswith('GPU-')
started=time.monotonic()
model=OLMoFinalPlain(here/'olmo-spec.json',assets[0].parent,bases[0].parent,None)
np.testing.assert_array_equal(model.readouts['classes'],labels)
from worker_barrier import ready_and_wait
ready_and_wait('olmo')
import torch
assert torch.cuda.device_count()==1
begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
begin.record();result=model.predict_plain_logp(ids,query.text.tolist());end.record();torch.cuda.synchronize()
event_ms=float(begin.elapsed_time(end));assert event_ms>0
np.testing.assert_array_equal(result['ids'],ids);np.testing.assert_array_equal(result['classes'],labels)
assert result['plain_logp'].dtype==np.float32 and result['plain_logp'].shape==(len(ids),18) and np.isfinite(result['plain_logp']).all()
assert torch.cuda.max_memory_reserved()<=13*1024**3
from numeric_transport import save_numeric
save_numeric(cache/'olmo_plain.npz',ids=ids,labels=labels,plain_logp=result['plain_logp'])
save_numeric(Path('/kaggle/working/olmo_plain_recovery.npz'),ids=ids,labels=labels,plain_logp=result['plain_logp'])
Path('/kaggle/working/olmo_inference_audit.json').write_text(json.dumps(dict(status='FROZEN_OLMO2_NATIVE24_PLAIN_QUERY_COMPLETE',rows=len(ids),feature_width=24576,plain_dtype='float32',classes=labels.tolist(),original_model_first24_final_RMSNorm=True,pool_layers=[4,8,12,16,20,24],synthetic_anchor_replay=model.anchor_replay,CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],query_cuda_event_ms=event_ms,peak_reserved_bytes=torch.cuda.max_memory_reserved(),wall_seconds=time.monotonic()-started,model_fits=0),indent=2))
