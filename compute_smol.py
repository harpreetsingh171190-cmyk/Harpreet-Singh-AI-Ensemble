"""One frozen Smol query forward and read-only Gaussian score, GPU1 only."""
import json,os,time
from pathlib import Path
import numpy as np
import pandas as pd
from smol_component import SmolFinal,sha
from numeric_transport import load_numeric,save_numeric
from load_gaussian import load_gaussian
from smol_readout import smol_score
from worker_barrier import ready_and_wait
HERE=Path(__file__).resolve().parent
C=json.loads((HERE/'candidate-config.json').read_text());S=json.loads((HERE/'smol-spec.json').read_text())
assert C['source_ready'] and C['Smol_raw_coefficient']==.03 and C['Smol_Gaussian_weight']==.10
assert S['public_asset_release_admitted'] and S['asset_ref']
cache=Path(os.environ['NTL_CANDIDATE_TEMP']).resolve();assert cache.is_relative_to(Path('/kaggle/temp').resolve())
assert not (cache/'smol_score.npz').exists(),'No automatic repeat/reuse'
assets=[p for p in Path('/kaggle/input').rglob('final_inference_manifest.json') if sha(p)==S['compact_manifest_sha256']]
markers=[p for p in Path('/kaggle/input').rglob('smollm3_staging_complete.json') if sha(p)==S['base_manifest_sha256']]
tests=[p/'test.csv' for p in (Path('/kaggle/input/competitions/name-that-llm'),Path('/kaggle/input/name-that-llm')) if (p/'test.csv').is_file()]
if os.environ.get('NTL_QUERY_CSV'):tests=[Path(os.environ['NTL_QUERY_CSV'])]
assert len(assets)==len(markers)==len(tests)==1
query=pd.read_csv(tests[0],usecols=['id','text'],dtype=str);query['text']=query['text'].fillna('').astype(str)
ids=query.id.to_numpy(dtype=str);groups=load_numeric(cache/'baseline-groups.npz')
np.testing.assert_array_equal(groups['ids'],ids)
assert len(set(ids))==len(ids)>0 and os.environ['CUDA_VISIBLE_DEVICES'].startswith('GPU-')
model=SmolFinal(HERE/'smol-spec.json',assets[0].parent,markers[0].parent/'smollm3-base')
assert sha(assets[0].parent/'gaussian-parameters.npz')==S['Gaussian_parameter_sha256']
assert sha(assets[0].parent/'gaussian-metadata.json')==S['Gaussian_metadata_sha256']
hashes={key:S['model_files'][rel] for key,rel in [('model_adapter_sha256','adapter/adapter_model.safetensors'),('heads_sha256','heads.safetensors'),('readouts_sha256','readouts.npz')]}
prm,labels=load_gaussian(assets[0].parent/'gaussian-parameters.npz',model.ro['classes'],hashes)
ready_and_wait('smol')
import torch
begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
before=time.monotonic();begin.record();z=model.features(query.text.tolist());end.record();torch.cuda.synchronize()
event_ms=float(begin.elapsed_time(end));assert event_ms>0
expected_batches=1+(len(ids)+2)//3
assert model.norm_calls==[expected_batches]*6,'One trained LN per pool per batch, including synthetic3 replay'
score,routed=smol_score(z,model.ro,groups['groups'],groups['eligible'],prm)
save_numeric(cache/'smol_score.npz',ids=ids,labels=labels,score=score)
save_numeric(Path('/kaggle/working/smol_score_recovery.npz'),ids=ids,labels=labels,score=score)
assert torch.cuda.max_memory_reserved()<=13*1024**3
report=dict(status='FROZEN_SMOL36_ROUTED_GAUSSIAN_QUERY_COMPLETE',PID=os.getpid(),CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],rows=len(ids),eligible_rows=int(groups['eligible'].sum()),feature_width=12288,feature_dtype='float32',routed_dtype=str(routed.dtype),score_dtype=str(score.dtype),query_cuda_event_ms=event_ms,wall_seconds=time.monotonic()-before,peak_reserved_bytes=torch.cuda.max_memory_reserved(),anchor_replay=model.anchor_replay,norm_calls=model.norm_calls,model_fits=0,head_fits=0,Gaussian_fits=0,raw_member_weight=.03,Gaussian_weight=.10,member_renormalized=False,original_Nishant_groups=True)
Path('/kaggle/working/smol_inference_audit.json').write_text(json.dumps(report,indent=2))
