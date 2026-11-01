"""Two-stage Qwen and grouped Gaussian pass; retain numeric transport only."""
import json,os
from pathlib import Path
from baseline_source import source_parts
from numeric_transport import save_numeric

HERE=Path(__file__).resolve().parent
OUT=Path('/kaggle/working')
cache=Path(os.environ['NTL_CANDIDATE_TEMP']).resolve()
assert cache.is_relative_to(Path('/kaggle/temp').resolve())
query=os.environ.get('NTL_QUERY_CSV')
import numpy as np
import pandas as pd
# Identity failures are detected before any model is loaded.
paths=[Path(query)] if query is not None else [p/'test.csv' for p in
    (Path('/kaggle/input/competitions/name-that-llm'),Path('/kaggle/input/name-that-llm')) if (p/'test.csv').is_file()]
assert len(paths)==1
canonical=pd.read_csv(paths[0],usecols=['id'],dtype=str)
assert not canonical['id'].isna().any()
canonical_ids=np.asarray(canonical['id'].tolist(),dtype=str)
assert canonical_ids.dtype.kind=='U' and len(np.unique(canonical_ids))==len(canonical_ids)>0
prefix,tail,functions=source_parts(HERE,query)
env={'__name__':'__main__','__file__':str(HERE/'nishant_0592_original.ipynb')}
exec(compile(prefix,'<two-stage-plus-grouped-Gaussian-before-final-softmax>','exec'),env)
np=env['np']
# Canonical identity was checked before the heavy source was executed.
ids=canonical_ids
assert len(ids)==len(env['test'])
labels=np.asarray(env['LABELS'],dtype=str)
raw=env['LOGIT']
assert raw.dtype==np.float64 and raw.shape==(len(ids),18) and np.isfinite(raw).all()
assert len(set(ids.tolist()))==len(ids) and set(labels)==set(json.loads((HERE/'olmo-spec.json').read_text())['classes'])
assert env['W_SVC']==1.021 and env['TWIN']==dict(thresh=.85,min_size=2,max_size=18)
assert env['LORAS']['lora8b_v1']['w']==.226 and env['LORAS']['lora8b_v1']['twin_w']==.079
assert env['LORAS']['lora14b_v1']['w']==.429 and env['LORAS']['lora14b_v1']['twin_w']==.150
arrays=dict(raw=raw,ids=ids,labels=labels)
save_numeric(cache/'baseline-state.npz',**arrays)
save_numeric(OUT/'two_stage_baseline_recovery.npz',**arrays)
meta=dict(source_notebook_id=139520703,version=2,script_version_id=359887828,grouped_Gaussian_method_notebook_id=138904218,grouped_Gaussian_method_version=4,grouped_Gaussian_method_script_version_id=360050107,rows=len(ids),baseline_start_unix=env['T0'],
    all_original_weights=True,heavy_baseline_passes=1,query_csv_override=query is not None,train_model_forward_passes=1,train_8B_rows=2699,Gaussian_fit=True,Gaussian_dimension=128,Gaussian_temperature=3.,Gaussian_weight=.10,Gaussian_group_sizes=[2,18],Gaussian_core_author="prvsiyan",new_Gaussian_head_OOF_available=False)
(cache/'baseline-meta.json').write_text(json.dumps(meta,indent=2))
(OUT/'two_stage_baseline_audit.json').write_text(json.dumps(meta,indent=2))
print('TWO_STAGE_RAW_COMPLETE_EXITING_TO_RELEASE_BOTH_GPUS',flush=True)
