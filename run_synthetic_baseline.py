"""Diagnostic stub only: numeric zeros and fixed synthetic groups, NO baseline model."""
import hashlib,json,os
from pathlib import Path
import numpy as np
import pandas as pd
from numeric_transport import save_numeric
HERE=Path(__file__).resolve().parent
assert os.environ.get('NTL_RUN_MODE')=='diagnostic'
C=json.loads((HERE/'candidate-config.json').read_text())
assert C['cloud_probe_authorized'] and not C['candidate_admitted']
cache=Path(os.environ['NTL_CANDIDATE_TEMP']).resolve()
assert cache.is_relative_to(Path('/kaggle/temp').resolve())
fixture=json.loads((HERE/'diagnostic60.json').read_text())
assert fixture['rows']==60 and fixture['eligible_rows']==27
ids=np.asarray(fixture['ids'],dtype=str);groups=np.asarray(fixture['groups'],dtype=np.int64)
assert len(ids)==len(set(ids))==60 and len(groups)==60
size=np.bincount(groups)[groups];eligible=(size>=2)&(size<=18)
assert int(eligible.sum())==27
labels=np.asarray(fixture['Nishant_original_labels'],dtype=str)
assert len(set(labels))==18 and labels.tolist()!=sorted(labels.tolist())
query=Path(os.environ['NTL_QUERY_CSV']);assert query.parent==cache
pd.DataFrame({'id':ids,'text':fixture['texts']}).to_csv(query,index=False)
check=pd.read_csv(query,dtype=str);np.testing.assert_array_equal(check.id.to_numpy(dtype=str),ids)
save_numeric(cache/'baseline-state.npz',ids=ids,labels=labels,raw=np.zeros((60,18),np.float64))
save_numeric(cache/'baseline-groups.npz',ids=ids,groups=groups,size=size,eligible=eligible)
Path('/kaggle/working/diagnostic_stub_audit.json').write_text(json.dumps(dict(status='SYNTHETIC_ZERO_RAW_NOT_BASELINE_PREDICTIONS',rows=60,eligible_rows=27,group_sizes=[18,4,3,2,19]+[1]*14,fixture_sha256=hashlib.sha256((HERE/'diagnostic60.json').read_bytes()).hexdigest(),real_Qwen_or_SVC_or_Gaussian_fit=False,official_rows=0,not_validation=True),indent=2))
