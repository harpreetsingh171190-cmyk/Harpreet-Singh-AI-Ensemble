"""Run the frozen v14 prefix, then export its ACTUAL original Nishant groups."""
import os,runpy
from pathlib import Path
import numpy as np
from numeric_transport import save_numeric
HERE=Path(__file__).resolve().parent
scope=runpy.run_path(str(HERE/'run_baseline.py'),run_name='__main__')
env=scope['env'];ids=scope['canonical_ids']
assert env['TWIN']==dict(thresh=.85,min_size=2,max_size=18)
groups=np.asarray(env['groups'],dtype=np.int64)
size=np.asarray(env['size'],dtype=np.int64)
eligible=np.asarray(env['twin_rows'],dtype=bool)
assert groups.shape==size.shape==eligible.shape==ids.shape
np.testing.assert_array_equal(size,np.bincount(groups)[groups])
np.testing.assert_array_equal(eligible,(size>=2)&(size<=18))
cache=Path(os.environ['NTL_CANDIDATE_TEMP']).resolve()
assert cache.is_relative_to(Path('/kaggle/temp').resolve())
save_numeric(cache/'baseline-groups.npz',ids=ids,groups=groups,size=size,eligible=eligible)
print('ORIGINAL_NISHANT_GROUPS_EXPORTED_NO_RECOMPUTATION',flush=True)
