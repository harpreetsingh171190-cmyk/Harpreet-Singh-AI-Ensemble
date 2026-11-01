"""No-pickle numeric transport; validate the bytes actually written."""
from pathlib import Path
import numpy as np

def save_numeric(path, **arrays):
    arrays={key:np.asarray(value) for key,value in arrays.items()}
    assert all(not value.dtype.hasobject for value in arrays.values()), 'Object arrays forbidden'
    assert all(value.dtype.kind in 'USbiuf' for value in arrays.values())
    assert all(np.isfinite(value).all() for value in arrays.values() if value.dtype.kind in 'f')
    path=Path(path)
    assert not path.exists(), 'No automatic overwrite or reuse'
    np.savez_compressed(path,**arrays)
    recovered=load_numeric(path)
    assert set(recovered)==set(arrays)
    for key,value in arrays.items():
        assert recovered[key].dtype==value.dtype
        np.testing.assert_array_equal(recovered[key],value)

def load_numeric(path):
    with np.load(path,allow_pickle=False) as saved:
        arrays={key:saved[key].copy() for key in saved.files}
    assert all(not value.dtype.hasobject for value in arrays.values())
    return arrays
