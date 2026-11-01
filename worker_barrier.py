"""File barrier after native loading/replay and before each owned query forward."""
import json,os,time
from pathlib import Path

def ready_and_wait(name,timeout=900):
    assert name in ('olmo','smol')
    root=Path(os.environ['NTL_CANDIDATE_TEMP']).resolve()
    assert root.is_relative_to(Path('/kaggle/temp').resolve())
    assert os.environ['CUDA_VISIBLE_DEVICES'].startswith('GPU-')
    path=root/(name+'-READY.json');assert not path.exists()
    path.write_text(json.dumps(dict(name=name,PID=os.getpid(),UUID=os.environ['CUDA_VISIBLE_DEVICES'])))
    started=time.monotonic()
    while not (root/'OWNED-GO').exists():
        assert time.monotonic()-started<timeout,'Bounded READY/GO wait; no retry'
        time.sleep(.1)
