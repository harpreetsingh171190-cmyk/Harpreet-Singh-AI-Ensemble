"""Frozen baseline exits, then UUID-isolated OLMo0+Smol1 with actual telemetry."""
import hashlib,importlib.metadata,json,os,signal,subprocess,sys,tempfile,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
C=json.loads((HERE/'candidate-config.json').read_text())
assert os.name!='nt' and Path('/kaggle/input').is_dir() and C['source_ready']
mode=os.environ.get('NTL_RUN_MODE','candidate');assert mode in ('candidate','diagnostic')
if mode=='candidate':assert C['candidate_admitted'],'CLOSED draft; root admission required before candidate dispatch'
else:assert C['cloud_probe_authorized'] and not C['candidate_admitted'],'CLOSED targeted diagnostic; separate root ACK required'
assert C['Smol_asset_publication_admitted'],'Required final Smol dataset not yet published'
for name,digest in C['source_sha256'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest,name
S=json.loads((HERE/'smol-spec.json').read_text())
assert S['public_asset_release_admitted'] and S['asset_ref']
assert {n:importlib.metadata.version(n) for n in S['core_versions']}==S['core_versions']
assert not os.environ.get('NTL_QUERY_CSV') and not os.environ.get('NTL_SMOKE')
mapping=[]
for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader,nounits'],text=True).strip().splitlines():
    index,uuid,name=[x.strip() for x in line.split(',',2)]
    mapping.append(dict(index=int(index),uuid=uuid,name=name))
mapping.sort(key=lambda x:x['index'])
assert [x['index'] for x in mapping]==[0,1] and len({x['uuid'] for x in mapping})==2 and all('T4' in x['name'] for x in mapping)
Path('/kaggle/temp').mkdir(exist_ok=True)
cache=Path(tempfile.mkdtemp(prefix='ntl-qwen-olmo-smol-',dir='/kaggle/temp'))
env=os.environ.copy();env['NTL_CANDIDATE_TEMP']=str(cache)
if mode=='diagnostic':env['NTL_QUERY_CSV']=str(cache/'diagnostic60-query.csv')
env['HF_HUB_OFFLINE']=env['TRANSFORMERS_OFFLINE']='1';env['TOKENIZERS_PARALLELISM']='false'
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[n]='2'
# Leave 30s for bounded child termination before the notebook's external 900s cap.
deadline=time.monotonic()+(870 if mode=='diagnostic' else 10800)

def remaining():
    seconds=deadline-time.monotonic()
    assert seconds>0,'Bounded total runtime; no retry'
    return seconds

def stop(children):
    for child in children:
        if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
    for child in children:
        try:child.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()

def devices():
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True,timeout=10)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True,timeout=10)
    values=[]
    for line in gpu.strip().splitlines():
        index,uuid,util,memory=[x.strip() for x in line.split(',')]
        values.append(dict(index=int(index),uuid=uuid,util=float(util),memory_MiB=float(memory)))
    values.sort(key=lambda x:x['index'])
    assert [x['uuid'] for x in values]==[x['uuid'] for x in mapping]
    pairs={tuple(x.strip() for x in line.split(',')[:2]) for line in apps.strip().splitlines() if line.strip()}
    return values,pairs,apps

reports=[];baseline_env=env.copy();baseline_env['CUDA_VISIBLE_DEVICES']=','.join(x['uuid'] for x in mapping)
with Path('/kaggle/working/baseline.log').open('w',buffering=1) as log:
    start=time.monotonic()
    baseline_script='run_synthetic_baseline.py' if mode=='diagnostic' else 'run_baseline_with_groups.py'
    baseline=subprocess.Popen([sys.executable,'-u',str(HERE/baseline_script)],env=baseline_env,cwd='/kaggle/working',stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        while baseline.poll() is None:
            remaining();time.sleep(.5)
        assert baseline.returncode==0,'Baseline failed; no retry'
    finally:stop([baseline])
    reports.append(dict(stage='baseline',PID=baseline.pid,seconds=time.monotonic()-start,exit_code=0))
assert (cache/'baseline-state.npz').is_file() and (cache/'baseline-groups.npz').is_file()
from numeric_transport import load_numeric
baseline_rows=int(len(load_numeric(cache/'baseline-state.npz')['ids']))
if mode=='diagnostic':assert baseline_rows==60
# Process termination, not a Python del, releases every baseline CUDA allocation.
values,pairs,apps=devices()
assert not pairs,('Baseline exited but a CUDA process remains',apps)
children=[];logs=[];seen=set();overlap=0;started=time.monotonic()
try:
    for i,name in enumerate(['olmo','smol']):
        child_env=env.copy();child_env['CUDA_VISIBLE_DEVICES']=mapping[i]['uuid']
        log=Path('/kaggle/working',name+'.log').open('w',buffering=1);logs.append(log)
        children.append(subprocess.Popen([sys.executable,'-u',str(HERE/('compute_'+name+'.py'))],env=child_env,cwd='/kaggle/working',stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
    while True:
        remaining();codes=[c.poll() for c in children]
        assert all(c in (None,0) for c in codes),('Owned worker failed; no retry',codes)
        ready=[cache/(name+'-READY.json') for name in ['olmo','smol']]
        if all(p.exists() for p in ready) and not (cache/'OWNED-GO').exists():
            for i,p in enumerate(ready):
                r=json.loads(p.read_text());assert r['PID']==children[i].pid and r['UUID']==mapping[i]['uuid']
            (cache/'OWNED-GO').touch()
        values,pairs,apps=devices()
        matched=[(str(c.pid),mapping[i]['uuid']) in pairs for i,c in enumerate(children)]
        for i,v in enumerate(matched):
            if v:seen.add(i)
        active=(cache/'OWNED-GO').exists() and all(matched) and all(v['util']>0 and v['memory_MiB']>0 for v in values)
        overlap+=int(active)
        with Path('/kaggle/working/owned_gpu_telemetry.jsonl').open('a') as f:
            f.write(json.dumps(dict(seconds=time.monotonic()-started,PID=[c.pid for c in children],physical_devices=values,processes=apps,matched=matched,after_GO=(cache/'OWNED-GO').exists(),simultaneous_PID_UUID_util=active))+'\n')
        if all(c==0 for c in codes):break
        time.sleep(.25)
    assert (cache/'olmo_plain.npz').is_file() and (cache/'smol_score.npz').is_file()
    from concurrency_evidence import assess_owned_evidence
    evidence=assess_owned_evidence(baseline_rows,seen,overlap,
        [json.loads((cache/(name+'-READY.json')).read_text()) for name in ['olmo','smol']],
        [json.loads(Path('/kaggle/working',name+'_inference_audit.json').read_text()) for name in ['olmo','smol']],
        [c.pid for c in children],mapping)
finally:
    stop(children)
    for log in logs:log.close()
values,pairs,apps=devices();assert not pairs,('Owned CUDA child survives exit',apps)
subprocess.run([sys.executable,'-u',str(HERE/'finish_prediction.py')],env=env,cwd='/kaggle/working',check=True,timeout=min(120,remaining()))
assert json.loads(Path('/kaggle/working/candidate_audit.json').read_text())['status']=='TWO_STAGE_GAUSSIAN_OLMO03_SMOL03_COMPLETE'
Path('/kaggle/working/parallel_execution_audit.json').write_text(json.dumps(dict(status='COMPLETE_BASELINE_THEN_TWO_OWNED_WORKERS',mode=mode,baseline=reports[0],baseline_is_synthetic_stub=mode=='diagnostic',own_worker_PIDs=[c.pid for c in children],GPU_mapping=mapping,evidence=evidence,baseline_owned_overlap=False,parent_model_initializations=0,automatic_retries=0,addorder=['original_Qwen_Gaussian','OLMo03','Smol03']),indent=2))
