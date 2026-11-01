"""Same native QA requirements at every size; honest short-workload overlap scope."""
import math

def assess_owned_evidence(rows,seen,overlap,ready,audits,pids,mapping):
    assert isinstance(rows,int) and rows>0
    assert len(ready)==len(audits)==len(pids)==len(mapping)==2
    assert len(set(pids))==2 and len({m['uuid'] for m in mapping})==2
    for i,(r,a) in enumerate(zip(ready,audits)):
        assert r['PID']==pids[i] and r['UUID']==mapping[i]['uuid']
        assert a['CUDA_VISIBLE_DEVICES']==mapping[i]['uuid'] and a['rows']==rows
        assert math.isfinite(a['query_cuda_event_ms']) and a['query_cuda_event_ms']>0
        assert 0<a['peak_reserved_bytes']<=13*1024**3
        native=a['synthetic_anchor_replay'] if i==0 else a['anchor_replay']
        assert native['feature_rtol']==native['feature_atol']==1e-5 and native['probability_atol']==1e-6
        assert math.isfinite(native['feature_maxabs']) and native['feature_maxabs']>=0
        assert set(native['probability_maxabs'])=={'plain','center'}
        assert all(math.isfinite(v) and 0<=v<=1e-6 for v in native['probability_maxabs'].values())
    assert overlap>=0 and seen<={0,1}
    proven=seen=={0,1} and overlap>0
    if rows>=50:
        assert proven,'N>=50 requires both sampled actual PID/UUID bindings and simultaneous GPU utilization after GO'
    return dict(rows=rows,concurrent_execution_proven=proven,overlap_evidence='OBSERVED_PID_UUID_UTIL_OVERLAP' if proven else 'INSUFFICIENT_SHORT_WORKLOAD',actual_PID_UUID_util_overlap_samples=overlap,actual_PID_UUID_seen=sorted(seen),both_native_QA_and_positive_query_CUDA_events=True,distinct_worker_PID_UUID_READY_bindings=True,strict_overlap_required=rows>=50)
