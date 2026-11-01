"""Add one owned FP32 term, then execute the original full Sinkhorn tail."""
import json,os,time
from pathlib import Path
import numpy as np
import pandas as pd
from baseline_source import source_parts
from numeric_transport import load_numeric,save_numeric

HERE=Path(__file__).resolve().parent

def finish(data,own,output_dir,coefficient=.03,smol=None,smol_coefficient=.03):
    output_dir=Path(output_dir)
    assert coefficient in (0.,.03)
    assert set(data)=={'ids','labels','raw'} and set(own)=={'ids','labels','plain_logp'}
    ids,labels,raw=data['ids'],data['labels'],data['raw']
    assert ids.dtype.kind=='U' and labels.dtype.kind=='U' and len(np.unique(ids))==len(ids)
    assert raw.dtype==np.float64 and raw.shape==(len(ids),18) and np.isfinite(raw).all()
    np.testing.assert_array_equal(own['ids'],ids)
    assert own['labels'].dtype.kind=='U' and len(set(own['labels'].tolist()))==18
    assert set(own['labels'].tolist())==set(labels.tolist())
    reorder=np.asarray([own['labels'].tolist().index(label) for label in labels],dtype=int)
    original=own['plain_logp']
    assert original.dtype==np.float32 and original.shape==raw.shape and np.isfinite(original).all()
    ordered=original[:,reorder]
    # Match the established owned-member FP32 exp/clip/log contribution exactly.
    term=coefficient*np.log(np.clip(np.exp(ordered),1e-9,1.))
    assert term.dtype==np.float32 and np.isfinite(term).all()
    LOGIT=raw.copy()
    LOGIT+=term
    np.testing.assert_array_equal(LOGIT,raw+term)
    assert smol_coefficient in (0.,.03)
    if smol is not None:
        assert set(smol)=={'ids','labels','score'}
        np.testing.assert_array_equal(smol['ids'],ids)
        assert smol['labels'].dtype.kind=='U' and len(set(smol['labels']))==18
        assert set(smol['labels'])==set(labels)
        smol_order=np.asarray([smol['labels'].tolist().index(label) for label in labels],dtype=int)
        S=smol['score']
        assert S.dtype==np.float64 and S.shape==raw.shape and np.isfinite(S).all()
        # Preserve parent addorder exactly: original Qwen/Gaussian -> OLMo03 -> Smol03.
        # S already contains routed FP32 logp + eligible-only .10 FP64 Gaussian.
        # Never normalize, exponentiate or clip this member again.
        smol_term=smol_coefficient*S[:,smol_order]
        LOGIT+=smol_term
        np.testing.assert_array_equal(LOGIT,(raw+term)+smol_term)
    prefix,tail,functions=source_parts(HERE)
    env=dict(np=np,pd=pd,time=time,T0=time.time(),LOGIT=LOGIT,LABELS=labels.tolist(),K=18,test=pd.DataFrame({'id':ids}))
    exec(compile(functions['softmax'],'<original-nishant-softmax>','exec'),env)
    # Change only the output pathname, preserving every mathematical operation.
    needle='sub.to_parquet("submission.parquet", index=False)'
    assert tail.count(needle)==1
    tail=tail.replace(needle,'sub.to_parquet('+repr(str(output_dir/'submission.parquet'))+', index=False)')
    exec(compile(tail,'<exact-nishant-v2-full-100-step-sinkhorn-tail>','exec'),env)
    observed=pd.read_parquet(output_dir/'submission.parquet')
    np.testing.assert_array_equal(observed.id.to_numpy(dtype=str),ids)
    np.testing.assert_array_equal(observed.model.to_numpy(dtype=str),env['sub'].model.to_numpy(dtype=str))
    assert observed.columns.tolist()==['id','model'] and len(observed)==len(ids)
    return env

def main():
    cache=Path(os.environ['NTL_CANDIDATE_TEMP']).resolve()
    assert cache.is_relative_to(Path('/kaggle/temp').resolve())
    data=load_numeric(cache/'baseline-state.npz')
    own=load_numeric(cache/'olmo_plain.npz')
    smol=load_numeric(cache/'smol_score.npz')
    out=Path('/kaggle/working')
    candidate=finish(data,own,out,.03,smol,.03)
    save_numeric(out/'two_stage_final_scores.npz',ids=data['ids'],labels=data['labels'],probabilities=candidate['P'])
    audit=dict(status='TWO_STAGE_GAUSSIAN_OLMO03_SMOL03_COMPLETE',rows=len(data['ids']),
        OLMo_raw_weight=.03,Smol_raw_weight=.03,Smol_Gaussian_weight=.10,
        Smol_member_renormalized=False,original_uniform_Sinkhorn100=True,
        original_Nishant_group_sizes=[2,18],own_class_name_remaps=2,
        deterministic_addorder=['original_Qwen_Gaussian','OLMo03','Smol03'],
        full_stack_OOF_available=False,new_candidate_score=None)
    (out/'candidate_audit.json').write_text(json.dumps(audit,indent=2))

if __name__=='__main__':main()
