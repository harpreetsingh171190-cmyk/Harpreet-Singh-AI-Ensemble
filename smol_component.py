"""Frozen full36 native SmolLM3 loader and B3 representation; no head fitting."""
import hashlib,importlib.metadata,json,time
from pathlib import Path
import numpy as np
from numeric_transport import load_numeric

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

class SmolFinal:
    def __init__(self,spec_path,assets,base):
        import importlib.util
        finder=importlib.util.find_spec
        importlib.util.find_spec=lambda n,*a,**kw:None if n.split('.')[0]=='torchao' else finder(n,*a,**kw)
        import torch
        from torch import nn
        from peft import PeftModel
        import peft.import_utils as pi
        pi.is_torchao_available=lambda *a,**kw:False
        from safetensors.torch import load_file
        from scipy.special import softmax
        from smol_owned_path import load_owned_base
        self.started=time.monotonic();self.S=json.loads(Path(spec_path).read_text());S=self.S
        self.assets=Path(assets);self.base=Path(base)
        assert S['public_asset_release_admitted'] and S['asset_ref'],'Required public input still closed'
        assert {n:importlib.metadata.version(n) for n in S['core_versions']}==S['core_versions']
        assert torch.cuda.device_count()==1 and torch.cuda.get_device_capability(0)==(7,5)
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(13*1024**3/torch.cuda.get_device_properties(0).total_memory,0)
        manifest=self.assets/'final_inference_manifest.json'
        assert sha(manifest)==S['compact_manifest_sha256']
        M=json.loads(manifest.read_text())
        for name,record in M['files'].items():assert sha(self.assets/name)==record['sha256'],name
        for name,digest in S['model_files'].items():assert sha(self.assets/name)==digest,name
        for record in S['base_files']:assert sha(self.base/record['name'])==record['sha256']
        loading=[];frozen=load_owned_base(self.base,loading,self.deadline)
        self.model=PeftModel.from_pretrained(frozen,self.assets/'adapter',is_trainable=False,local_files_only=True).eval()
        assert not any(x.requires_grad for x in self.model.parameters())
        assert all(x.dtype==torch.float32 for n,x in self.model.named_parameters() if 'lora_' in n)
        self.layers=[6,12,18,24,30,36]
        self.norms=nn.ModuleList([nn.LayerNorm(2048) for _ in self.layers]).cuda().float()
        heads=load_file(self.assets/'heads.safetensors')
        self.norms.load_state_dict({k.removeprefix('norms.'):v for k,v in heads.items() if k.startswith('norms.')},strict=True)
        self.norms.eval();self.norms.requires_grad_(False)
        self.ro=load_numeric(self.assets/'readouts.npz')
        np.testing.assert_array_equal(self.ro['classes'],np.asarray(S['classes'],dtype=str))
        for route,prefix in [('plain','plain'),('center','centered')]:
            np.testing.assert_array_equal(self.ro[route+'_coef'],heads[prefix+'.weight'].numpy())
            np.testing.assert_array_equal(self.ro[route+'_intercept'],heads[prefix+'.bias'].numpy())
        self.norm_calls=[0]*6
        def observe(i):
            def hook(*unused):self.norm_calls[i]+=1
            return hook
        self.hooks=[norm.register_forward_hook(observe(i)) for i,norm in enumerate(self.norms)]
        anchors=load_numeric(self.assets/'final-production-T4-synthetic-anchors.npz')
        assert sha(self.assets/'final-production-T4-synthetic-anchors.npz')==S['production_anchor_sha256']
        assert sha(self.assets/'synthetic3.json')==S['synthetic_sha256']==str(anchors['synthetic_text_sha256'])
        for key,rel in [('model_adapter_sha256','adapter/adapter_model.safetensors'),('heads_sha256','heads.safetensors'),('readouts_sha256','readouts.npz')]:assert str(anchors[key])==S['model_files'][rel]
        np.testing.assert_array_equal(anchors['classes'],self.ro['classes'])
        a=self.features(json.loads((self.assets/'synthetic3.json').read_text())['texts'],sort=False)
        for key in ['features','reloaded_features']:np.testing.assert_allclose(a,anchors[key],rtol=1e-5,atol=1e-5)
        drift={}
        for route in ['plain','center']:
            x=a if route=='plain' else a-a.mean(0)
            b=anchors['features'];y=b if route=='plain' else b-b.mean(0)
            d=float(np.abs(softmax(x@self.ro[route+'_coef'].T+self.ro[route+'_intercept'],axis=1)-softmax(y@self.ro[route+'_coef'].T+self.ro[route+'_intercept'],axis=1)).max())
            assert d<=1e-6;drift[route]=d
        self.anchor_replay=dict(feature_maxabs=float(np.abs(a-anchors['features']).max()),probability_maxabs=drift,feature_rtol=1e-5,feature_atol=1e-5,probability_atol=1e-6,scope='Saved SAME-production-T4 synthetic3; cross-backend tolerances not widened')
        assert self.norm_calls==[1]*6,'Exactly one trained LayerNorm per pool, no second LN'

    def deadline(self):assert time.monotonic()-self.started<9000,'Bounded Smol worker; no retry'

    def features(self,texts,sort=True):
        import torch
        from smol_owned_path import tokenize_owned,make_representation
        self.deadline();tokenizer,tokens=tokenize_owned(self.base,texts)
        fn=make_representation(self.model,tokenizer,tokens,self.layers,self.norms)
        order=np.argsort([len(t) for t in tokens],kind='stable') if sort else np.arange(len(tokens))
        z=np.empty((len(tokens),12288),np.float32)
        with torch.inference_mode():
            for start in range(0,len(order),3):
                self.deadline();ii=order[start:start+3];z[ii]=fn(ii)[0].cpu().numpy()
        assert np.isfinite(z).all();return z
