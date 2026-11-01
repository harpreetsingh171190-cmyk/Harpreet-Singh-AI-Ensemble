"""Portable trained final450 OLMo2 native first24 plain readout; no fitting."""
import hashlib,importlib.metadata,json,os
from pathlib import Path
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def validate_contract(spec,artifact_root,base_root,diagnostic_only=False):
    assert not diagnostic_only and spec['inference_authorized'] and spec['root_production_admitted']
    root,base=Path(artifact_root),Path(base_root)
    assert sha(root/'asset_manifest.json')==spec['compact_manifest_sha256']
    manifest=json.loads((root/'asset_manifest.json').read_text())
    assert manifest['status']=='PUBLIC_SAFE_OLMO2_NATIVE24_FINAL_INFERENCE'
    for flag in ('contains_canonical_competition_IDs','contains_competition_text_or_token_IDs','contains_targets','contains_optimizer_or_RNG','contains_training_or_query_features'):
        assert manifest[flag] is False
    assert {r['relative_path'] for r in manifest['files']}==set(spec['public_file_allowlist'])
    for rec in manifest['files']:
        path=root/rec['relative_path'];assert path.resolve().is_relative_to(root.resolve())
        assert path.stat().st_size==rec['bytes'] and sha(path)==rec['sha256']
    provenance=json.loads((root/'provenance.json').read_text())
    assert provenance['source_notebook_id']==139418845 and provenance['source_script_version_id']==359697707
    assert provenance['source_worker_sha256']==spec['training_worker_sha256']
    assert provenance['base_revision']==spec['base_revision'] and provenance['native_final_RMSNorm_retained']
    assert provenance['retained_layers']==24 and provenance['pool_layers']==[4,8,12,16,20,24] and provenance['feature_width']==24576
    assert provenance['successful_updates']==450 and provenance['epochs']==3 and provenance['final_not_OOF']
    assert provenance['final_overflow_attempts']==0 and provenance['outer0_bounded_AMP_recoveries']==4
    assert provenance['classes']==spec['classes'] and provenance['synthetic_texts']==spec['synthetic_texts']
    assert provenance['original_private_anchor_replay_passed'] and provenance['actual_GPU_diagnostic_passed']
    config=json.loads((root/'adapter/adapter_config.json').read_text())
    assert config['r']==8 and config['lora_alpha']==16 and config['lora_dropout']==.05 and config['bias']=='none'
    assert set(config['target_modules'])=={'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'}
    assert config['auto_mapping']['base_model_class']=='Olmo2Model' and config['peft_type']=='LORA'
    for rec in spec['base_files']:
        path=base/rec['name'];assert path.stat().st_size==rec['bytes'] and sha(path)==rec['sha256']
    return root,base,provenance
class OLMoFinalPlain:
    def __init__(self, spec_path, artifact_root, base_root, anchor_train_csv, diagnostic_only=False):
        self.spec = json.loads(Path(spec_path).read_text())
        for name, digest in self.spec["source_hashes"].items():
            assert sha(Path(__file__).parent / name) == digest
        checkpoint, base_dir, report = validate_contract(self.spec, artifact_root, base_root, diagnostic_only=diagnostic_only)
        assert os.name != "nt" and Path("/kaggle/input").is_dir(), "Remote-only component"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import importlib.util
        finder = importlib.util.find_spec
        importlib.util.find_spec = lambda name, *a, **kw: None if name.split(".")[0] == "torchao" else finder(name, *a, **kw)
        # Keep the successful FULL worker's torchao mask active through PEFT dispatch.
        import torch
        from torch import nn
        from peft import PeftModel
        import peft.import_utils as peft_imports
        peft_imports.is_torchao_available = lambda *a, **kw: False
        import numpy as np
        import pandas as pd
        from safetensors.torch import load_file
        from scipy.special import log_softmax, softmax
        from olmo_owned_path import load_owned_base, make_representation, tokenize_owned

        assert torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (7, 5)
        assert {name: importlib.metadata.version(name) for name in self.spec["core_versions"]} == self.spec["core_versions"]
        torch.set_num_threads(2)
        loading = []
        # Fail before LM allocation on actual serialized readout geometry/class mismatch.
        with np.load(checkpoint/'readouts.npz',allow_pickle=False) as checked:
            np.testing.assert_array_equal(checked['classes'],np.asarray(self.spec['classes'],dtype=str))
            np.testing.assert_array_equal(checked['pool_layers'],[4,8,12,16,20,24])
            for route in ('plain','center'):
                assert checked[route+'_coef'].shape==(18,24576) and checked[route+'_coef'].dtype==np.float32
                assert checked[route+'_intercept'].shape==(18,) and checked[route+'_intercept'].dtype==np.float32
        base = load_owned_base(base_dir, loading, lambda: None)
        # Semantic resolution of actual serialized7suffixes; never assert serialized168 paths.
        native_targets={name:module for name,module in base.named_modules() if isinstance(module,nn.Linear) and name.startswith('layers.') and name.rsplit('.',1)[-1] in {'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'}}
        assert len(native_targets)==168
        shapes={'self_attn.q_proj':(4096,4096),'self_attn.k_proj':(4096,4096),'self_attn.v_proj':(4096,4096),'self_attn.o_proj':(4096,4096),'mlp.gate_proj':(4096,11008),'mlp.up_proj':(4096,11008),'mlp.down_proj':(11008,4096)}
        for layer in range(24):
            for suffix,shape in shapes.items():
                module=native_targets[f'layers.{layer}.{suffix}'];assert (module.in_features,module.out_features)==shape and module.bias is None
        del native_targets,module
        self.loading = loading
        self.model = PeftModel.from_pretrained(base, checkpoint / "adapter", is_trainable=False, local_files_only=True).eval()
        expanded=[name for name,module in self.model.named_modules() if hasattr(module,'lora_A')]
        assert len(expanded)==168
        assert len(self.model.base_model.model.layers)==24 and self.model.base_model.model.norm is not None
        assert not any(parameter.requires_grad for parameter in self.model.parameters())
        assert all(parameter.dtype == torch.float32 for name, parameter in self.model.named_parameters() if "lora_" in name)
        self.layers = [4, 8, 12, 16, 20, 24]
        self.norms = nn.ModuleList([nn.LayerNorm(4096) for _ in self.layers]).cuda().float()
        saved = load_file(checkpoint / "heads.safetensors")
        assert all(value.dtype == torch.float32 and bool(torch.isfinite(value).all()) for value in saved.values())
        self.norms.load_state_dict({key.removeprefix("norms."): value for key, value in saved.items() if key.startswith("norms.")}, strict=True)
        self.norms.eval()
        self.norms.requires_grad_(False)
        with np.load(checkpoint / "readouts.npz", allow_pickle=False) as readouts:
            self.readouts = {key: readouts[key].copy() for key in readouts.files}
        np.testing.assert_array_equal(self.readouts["classes"], np.asarray(self.spec["classes"], dtype=str))
        np.testing.assert_array_equal(self.readouts["pool_layers"], self.layers)
        assert self.readouts["plain_coef"].shape == self.readouts["center_coef"].shape == (18, 24576)
        assert self.readouts["plain_intercept"].shape == self.readouts["center_intercept"].shape == (18,)
        for route, prefix in (("plain", "plain"), ("center", "centered")):
            for key, state_key in (("coef", "weight"), ("intercept", "bias")):
                array = self.readouts[route + "_" + key]
                assert array.dtype == np.float32 and np.isfinite(array).all()
                np.testing.assert_array_equal(array, saved[prefix + "." + state_key].numpy())
        # Synthetic literals only; canonical private B2 replay was checked before packaging.
        self._diagnostic_only=False
        self._torch,self._np=torch,np
        self._make_representation,self._tokenize_owned=make_representation,tokenize_owned
        self._log_softmax=log_softmax
        with np.load(checkpoint/'synthetic_anchors.npz',allow_pickle=False) as anchors:
            assert set(anchors.files)=={'ids','texts','features','plain_logp','classes'}
            np.testing.assert_array_equal(anchors['ids'],['synthetic-a','synthetic-b','synthetic-c'])
            np.testing.assert_array_equal(anchors['classes'],self.readouts['classes'])
            texts=anchors['texts'].tolist();assert texts==self.spec['synthetic_texts']
            expected=anchors['features'].copy()
            expected_logp=anchors['plain_logp'].copy()
            assert expected.shape==(3,24576) and expected.dtype==np.float32
            assert expected_logp.shape==(3,18) and expected_logp.dtype==np.float32
        self.tokenizer,_=tokenize_owned(base_dir,texts)
        recorded_reloaded=expected
        fresh = self._features(texts, sort_by_length=True)
        fresh_logp=log_softmax(fresh @ self.readouts['plain_coef'].T+self.readouts['plain_intercept'],axis=1)
        np.testing.assert_allclose(fresh_logp,expected_logp,rtol=0,atol=1e-6)
        assert expected.dtype == fresh.dtype == np.float32 and np.isfinite(expected).all()
        np.testing.assert_allclose(fresh, expected, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(fresh, recorded_reloaded, rtol=1e-5, atol=1e-5)
        parity = {}
        for route in ("plain", "center"):
            a = expected if route == "plain" else expected - expected.mean(0)
            b = fresh if route == "plain" else fresh - fresh.mean(0)
            coef, intercept = self.readouts[route + "_coef"], self.readouts[route + "_intercept"]
            before = softmax(log_softmax(a @ coef.T + intercept, axis=1), axis=1)
            after = softmax(log_softmax(b @ coef.T + intercept, axis=1), axis=1)
            parity[route] = float(np.max(np.abs(before - after)))
            assert parity[route] <= 1e-6
        self.anchor_replay = {"feature_maxabs": float(np.max(np.abs(fresh - expected))), "probability_maxabs": parity,
                              "feature_rtol": 1e-5, "feature_atol": 1e-5, "probability_atol": 1e-6}

    def _features(self, texts, sort_by_length=True):
        torch, np = self._torch, self._np
        tokenizer, tokens = self._tokenize_owned(None, texts, tokenizer=self.tokenizer)
        representation = self._make_representation(self.model, tokenizer, tokens, self.layers, self.norms)
        features = np.empty((len(tokens), 24576), dtype=np.float32)
        order = np.argsort(np.asarray([len(row) for row in tokens]), kind="stable") if sort_by_length else np.arange(len(tokens))
        with torch.inference_mode():
            for start in range(0, len(order), 2):
                positions = order[start:start + 2]
                features[positions] = representation(positions)[0].cpu().numpy()
        assert np.isfinite(features).all()
        return features

    def predict_plain_logp(self, ids, texts):
        np = self._np
        ids = np.asarray(ids, dtype=str)
        assert ids.ndim == 1 and len(ids) == len(texts) > 0 and len(np.unique(ids)) == len(ids)
        if self._diagnostic_only:
            assert len(ids) in (3, 5) and ids.tolist() == self._diagnostic_ids[:len(ids)]
            assert list(texts) == self._diagnostic_texts[:len(ids)], "Only fixed synthetic3/5-row diagnostic fixtures admitted"
        features = self._features(texts)
        logp = self._log_softmax(features @ self.readouts["plain_coef"].T + self.readouts["plain_intercept"], axis=1)
        assert logp.shape == (len(ids), 18) and logp.dtype == np.float32 and np.isfinite(logp).all()
        return {"ids": ids, "classes": self.readouts["classes"].copy(), "plain_logp": logp,
                "features": features, "pool_layers": np.asarray(self.layers)}
