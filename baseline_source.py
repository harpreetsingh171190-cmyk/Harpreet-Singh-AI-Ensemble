"""Nishant v2 partition plus independently written grouped Gaussian8B integration."""
import ast, hashlib, json
from pathlib import Path

def source_parts(here, query_csv=None):
    raw=(Path(here)/'nishant_0592_original.ipynb').read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='12e1e65bbc48a2839842d439fd7ae3846b9bf0d9913c4cdc308ebc698974b3fb'
    notebook=json.loads(raw)
    code='\n\n'.join(c['source'] if isinstance(c['source'],str) else ''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
    marker='P = softmax(LOGIT)'
    assert code.count(marker)==1
    prefix,tail=code.split(marker)
    original='Z, heads = lora_features(ckpt, find_base(mc["base"]), test.text.tolist(), mc["bs"])'
    assert prefix.count(original)==1
    patch=(Path(here)/'gaussian_integration.txt').read_text(encoding='utf-8').rstrip()
    prefix=prefix.replace(original,patch)
    marker_loop='for ckpt_name, mc in LORAS.items():'
    assert prefix.count(marker_loop)==1
    prefix=prefix.replace(marker_loop,'from gaussian_core import fit, scores\n\n'+marker_loop)
    if query_csv is not None:
        needle='test = pd.read_csv(find("test.csv"))'
        assert prefix.count(needle)==1
        prefix=prefix.replace(needle,'test = pd.read_csv('+repr(str(query_csv))+')')
    checkpoint_loop='    for cfg in sorted(glob.glob(f"{INPUT_DIR}/**/config.json", recursive=True)):\n'
    assert prefix.count(checkpoint_loop)==1
    prefix=prefix.replace(checkpoint_loop,checkpoint_loop+"        directory = os.path.dirname(cfg)\n        if not any(os.path.isfile(os.path.join(directory, name)) for name in ('model.safetensors', 'model.safetensors.index.json', 'pytorch_model.bin', 'pytorch_model.bin.index.json')):\n            continue  # metadata-only training copies are not checkpoints\n")
    functions={n.name:ast.get_source_segment(code,n) for n in ast.parse(code).body if isinstance(n,ast.FunctionDef)}
    return prefix,marker+tail,functions
