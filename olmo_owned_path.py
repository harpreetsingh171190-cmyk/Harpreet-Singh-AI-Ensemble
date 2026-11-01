"""Exact native final450 Olmo2Model prefix24 loader, tokenizer and representation."""
def load_owned_base(base_dir, loading, deadline):
    import json,torch
    from transformers import Olmo2Config,Olmo2Model
    original = Olmo2Config.from_pretrained(base_dir, local_files_only=True, trust_remote_code=False)
    assert (original.model_type, original.hidden_size, original.num_hidden_layers) == ('olmo2',4096,32)
    assert (original.num_attention_heads, original.num_key_value_heads, original.vocab_size) == (32,32,100352)
    assert original.intermediate_size == 11008 and not original.tie_word_embeddings and not original.attention_bias
    assert original.bos_token_id is None and original.eos_token_id == 100257 and original.pad_token_id == 100277
    assert original.rope_parameters['rope_theta'] == 500000 and original.rope_parameters['rope_type'] == 'default'
    assert original.rms_norm_eps == 1e-6 and original.attention_dropout == 0 and original.max_position_embeddings == 4096
    assert not getattr(original,'quantization_config',None) and not getattr(original,'auto_map',None)
    # Instantiate only the retained prefix: no full32 GPU model is ever constructed and pruned.
    retained_config = Olmo2Config.from_dict(original.to_dict())
    retained_config.num_hidden_layers = 24
    assert retained_config.hidden_size == 4096 and retained_config.num_hidden_layers == 24
    index = json.loads((base_dir/'model.safetensors.index.json').read_text())['weight_map']
    assert len(index) == 355 and 'lm_head.weight' in index
    original_layer_keys = {key for key in index if key.startswith('model.layers.')}
    assert len(original_layer_keys) == 352
    assert {int(key.split('.')[2]) for key in original_layer_keys} == set(range(32))
    ignored_index_keys = {'lm_head.weight'} | {key for key in original_layer_keys if int(key.split('.')[2]) >= 24}
    used_index_keys = set(index) - ignored_index_keys
    assert len(ignored_index_keys) == 89 and len(used_index_keys) == 266
    expected_ignored = {key.removeprefix('model.') for key in ignored_index_keys}
    expected_used = {key.removeprefix('model.') for key in used_index_keys}
    def load_base():
        deadline()
        base, info = Olmo2Model.from_pretrained(base_dir, config=retained_config, dtype=torch.float16, device_map={"": 0},
            attn_implementation="eager", trust_remote_code=False, local_files_only=True, output_loading_info=True)
        assert not info.get("missing_keys") and not info.get("mismatched_keys") and not info.get("error_msgs"), info
        assert {key.removeprefix('model.') for key in info.get('unexpected_keys', [])} == expected_ignored, info
        assert set(base.state_dict()) == expected_used
        assert sum(parameter.numel() for parameter in base.parameters()) == 5268443136
        assert len(base.layers) == 24 and not hasattr(base, "lm_head")
        assert base.__class__.__name__ == 'Olmo2Model' and base.__class__.__module__.startswith('transformers.models.olmo2.')
        assert base.config._attn_implementation == 'eager'
        assert not any(name.endswith('lm_head') for name, module in base.named_modules())
        assert all(parameter.device.type == "cuda" and parameter.device.index == 0 for parameter in base.parameters())
        assert all(parameter.dtype == torch.float16 for parameter in base.parameters())
        base.requires_grad_(False)
        loading.append(info)
        return base
    return load_base()

def make_representation(model,tokenizer,tokens,layers,norms):
    import torch
    def representation(rows):
        batch = tokenizer.pad({"input_ids": [tokens[i] for i in rows]}, return_tensors="pt").to("cuda:0")
        with torch.autocast("cuda", dtype=torch.float16):
            hidden = model(**batch, output_hidden_states=True, use_cache=False).hidden_states
        assert len(hidden) == 25
        pieces = []
        with torch.autocast("cuda", enabled=False):
            for layer, norm in zip(layers, norms):
                mask = batch.attention_mask.unsqueeze(-1).float()
                mask[:, 0] = 0
                assert bool((mask.sum(1) > 0).all())
                mean = (hidden[layer].float() * mask).sum(1) / mask.sum(1)
                pieces.append(norm(mean))
            z = torch.cat(pieces, -1)
        assert bool(torch.isfinite(z).all())
        return z, int(batch.attention_mask.sum()), int(batch.input_ids.numel())
    return representation

def tokenize_owned(base_dir,texts,tokenizer=None):
    from transformers import AutoTokenizer,Olmo2Config
    if tokenizer is None:
        original = Olmo2Config.from_pretrained(base_dir,local_files_only=True,trust_remote_code=False)
        tokenizer = AutoTokenizer.from_pretrained(str(base_dir), use_fast=True, local_files_only=True, trust_remote_code=False)
        assert tokenizer.is_fast and tokenizer.__class__.__module__.startswith('transformers.')
        # Official tokenizer metadata aliases BOS to EOS; model config has no native BOS.
        assert tokenizer.bos_token_id == tokenizer.eos_token_id == 100257 and tokenizer.pad_token_id == 100277
        assert len(tokenizer) == 100278 and original.vocab_size == 100352
        assert tokenizer.convert_tokens_to_ids('<|endoftext|>') == 100257
        assert tokenizer.convert_tokens_to_ids('<|pad|>') == 100277
        tokenizer.padding_side = 'right'
    content = tokenizer(texts, add_special_tokens=False, truncation=True, max_length=446)['input_ids']
    tokens = [[100257]+row+[100257] for row in content]
    assert all(2 <= len(row) <= 448 and row[0] == row[-1] == 100257 for row in tokens)
    del content
    return tokenizer,tokens