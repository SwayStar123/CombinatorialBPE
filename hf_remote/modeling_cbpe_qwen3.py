"""AutoModelForCausalLM entry point (trust_remote_code=True) for Qwen3 models on a factored tokenizer.

    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)       # bf16, GPU if available
    out = model.generate(**tok("The capital of France is", return_tensors="pt").to(model.device),
                         max_new_tokens=40, tokenizer=tok)                            # tokenizer= turns token healing on
    print(tok.decode(out[0]))

`input_ids` hold one integer per token: the four fields (variation, prefix, core, suffix) packed in
mixed radix (the tokenizer's `pack`). The model has no flat vocabulary, so there are no `logits`: the
output is predicted field by field (the "chain head", see cbpe_qwen3.py); `forward(labels=...)` returns
the mean negative log-likelihood per token, `token_nll` its per-field values, and `generate` is the
field-by-field sampler with a KV cache (greedy, or do_sample with temperature / top_p).
"""
import os

import torch
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .cbpe_qwen3 import FactoredQwen3, generate_tokens
from .configuration_cbpe_qwen3 import CBPEQwen3Config

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class CBPEQwen3ForCausalLM(PreTrainedModel):
    config_class = CBPEQwen3Config
    main_input_name = "input_ids"

    def __init__(self, config, dtype=torch.bfloat16):
        super().__init__(config)
        c = config.cbpe
        self.sizes = tuple(c["sizes"])
        self.core = FactoredQwen3(config.to_dict(), self.sizes, c["order"], dtype)

    # ------------------------------------------------------------- loading
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, config=None, dtype=None, torch_dtype=None,
                        device_map=None, revision=None, token=None, **kwargs):
        path = pretrained_model_name_or_path
        if not os.path.isdir(path):
            from huggingface_hub import snapshot_download
            path = snapshot_download(path, revision=revision, token=token, allow_patterns=["config.json", "*.safetensors"])
        if config is None:
            config = CBPEQwen3Config.from_pretrained(path)
        dtype = dtype if dtype is not None else torch_dtype
        dtype = _DTYPES.get(dtype, dtype) if isinstance(dtype, str) else dtype
        if dtype is None or dtype == "auto":
            dtype = torch.bfloat16
        model = cls(config, dtype)
        model.core.load_weights(os.path.join(path, "model.safetensors"))
        if device_map in (None, "auto"):
            device_map = "cuda" if torch.cuda.is_available() else "cpu"
        if not isinstance(device_map, (str, torch.device, int)):
            raise ValueError("device_map: a device (\"cuda\", \"cpu\", ...) only")
        return model.to(device_map).eval()

    def save_pretrained(self, *args, **kwargs):
        raise NotImplementedError("this model is loaded from its original files; copy them instead")

    # --------------------------------------------------------------- ids
    def unpack(self, ids):
        """packed ids [...] -> the four fields [..., 4] (variation, prefix, core, suffix)"""
        _, np_, nc, ns = self.sizes
        s = ids % ns
        ids = ids // ns
        c = ids % nc
        ids = ids // nc
        return torch.stack([ids // np_, ids % np_, c, s], -1)

    def pack(self, fields):
        _, np_, nc, ns = self.sizes
        return ((fields[..., 0] * np_ + fields[..., 1]) * nc + fields[..., 2]) * ns + fields[..., 3]

    # ------------------------------------------------------------ forward
    def token_nll(self, input_ids):
        """negative log-likelihood (nats) of tokens 1.. given the ones before: [B, T - 1, 4] (variation,
        prefix, core, suffix); a token's NLL is the sum of the last dimension"""
        ids = self.unpack(input_ids)
        h = self.core.hidden(ids[:, :-1])
        return self.core.nll_fields(h, ids[:, 1:])

    def forward(self, input_ids, attention_mask=None, labels=None, past_key_values=None, use_cache=None,
                output_hidden_states=None, return_dict=True, **kwargs):
        ids = self.unpack(input_ids)
        h = self.core.hidden(ids, past_key_values if use_cache else None)
        loss = None
        if labels is not None:
            valid = labels >= 0
            y = self.unpack(labels.clamp(min=0))
            nll = self.core.nll_fields(h[:, :-1], y[:, 1:]).sum(-1)
            valid = valid[:, 1:] if attention_mask is None else valid[:, 1:] & attention_mask[:, 1:].bool()
            loss = (nll * valid).sum() / valid.sum().clamp(min=1)
        return CausalLMOutputWithPast(loss=loss, logits=None, past_key_values=past_key_values if use_cache else None,
                                      hidden_states=(h,) if output_hidden_states else None)

    # ----------------------------------------------------------- generate
    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=64, do_sample=False, temperature=1.0,
                 top_p=0.9, tokenizer=None, heal=None, seed=0, n_cand=64, stop_strings=None, **ignored):
        """Continue each row of input_ids [B, T] (packed ids; padding is dropped with attention_mask).
        Returns [B, T'] packed ids: the prompt (without its last token if healed) and the new tokens,
        right-padded with 0. Greedy unless do_sample. Pass `tokenizer` for token healing (the prompt's
        last token is dropped and respelled by the first new token(s), see cbpe_qwen3.generate_tokens;
        on by default when a tokenizer is given, heal=False turns it off) and for stop_strings."""
        heal = tokenizer is not None if heal is None else heal
        rows = []
        for b in range(input_ids.shape[0]):
            row = input_ids[b] if attention_mask is None else input_ids[b][attention_mask[b].bool()]
            kept, new, _ = generate_tokens(self.core, self.unpack(row).tolist(), tokenizer, max_new_tokens,
                                           temperature if do_sample else 0.0, top_p, heal, seed + b, n_cand,
                                           tuple(stop_strings or ()))
            rows.append(self.pack(torch.tensor(kept + new, dtype=torch.long)))
        out = torch.zeros(len(rows), max(len(r) for r in rows), dtype=torch.long)
        for b, r in enumerate(rows):
            out[b, :len(r)] = r
        return out.to(input_ids.device)
