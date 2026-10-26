"""AutoTokenizer entry point (trust_remote_code=True) for Qwen3 models on a factored tokenizer: the
unrestricted Combinatorial BPE tokenizer (unrestricted.py) with every token's four fields (variation,
prefix, core, suffix) packed into one integer, so that `input_ids` look like any other model's.

Encoding uses the Rust binary of https://github.com/SwayStar123/CombinatorialBPE (native/dict_search) if
it is found (DS_BIN=path), else a pure-Python encoder with identical output (unrestricted_py.py)."""
import json
import os

import torch
from transformers import BatchEncoding

from .unrestricted import UnrestrictedBPE
from .unrestricted_py import PyEncoder  # noqa: F401  (loaded lazily by unrestricted.py; imported here so the loader copies it)


class CBPEQwen3Tokenizer:
    padding_side = "right"
    pad_token_id = 0  # (the packed id of the token (0, 0, 0, 0); padding is masked by attention_mask)

    def __init__(self, tok: UnrestrictedBPE, name_or_path=""):
        self.tok = tok
        self.name_or_path = name_or_path
        self.sizes = tuple(tok.sizes.values())

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoTokenizer"):
        pass  # (called by AutoTokenizer for remote-code classes)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, revision=None, token=None, **kwargs):
        path = pretrained_model_name_or_path
        if not os.path.isdir(path):
            from huggingface_hub import snapshot_download
            path = snapshot_download(path, revision=revision, token=token, allow_patterns=["config.json", "tokenizer/*"])
        with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
            rel = json.load(f)["cbpe"]["tokenizer"]
        jpath = os.path.join(path, rel)
        with open(jpath, encoding="utf-8") as f:
            d = json.load(f)
        tok = UnrestrictedBPE(os.path.join(os.path.dirname(jpath), d["model"]), d["alphabet"], d.get("segmentation", "chunks"))
        return cls(tok, pretrained_model_name_or_path)

    # ------------------------------------------------------------------ ids
    @property
    def vocab_size(self):
        return sum(self.sizes)

    def __len__(self):
        return self.vocab_size

    def pack(self, t):
        _, np_, nc, ns = self.sizes
        return ((t[0] * np_ + t[1]) * nc + t[2]) * ns + t[3]

    def unpack(self, i):
        _, np_, nc, ns = self.sizes
        i, s = divmod(int(i), ns)
        i, c = divmod(i, nc)
        v, p = divmod(i, np_)
        return v, p, c, s

    def token_bytes(self, t):
        """the text bytes of a token given as a (v, p, c, s) tuple or a packed id"""
        return self.tok.token_bytes(self.unpack(t) if isinstance(t, int) else tuple(t))

    def encode(self, text, add_special_tokens=False, **kwargs):
        return [self.pack(t) for t in self.tok.encode(text)]

    def decode(self, ids, skip_special_tokens=False, **kwargs):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.tok.decode([self.unpack(i) for i in ids])

    def batch_decode(self, sequences, **kwargs):
        return [self.decode(s, **kwargs) for s in sequences]

    def __call__(self, text, return_tensors=None, padding=False, truncation=False, max_length=None,
                 add_special_tokens=False, **kwargs):
        single = isinstance(text, str)
        ids = [self.encode(t) for t in ([text] if single else text)]
        if truncation and max_length:
            ids = [x[:max_length] for x in ids]
        mask = [[1] * len(x) for x in ids]
        if padding and padding != "do_not_pad":
            n = max_length if padding == "max_length" and max_length else max(len(x) for x in ids)
            left = self.padding_side == "left"
            ids = [([self.pad_token_id] * (n - len(x)) + x) if left else (x + [self.pad_token_id] * (n - len(x))) for x in ids]
            mask = [([0] * (n - len(m)) + m) if left else (m + [0] * (n - len(m))) for m in mask]
        data = {"input_ids": ids, "attention_mask": mask}
        if return_tensors is None and single:
            data = {k: v[0] for k, v in data.items()}
        return BatchEncoding(data, tensor_type=return_tensors)
