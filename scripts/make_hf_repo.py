"""Assemble a Hugging Face model directory that loads with AutoTokenizer / AutoModelForCausalLM
(trust_remote_code=True) from the files of a converted Qwen3 model:

    python scripts/make_hf_repo.py --out DIR --weights model.safetensors --qwen_config Qwen3-1.7B-Base/config.json \
        --tokenizer results/tokenizers/<name>.json [--order 2,1,0,3]

The tokenizer's table sizes are read from the tokenizer. DIR gets the weights, config.json (the Qwen3 config +
auto_map + the factored vocabulary), tokenizer/, tokenizer_config.json and the code the remote loading needs: the
files of hf_remote/ and copies of cbpe/qwen3.py (as cbpe_qwen3.py) and of the pure-Python tokenizer
(cbpe/unrestricted.py, cbpe/unrestricted_py.py). Without --tokenizer the sizes come from --sizes (tests).
"""
import argparse
import json
import os
import shutil

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CODE = [("hf_remote/configuration_cbpe_qwen3.py", "configuration_cbpe_qwen3.py"),
        ("hf_remote/modeling_cbpe_qwen3.py", "modeling_cbpe_qwen3.py"),
        ("hf_remote/tokenization_cbpe_qwen3.py", "tokenization_cbpe_qwen3.py"),
        ("cbpe/qwen3.py", "cbpe_qwen3.py"),
        ("cbpe/unrestricted.py", "unrestricted.py"),
        ("cbpe/unrestricted_py.py", "unrestricted_py.py")]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", required=True, help="model.safetensors (bf16, names as cbpe/qwen3.py loads them)")
    ap.add_argument("--qwen_config", required=True, help="the base model's config.json")
    ap.add_argument("--tokenizer", help="tokenizer .json (its .bin next to it)")
    ap.add_argument("--sizes", help="v,p,c,s table sizes if there is no --tokenizer")
    ap.add_argument("--order", default="2,1,0,3")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    cfg = json.load(open(a.qwen_config, encoding="utf-8"))
    for k in ("vocab_size", "use_cache", "transformers_version", "architectures", "torch_dtype", "dtype"):
        cfg.pop(k, None)
    tok_rel = None
    if a.tokenizer:
        with open(a.tokenizer, encoding="utf-8") as f:
            t = json.load(f)
        sys_path = os.path.abspath(os.path.join(ROOT))
        import sys
        sys.path.insert(0, sys_path)
        from cbpe.unrestricted import UnrestrictedBPE
        sizes = list(UnrestrictedBPE(os.path.join(os.path.dirname(os.path.abspath(a.tokenizer)), t["model"]), t["alphabet"],
                                     t.get("segmentation", "chunks")).sizes.values())
        os.makedirs(os.path.join(a.out, "tokenizer"), exist_ok=True)
        t["model"] = "cbpe_tokenizer.bin"
        with open(os.path.join(a.out, "tokenizer", "cbpe_tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump(t, f, ensure_ascii=False)
        shutil.copy(os.path.join(os.path.dirname(os.path.abspath(a.tokenizer)), json.load(open(a.tokenizer, encoding="utf-8"))["model"]),
                    os.path.join(a.out, "tokenizer", "cbpe_tokenizer.bin"))
        tok_rel = "tokenizer/cbpe_tokenizer.json"
    else:
        sizes = [int(x) for x in a.sizes.split(",")]
    cfg.update({"model_type": "cbpe_qwen3", "architectures": ["CBPEQwen3ForCausalLM"], "torch_dtype": "bfloat16",
                "auto_map": {"AutoConfig": "configuration_cbpe_qwen3.CBPEQwen3Config",
                             "AutoModelForCausalLM": "modeling_cbpe_qwen3.CBPEQwen3ForCausalLM"},
                "cbpe": {"sizes": sizes, "order": [int(x) for x in a.order.split(",")], "tokenizer": tok_rel}})
    with open(os.path.join(a.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(a.out, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump({"tokenizer_class": "CBPEQwen3Tokenizer",
                   "auto_map": {"AutoTokenizer": ["tokenization_cbpe_qwen3.CBPEQwen3Tokenizer", None]}}, f, indent=2)
    for src, dst in CODE:
        shutil.copy(os.path.join(ROOT, src), os.path.join(a.out, dst))
    if os.path.abspath(a.weights) != os.path.abspath(os.path.join(a.out, "model.safetensors")):
        shutil.copy(a.weights, os.path.join(a.out, "model.safetensors"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
