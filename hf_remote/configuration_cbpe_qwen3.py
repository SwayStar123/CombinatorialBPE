"""Configuration of Qwen3 models on a factored (variation, prefix, core, suffix) tokenizer: a Qwen3 config
plus `cbpe` = {"sizes": rows of the four tables, "order": the chain head's field order, "tokenizer": path
of the tokenizer json}. See cbpe_qwen3.py (cbpe/qwen3.py of https://github.com/SwayStar123/CombinatorialBPE)."""
from transformers import Qwen3Config


class CBPEQwen3Config(Qwen3Config):
    model_type = "cbpe_qwen3"
