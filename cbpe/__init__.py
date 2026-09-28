import os as _os
import tempfile as _tempfile

# temp files (trainer inputs of several GB, encoder scratch) go to CBPE_TMP, by default tmp/ in the
# repo, not the system temp dir (the system drive may be small)
_tempfile.tempdir = _os.environ.get("CBPE_TMP") or _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "tmp")
_os.makedirs(_tempfile.tempdir, exist_ok=True)

from .bpe import CharBPE
from .tokenizers import VARIATIONS, CombinatorialBPE, StandardBPE, load

__all__ = ["CharBPE", "CombinatorialBPE", "StandardBPE", "VARIATIONS", "load"]
