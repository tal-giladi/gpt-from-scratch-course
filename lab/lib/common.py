"""The course library.

One job: give every exercise and every check the same handle on the *real* autoresearch
code, at one small, fixed, reproducible scale.

The repo under study is mounted at /autoresearch (override with AR_REPO). Its `prepare.py`
imports normally. Its `train.py` does not: it is a script, so importing it would start a
training run. Everything above its hyperparameter section is pure definitions, so we exec
that prefix and hand back the objects — which means the course always reflects the file as
it is on disk right now, including any edit you have made to it.
"""

import os
import sys
import types
from functools import lru_cache

import torch

AR_REPO = os.environ.get("AR_REPO", "/autoresearch")
if AR_REPO not in sys.path:
    sys.path.insert(0, AR_REPO)

_TRAIN_PY = os.path.join(AR_REPO, "train.py")
_MARKER = "# Hyperparameters (edit these directly"

# The course's fixed toy scale. Small enough that every check runs in seconds on a CPU,
# big enough to be the real thing: 2 layers, 128 dims, 2 heads, 128 tokens of context.
TOY = dict(n_layer=2, n_embd=128, head_dim=64, sequence_len=128, window_pattern="SSSL")


@lru_cache(maxsize=1)
def train_defs() -> types.SimpleNamespace:
    """Every top-level definition in train.py, up to (not including) the hyperparameters.

    Gives you: GPT, GPTConfig, Block, CausalSelfAttention, MLP, norm, apply_rotary_emb,
    has_ve, attention, _causal_window_mask, MuonAdamW, adamw_step_fused, muon_step_fused.
    """
    with open(_TRAIN_PY, encoding="utf-8") as f:
        source = f.read()
    if _MARKER not in source:
        raise RuntimeError(
            f"{_TRAIN_PY} has no hyperparameter section marker; the course expects the "
            "upstream file layout."
        )
    ns: dict = {"__name__": "train_defs", "__file__": _TRAIN_PY}
    exec(compile(source.split(_MARKER)[0], _TRAIN_PY, "exec"), ns)
    return types.SimpleNamespace(**ns)


@lru_cache(maxsize=1)
def tokenizer():
    """The BPE tokenizer prepare.py trained (8192 tokens), from the shared cache."""
    from prepare import Tokenizer

    return Tokenizer.from_directory()


def toy_config(**overrides):
    """A GPTConfig at the course's fixed scale. Pass overrides to vary one thing."""
    defs = train_defs()
    spec = dict(TOY)
    spec.update(overrides)
    head_dim = spec.pop("head_dim")
    n_embd = spec["n_embd"]
    assert n_embd % head_dim == 0, "n_embd must be a whole number of heads"
    n_head = n_embd // head_dim
    return defs.GPTConfig(
        sequence_len=spec["sequence_len"],
        vocab_size=spec.get("vocab_size") or tokenizer().get_vocab_size(),
        n_layer=spec["n_layer"],
        n_head=n_head,
        n_kv_head=n_head,
        n_embd=n_embd,
        window_pattern=spec["window_pattern"],
    )


def build_model(config=None, seed=0):
    """A freshly initialised model at the course's scale, on the CPU, deterministic."""
    defs = train_defs()
    config = toy_config() if config is None else config
    torch.manual_seed(seed)
    model = defs.GPT(config)
    model.init_weights()
    return model


@lru_cache(maxsize=16)
def get_batch(B=2, T=128, split="train", index=0):
    """A real batch of packed token ids: (x, y), each (B, T), int64.

    Deterministic: a fresh dataloader is walked from the start every time and the shards
    are read in a fixed order, so batch `index` is always the same batch. `y` is `x`
    shifted one position left — the next-token target.
    """
    from prepare import make_dataloader

    loader = make_dataloader(tokenizer(), B, T, split)
    for _ in range(index + 1):
        x, y, _epoch = next(loader)
    # The loader hands out views onto one reused buffer; clone so a later call cannot
    # change a batch someone is still holding.
    return x.clone(), y.clone()


def set_seed(seed=0):
    torch.manual_seed(seed)


def repo_file(*parts):
    """Absolute path to a file inside the repo under study."""
    return os.path.join(AR_REPO, *parts)
