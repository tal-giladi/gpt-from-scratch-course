# Lesson 08 - reference solution.

from lib.common import train_defs


def predict_param_counts(config):
    has_ve = train_defs().has_ve
    n, V, L = config.n_embd, config.vocab_size, config.n_layer
    head_dim = n // config.n_head
    kv_dim = config.n_kv_head * head_dim
    ve_layers = sum(1 for i in range(L) if has_ve(i, L))

    per_layer = (
        n * n  # c_q
        + n * kv_dim  # c_k
        + n * kv_dim  # c_v
        + n * n  # attn c_proj
        + 4 * n * n  # mlp c_fc
        + 4 * n * n  # mlp c_proj
    )
    counts = {
        "wte": V * n,
        "value_embeds": ve_layers * V * kv_dim,
        "lm_head": V * n,
        "transformer_matrices": L * per_layer + ve_layers * 32 * config.n_kv_head,
        "scalars": 2 * L,
    }
    counts["total"] = sum(counts.values())
    return counts
