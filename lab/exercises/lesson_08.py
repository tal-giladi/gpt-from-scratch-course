# Lesson 08 - Assembling a GPT, and where the parameters went
# Read lessons/module-03/lesson-02.md before filling this in.

from lib.common import train_defs


def predict_param_counts(config):
    """Predict GPT.num_scaling_params() from the config alone - no model.

    Return a dict with exactly these keys:
        wte                   the token embedding table
        value_embeds          all extra embedding tables, on VE layers only
        lm_head               the output matrix
        transformer_matrices  everything inside the blocks, ve_gate included
        scalars               resid_lambdas + x0_lambdas
        total                 the sum of the five
    """
    # TODO: work out head_dim and kv_dim, count the VE layers with
    # train_defs().has_ve, then add up the table above from the lesson.
    n = config.n_embd
    wte = config.n_embd * config.vocab_size
    lm_head = config.n_embd * config.vocab_size
    head_dim = config.n_embd // config.n_head
    kv_dim = config.n_kv_head * head_dim
    ve_layers = [i for i in range(config.n_layer) if train_defs().has_ve(i, config.n_layer)]
    value_embeds = len(ve_layers) * kv_dim * config.vocab_size
    transformer_matrices = config.n_layer * (4 * config.n_embd ** 2 + 4 * n * n + 4 * n * n) + len(ve_layers) * 32 * config.n_kv_head
    scalars= config.n_layer * 2
    total = wte + value_embeds + lm_head + transformer_matrices + scalars
    return {
        "wte": wte,
        "value_embeds": value_embeds,
        "lm_head": lm_head,
        "transformer_matrices": transformer_matrices,
        "scalars": scalars,
        "total": total
    }
    raise NotImplementedError
