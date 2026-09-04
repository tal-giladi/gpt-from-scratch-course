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
    raise NotImplementedError
