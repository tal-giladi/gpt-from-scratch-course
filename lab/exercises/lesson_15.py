# Lesson 15 - What a fixed budget actually buys
# Read lessons/module-05/lesson-03.md before filling this in.


def budget_report(model, seconds, flops_per_second):
    """What a time budget buys for this model.

    Returns a dict:
        params             every parameter, embeddings included
        flops_per_token    model.estimate_flops()
        tokens             int(flops_per_second * seconds / flops_per_token)
        tokens_per_param   tokens / params
        chinchilla_params  tokens / 20 - the model size those tokens would
                           have been compute-optimal for
    """
    # TODO: five numbers, plain Python types.
    raise NotImplementedError
