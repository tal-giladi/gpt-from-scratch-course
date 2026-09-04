from _lib import check, done, fail, stub_guard

from exercises.lesson_08 import predict_param_counts
from lib.common import build_model, toy_config

predict_param_counts = stub_guard(predict_param_counts, "predict_param_counts")

CONFIGS = [
    ("course scale", toy_config()),
    ("deeper", toy_config(n_layer=5)),
    ("wider", toy_config(n_embd=384)),
    ("odd depth, wide", toy_config(n_layer=3, n_embd=256, sequence_len=64)),
]

for name, config in CONFIGS:
    got = predict_param_counts(config)
    if not isinstance(got, dict):
        fail("predict_param_counts must return a dict")
    want = build_model(config).num_scaling_params()
    for key in ("wte", "value_embeds", "lm_head", "transformer_matrices", "scalars", "total"):
        if key not in got:
            fail(f"missing key {key!r} in the returned dict")
        check(got[key] == want[key], f"{name}: {key} == {want[key]:,}")

# The total must also equal a plain count of every parameter tensor in the model.
model = build_model()
real_total = sum(p.numel() for p in model.parameters())
check(predict_param_counts(model.config)["total"] == real_total,
      f"total matches sum(p.numel() for p in model.parameters()) == {real_total:,}")

# And the claim the lesson makes about small models: embeddings dominate.
counts = predict_param_counts(toy_config())
embedding_share = (counts["wte"] + counts["lm_head"] + counts["value_embeds"]) / counts["total"]
check(embedding_share > 0.8, f"at course scale the embedding tables are {embedding_share:.0%} of the model")

done("08")
