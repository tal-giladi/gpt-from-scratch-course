"""`bash lab/lab.sh up` runs this: proves the lab can see the repo, the tokenizer, the
data and torch before you spend time on lesson 01.
"""

from _lib import check, done

import os

from lib.common import AR_REPO, build_model, get_batch, tokenizer, train_defs

check(os.path.isfile(os.path.join(AR_REPO, "train.py")), f"repo under study mounted at {AR_REPO}")

defs = train_defs()
check(hasattr(defs, "GPT") and hasattr(defs, "GPTConfig"), "train.py definitions load (GPT, GPTConfig)")

tok = tokenizer()
check(tok.get_vocab_size() == 8192, f"tokenizer loaded, vocab_size={tok.get_vocab_size()}")

x, y = get_batch(B=2, T=128)
check(tuple(x.shape) == (2, 128) and tuple(y.shape) == (2, 128), "a real batch of data arrives, shape (2, 128)")

model = build_model()
loss = model(x, y)
check(6.0 < loss.item() < 12.0, f"an untrained model gives a plausible loss ({loss.item():.3f})")

done("00 (self-test)")
