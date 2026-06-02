"""Tokenize C4 English validation split with GPT-2 BPE (tiktoken).

The paper reports C4 test perplexity; on HuggingFace C4 exposes train/validation only,
so we tokenize the validation split into test.bin (standard LM eval proxy).
"""
import os

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

enc = tiktoken.get_encoding("gpt2")


if __name__ == "__main__":
    out_dir = os.path.dirname(__file__)
    out_path = os.path.join(out_dir, "test.bin")
    dtype = np.uint16

    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    with open(out_path, "wb") as f:
        for example in tqdm(ds, desc="writing test.bin"):
            text = example["text"]
            if not text or not text.strip():
                continue
            ids = enc.encode_ordinary(text)
            ids.append(enc.eot_token)
            f.write(np.array(ids, dtype=dtype).tobytes())

    nbytes = os.path.getsize(out_path)
    print(f"wrote {out_path} ({nbytes / 1e9:.2f} GB)")

    meta = {"vocab_size": enc.n_vocab, "tokenizer": "gpt2", "split": "validation"}
    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        import pickle
        pickle.dump(meta, f)
