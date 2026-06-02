"""Tokenize WikiText-103 validation/test splits with GPT-2 BPE (tiktoken)."""
import os

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

enc = tiktoken.get_encoding("gpt2")


def write_split(texts, filename):
    dtype = np.uint16
    with open(filename, "wb") as f:
        for text in tqdm(texts, desc=f"writing {os.path.basename(filename)}"):
            if not text or not text.strip():
                continue
            ids = enc.encode_ordinary(text)
            ids.append(enc.eot_token)
            f.write(np.array(ids, dtype=dtype).tobytes())
    nbytes = os.path.getsize(filename)
    print(f"wrote {filename} ({nbytes / 1e6:.1f} MB)")


if __name__ == "__main__":
    out_dir = os.path.dirname(__file__)
    ds = load_dataset("wikitext", "wikitext-103-v1")

    write_split(ds["validation"]["text"], os.path.join(out_dir, "val.bin"))
    write_split(ds["test"]["text"], os.path.join(out_dir, "test.bin"))

    meta = {"vocab_size": enc.n_vocab, "tokenizer": "gpt2"}
    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        import pickle
        pickle.dump(meta, f)
