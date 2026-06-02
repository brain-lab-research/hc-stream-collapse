# Analyzing Stream Collapse in Hyper-Connections: From Diagnosis to Mitigation

Official code release for the paper **"Analyzing Stream Collapse in Hyper-Connections: From Diagnosis to Mitigation"**  
([OpenReview](https://openreview.net/forum?id=Xyrii4O2nw)), accepted as a **poster** at the **ICML 2026 Workshop on Weight-Space Symmetries (WSS)**.

**Authors:** Ekaterina Alimaskina, Gleb Molodtsov, Aleksandr Beznosikov

The implementation builds on [nanoGPT](https://github.com/karpathy/nanoGPT) and follows the mHC / mHC-lite line of Hyper-Connection language models as in [**mHC-lite: You Don't Need 20 Sinkhorn-Knopp Iterations**](https://arxiv.org/abs/2601.05732).

## Abstract

Hyper-Connections (HC) replace the single Transformer residual stream with multiple streams, introducing a permutation symmetry over stream indices. We study how this symmetry is resolved in practice: whether streams specialize in a balanced way or exhibit dominant-stream usage. Using fine-grained diagnostics for HC-based language models, we trace how multi-stream representations are actually used. We find that after an early seeding stage, residual mixing often remains close to identity, limiting a core HC mechanism for exchanging information between streams. Moreover, both signal and interpretable features concentrate in a dominant stream. Thus, the nominally multi-stream residual connection can underutilize its capacity, behaving much closer to a single-stream residual pathway. Finally, we show that breaking symmetry at stream initialization reduces dominant behavior and improves performance across mHC variants.

## Contributions

- We identify a stream-level failure mode in HC-style residuals: models with multiple symmetric streams can rely on one dominant stream.
- We show that collapse arises in mechanics and semantics: residual mixing stays near identity, while read/write signal and representation content concentrate in one stream.
- Using **Learned Stream Scaling (LSS)**—a minimal parameterization at stream expansion—we show that a small controlled symmetry break reduces collapse and improves mHC variants without changing the core HC operator.

## Citation

If you use this code or build on our diagnostics, please cite:

```bibtex
@inproceedings{alimaskina2026stream,
  title     = {Analyzing Stream Collapse in Hyper-Connections: From Diagnosis to Mitigation},
  author    = {Alimaskina, Ekaterina and Molodtsov, Gleb and Beznosikov, Aleksandr},
  booktitle = {ICML 2026 Workshop on Weight-Space Symmetries},
  year      = {2026},
  url       = {https://openreview.net/forum?id=Xyrii4O2nw}
}
```

See also [`CITATION.bib`](CITATION.bib).

## Preparation

Install the required packages:

```sh
pip install -r requirements.txt
```

Or install manually:

```sh
pip install torch numpy transformers datasets tiktoken wandb tqdm einops
```

### Data preparation

To prepare the datasets, enter the corresponding dataset folder and run `prepare.py`:

```sh
cd data/shakespeare_char
python prepare.py

cd ../fineweb_edu
python prepare.py

cd ../openwebtext
python prepare.py

cd ../wikitext
python prepare.py

cd ../c4
python prepare.py
```

Training data preparation typically takes ~30 minutes (depending on your machine and disk speed). C4 tokenization can take longer (~1–2 GB output).

### Benchmark evaluation (WikiText-103 / C4)

For paper-style perplexity on held-out benchmarks, prepare:

```sh
cd data/wikitext && python prepare.py   # val.bin, test.bin
cd ../c4 && python prepare.py         # test.bin (C4 validation split)
```

During training, `train.py` automatically evaluates on available benchmark files every `eval_interval` steps and logs perplexity (`wt_val`, `wt_test`, `c4_test`). Disable with `--eval_benchmarks=False`.

To evaluate a saved checkpoint only:

```sh
python eval_benchmarks.py --out_dir=out-owt-medium-mhc-lite-lss-lss_init_0.6
```

## Training

To train a model, run `train.py`. Use `torchrun` to enable distributed training (see the original nanoGPT project for details). Combine config files to set the dataset, model scale, and method.

### Available config files

* **Model scales**

  * S: `config/small_model.py`
  * M: `config/medium_model.py`
  * L: `config/large_model.py`

* **Methods**

  * HC: `config/with_hc.py`
  * mHC: `config/with_mhc.py`
  * mHC-lite: `config/with_mhc_lite.py`
  * mHC-lite + LSS: `config/with_mhc_lite_lss.py`
  * Residual: (default)

* **Datasets**

  * OpenWebText: `config/train_owt.py`
  * FineWeb-Edu: `config/train_fineweb_edu.py`

### Example

Train a **small (S)** model with **mHC-lite** on **OpenWebText**:

```sh
torchrun --standalone --nproc_per_node=8 train.py \
  config/train_owt.py config/small_model.py config/with_mhc_lite.py
```

Set `--nproc_per_node` to the number of GPUs you have.

Train **mHC-lite + LSS** (symmetry-breaking intervention from the paper):

```sh
torchrun --standalone --nproc_per_node=8 train.py \
  config/train_owt.py config/medium_model.py config/with_mhc_lite_lss.py
```

Optional: override LSS initialization base via `--expand_lss_init=0.6` (default).

You can use `run.sh` as a starting point for batch experiment scripts.

## Inspecting LSS weights

For checkpoints trained with LSS:

```sh
python inspect_lss.py out-owt-medium-mhc-lite-lss-lss_init_0.6/ckpt.pt
```

## Acknowledgements

- This codebase is adapted from [nanoGPT](https://github.com/karpathy/nanoGPT).
- HC / mHC / mHC-lite methodology follows [mHC-lite (arXiv)](https://arxiv.org/abs/2601.05732) and related work; our Hyper-Connection layer design is informed by [hyper-connections](https://github.com/lucidrains/hyper-connections).
- We thank the [mHC reproduction](https://github.com/tokenbender/mHC-manifold-constrained-hyper-connections) project for early inspiration.
