# moe-str-control

Code and evaluation harness for the paper

> **Token-Level Mixture-of-Experts Does Not Improve Scene Text Recognition Under a Matched Computational Budget**

A token-level mixture of experts (MoE) replaces the decoder FFN with a gated mixture of four
experts, and this is often reported as an accuracy gain. This repository contains the controlled
study: five decoder variants trained under one protocol — identical data, schedule, augmentation
and evaluation harness — so that stored capacity, arithmetic per token and routing policy can be
varied one at a time.

Everything needed to retrain the configurations and to reproduce the ten-benchmark evaluation,
the routing statistics and the per-token latency measurements is here. Datasets and checkpoints
are not (see [Data](#data)).

## Headline result

Word accuracy (%), lexicon-free, case-insensitive, mean over ten standard benchmarks
(12,067 test images). Full per-benchmark table in the paper; `best`/`last` are the two
checkpoint rules.

| Config | Decoder FFN | Stored params | Expert arithmetic / token | Mean acc. (best / last) |
|---|---|---|---|---|
| A1 | 4 × 1024 gated mixture | 100 % | 100 % | 88.22 / 88.28 |
| A7a | 1 × 4096 dense | 100 % | 4 × | 88.20 / 88.14 |
| A7b | 1 × 1024 dense | 53 % | 1/4 × | 88.12 / 88.25 |
| A3-top2 | 4 × 1024, hard top-2 | 100 % | 1/2 × | 87.96 / 87.96 |
| A1v3 | 4 × 1024 gated mixture, re-weighted sampler | 100 % | 100 % | 88.12 / 88.26 |

The five configurations span 0.32 points, with paired-bootstrap intervals that contain zero.
A dense FFN matched on parameters (A7a) and one matched on arithmetic (A7b) both reproduce the
mixture, so neither the extra capacity nor the routing itself is converted into accuracy at this
scale.

## Repository layout

```
train.py                  training entry point (one config = one run)
test.py                   ten-benchmark evaluation of a checkpoint, plus routing statistics
evaluate.py               single-set evaluation helper
predict.py                inference on a folder of images
auto_eval.py              watches checkpoints/ and evaluates every new checkpoint
check_train_data.py       sanity check of an LMDB training set

configs/                  one YAML per configuration compared in the paper
models/                   SVTR backbone, MoE decoder, gating/router, experts, sequence head
data/                     LMDB dataset reader, tokenizer (98 classes), transforms
losses/                   cross-entropy + load-balancing auxiliary loss
utils/                    checkpoint I/O, logging, metrics, MoE monitoring

tools/diag_eos.py         EOS-bias sweep on IIIT5k (why predictions run long)
tools/diag_decode.py      greedy vs. teacher-forced agreement
tools/diag_labels.py      label/alphabet consistency checks
tools/check_budget.py     parameter and MAC accounting per configuration
tools/check_sparse.py     verifies that only k experts are evaluated per token
tools/bench_routing.py    per-token latency measurement
tools/build_sample_weights.py  builds the re-weighted sampler used by A1v3
tools/smoke_test.py       one-step forward/backward smoke test
```

## Environment

Tested on Python 3.10 with two Tesla V100-SXM2-32GB (sm_70).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins `torch==2.4.1` / `torchvision==0.19.1` (cu121 wheels), the combination
used for every number in the paper. Other recent versions should work; the reported latencies
are specific to the V100 and to the pinned build.

## Data

Training uses the standard synthetic corpus: MJSynth (7.22 M images) and SynthText (5.52 M),
30 epochs over their union. Evaluation uses the ten standard benchmarks: IIIT5k-3000, SVT, SVTP,
IC03-860, IC03-867, IC13-857, IC13-1015, IC15-1811, IC15-2077 and CUTE80.

Both have to be obtained from their original sources and converted to LMDB. The code expects
this layout, which the configs reference with relative paths:

```
data/
└── data_lmdb_release/
    ├── training/
    │   ├── MJ/MJ_train/          # data.mdb + lock.mdb
    │   └── SynthText/            # data.mdb + lock.mdb
    └── evaluation/
        ├── IIIT5k_3000/  SVT/  SVTP/  CUTE80/
        ├── IC03_860/  IC03_867/  IC13_857/  IC13_1015/
        └── IC15_1811/  IC15_2077/
```

`data/convert_ic15.py` converts raw IC15 crops to the LMDB layout. The vocabulary is the 95
printable ASCII symbols plus `<BOS>`, `<EOS>` and `<PAD>` (98 classes); the harness asserts this
against the checkpoint at load time, because a charset mismatch corrupts decoding while leaving
the loss healthy.

## Reproducing the paper

Each run is one config; a 30-epoch run takes about 50 GPU-hours on a V100.

```bash
# 1. train one configuration (A1 shown; swap the config for the others)
python train.py --config configs/a1_svtr.yaml

# 2. evaluate the ten benchmarks, lexicon-free greedy decoding
python test.py --checkpoint checkpoints/a1_svtr/best.pth

# 3. routing statistics, budget accounting and per-token latency
python tools/check_budget.py --config configs/a1_svtr.yaml
python tools/bench_routing.py --checkpoint checkpoints/a1_svtr/best.pth

# 4. the EOS diagnostic behind the length analysis
python tools/diag_eos.py --checkpoint checkpoints/a1_svtr/best.pth --dataset IIIT5k_3000
```

### Config to paper mapping

| Paper name | Config file | Notes |
|---|---|---|
| A1 | `configs/a1_svtr.yaml` | gated 4-expert mixture, `alpha = 0.01` |
| A7a | `configs/a7_dense_param_matched.yaml` | dense, parameter-matched to A1 |
| A7b | `configs/a7_dense_compute_matched.yaml` | dense, arithmetic-matched to one expert |
| A3-top2 | `configs/a3_top2.yaml` | hard top-2 over the mixture's own experts |
| A3-top1 | `configs/a3_top1.yaml` | hard top-1, latency measurement only |
| A1v3 | `configs/a1_v3.yaml` | A1 plus a sampler re-weighted toward digits and long strings |

`configs/a1_v2.yaml`, `configs/a7a_rerun.yaml` and `configs/a3_top1_rerun.yaml` are kept for the
audit trail: an over-aggressive re-weighting abandoned after 3 epochs, and two runs killed by an
out-of-memory event and redone. None of them is part of the reported comparison.

### Seeds

The paper reports three seeds for A1, A7a and A1v3 (seed 42 is the main run, plus 43 and 44):

```bash
python train.py --config configs/a1_svtr.yaml --seed 43 --out-dir checkpoints/a1_s1
```

Per-seed results are in [RESULTS.md](RESULTS.md). A7b and A3-top2 are single-seed.

## Notes and known limitations

- The model is deliberately small: 6.8 M parameters, a 2-layer decoder and an encoder trained
  from scratch. The paper's claim is scoped to this budget, not to large pretrained models.
- Decoding is greedy with no lexicon and no beam search, at a maximum length of 25 symbols
  (the longest test label has 22), so no test item is truncated.
- A1v3's `best` and `last` checkpoints score identically because the selected epoch is the
  final one; the two files differ only in serialisation metadata.
- Code comments and console output are partly in Chinese, the language the project was
  developed in. Everything a reader needs in order to run it — config files, README, RESULTS
  and every CLI flag — is in English. The per-benchmark lines printed by `test.py` are parsed
  by a regex in `auto_eval.py`, so that output is kept byte-stable and should not be translated.
- Latency numbers come from `tools/bench_routing.py` on a V100-SXM2-32GB with the pinned torch
  build. At `d_ff = 1024` the per-expert matrix products are bandwidth-bound, which is why a
  genuine 4× reduction in expert arithmetic only shortens decoding by 3–5 %.

## License

MIT, see [LICENSE](LICENSE).
