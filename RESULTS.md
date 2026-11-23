# Results

Numbers here are the ten-benchmark macro average (word accuracy, lexicon-free, case-insensitive)
used for the seed replication reported in the paper. Regenerate any row with
`python test.py --checkpoint <path>` and average the ten per-benchmark accuracies it prints.

## Seed replication

Three configurations were retrained with two additional seeds (43 and 44); seed 42 is the main
run. `best` selects the checkpoint on a held-out slice of MJ, `last` is epoch 30. The validation
set never overlaps the ten test benchmarks.

| Config | Seed | best | last |
|---|---|---|---|
| A1 | 42 (main) | 88.22 | 88.28 |
| A1 | 43 | 88.33 | 88.36 |
| A1 | 44 | 88.18 | 88.79 |
| A7a | 42 (main) | 88.20 | 88.14 |
| A7a | 43 | 87.45 | 88.50 |
| A7a | 44 | 87.87 | 88.65 |
| A1v3 | 42 (main) | 88.12 | 88.26 |
| A1v3 | 43 | 88.23 | 88.23 |
| A1v3 | 44 | 88.08 | 88.08 |

Aggregated over the three seeds:

| Config | best (mean ± SD) | last (mean ± SD) |
|---|---|---|
| A1 | 88.24 ± 0.08 | 88.48 ± 0.28 |
| A7a | 87.84 ± 0.37 | 88.43 ± 0.26 |
| A1v3 | 88.15 ± 0.08 | 88.19 ± 0.09 |

Seed-to-seed SD is at most 0.38 points, the same order as the spread between configurations.
Paired per-seed differences (A1 − control, computed from the rounded table above) take both
signs and stay below 0.9 points: A1 − A1v3 = +0.10/+0.10/+0.10 (best) and +0.02/+0.13/+0.71
(last); A1 − A7a = +0.02/+0.88/+0.31 (best) and +0.14/−0.14/+0.14 (last). So the null result is
a statement about effects of about half a point or smaller, not a proof of exact equivalence.

### Reading the spread

- A1v3's `best` and `last` are identical per seed because the selected epoch is the last one
  (the run sits on a 97.9–98.2 % validation plateau from epoch 20 onward). The two checkpoint
  files differ only in serialisation metadata.
- A7a has the largest `best` SD (0.38) because its best epoch wanders between epoch 8 and 28
  across seeds; under the `last` rule the spread drops to 0.26.
- A7b and A3-top2 were not replicated and remain single-seed in the paper.

## Per-benchmark table

The full 5-configuration × 10-benchmark × 2-checkpoint table is in the paper (Table 2). The
per-run logs used to build it are not shipped here; the harness prints exactly the numbers the
table needs, one line per benchmark.

## Routing statistics

Expert usage is strongly non-uniform and stable across benchmarks (shares 38.8 / 26.5 / 25.6 /
9.0 %, variance 0.015) without a measurable accuracy benefit. `test.py` prints the per-benchmark
and global allocation table at the end of an evaluation; `tools/bench_routing.py` reports the
per-token latency that goes with it.
