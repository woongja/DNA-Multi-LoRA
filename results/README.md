# Results

Raw evaluation scores — 146 files, ~1.3 GB — are not committed. What is committed is the
script that turns them into the paper's tables, the tables themselves, and the notebooks
behind the figures.

| Path | Contents |
|:--|:--|
| `compute_eer.py` | recomputes every table from raw score files |
| `evaluate_metrics.py` | DET curve and EER, as used throughout the paper |
| `tables/` | generated tables, CSV and Markdown |
| `figures/` | Figure 2 (sequential fine-tuning) and the Figure 3 t-SNE panels |
| `notebooks/` | per-pair EER and confusion matrices, figure generation, t-SNE |

```bash
python results/compute_eer.py --protocols <protocol_root> --scores <score_root> --out results/tables
```

## Provenance of each number

Everything except the DNA Multi LoRA rows is recomputed from raw scores and matches the
published values exactly. The DNA Multi LoRA scores were produced inside the
Lightning-hydra training tree and are not part of this snapshot, so those rows are quoted
from the paper and marked as such in the table headers and in `compute_eer.py`.

Two details worth knowing when reading `table2_overall`:

- The **pooled** column is a single EER over the concatenated benchmarks, not an average
  of the per-benchmark EERs. The baseline row pools 746,143 utterances (the three
  benchmarks plus ADD2022), the fine-tuning row pools the 636,944 of the three benchmarks
  reported in the table.
- `finetune` is per-domain fine-tuning (`FT-seperate`). Fine-tuning on the pooled
  augmented data instead gives a noticeably worse pooled EER (15.75 / 16.31), which is why
  the per-domain variant is the reference point.

In `fig2_sequential_ft`, the curve plotted in the paper is the `on_paper_curve = True`
subset: after fine-tuning on domain D*s*, the EER on the immediately preceding domain
D*(s−1)*. The CSV keeps the full grid — every fine-tuning stage against every earlier
domain — which shows the same trend with more detail than the figure.
