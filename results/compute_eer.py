"""Recompute every EER reported in the paper from raw score files.

The raw score files (one score per evaluation utterance) are too large to ship
with this repository, so this script regenerates `results/tables/*` from a local
copy of them.

    python results/compute_eer.py \
        --protocols /path/to/protocols \
        --scores    /path/to/results \
        --out       results/tables

Score file format
    <utt_id> <score>                 ConformerTCM (bonafide log-likelihood)
    <utt_id> <spoof> <bonafide>      AASIST-SSL   (last column is used)

Protocol file format
    <utt_id> <subset> <label>        label in {bonafide, spoof}
"""

import argparse
import os

import numpy as np
import pandas as pd

from evaluate_metrics import compute_eer

# ---------------------------------------------------------------- loading

def load_protocol(path, subset="eval"):
    df = pd.read_csv(path, sep=" ", header=None, names=["utt", "subset", "label"])
    if str(df.iloc[0]["utt"]) == "file_path":  # some protocols carry a header row
        df = df.iloc[1:]
    return df[df["subset"] == subset]


def load_scores(path):
    df = pd.read_csv(path, sep=" ", header=None)
    df.columns = ["utt"] + [f"c{i}" for i in range(1, len(df.columns))]
    df["score"] = df.iloc[:, -1].astype(float)  # bonafide score is always last
    return df[["utt", "score"]]


def eer(protocol_path, score_path, subset="eval"):
    """EER (%) of one score file, or None when the pair does not line up."""
    if not (os.path.exists(protocol_path) and os.path.exists(score_path)):
        return None, 0
    merged = pd.merge(load_protocol(protocol_path, subset), load_scores(score_path), on="utt")
    if merged.empty or merged["label"].nunique() < 2:
        return None, len(merged)
    value, _ = compute_eer(
        merged[merged["label"] == "bonafide"]["score"].values,
        merged[merged["label"] == "spoof"]["score"].values,
    )
    return value * 100, len(merged)


# ---------------------------------------------------------------- tables

MODELS = {"aasist": "AASIST-SSL", "conformertcm": "ConformerTCM"}

# Reported in the paper but not recomputable here: the DNA Multi LoRA score
# files were produced inside the training harness and are not part
# of this snapshot. See docs/reproduce.md.
OURS_OVERALL = {
    "aasist": {"asv19": 4.65, "df21": 10.85, "itw": 8.88, "pooled": 7.93, "params": 26.7},
    "conformertcm": {"asv19": 4.37, "df21": 10.30, "itw": 8.13, "pooled": 7.55, "params": 27.2},
}
OURS_PER_DOMAIN = {
    "aasist": {"D1": 2.30, "D2": 0.70, "D3": 1.79, "D4": 0.92, "D5": 4.19, "D6": 3.57, "D7": 4.38},
    "conformertcm": {"D1": 1.42, "D2": 0.87, "D3": 1.94, "D4": 1.18, "D5": 4.57, "D6": 3.26, "D7": 3.40},
}
FT_PARAMS = {"aasist": 317.8, "conformertcm": 319.8}


def table_overall(protocols, scores):
    """Table 2 -- baseline / full fine-tuning / DNA Multi LoRA on three benchmarks."""
    benchmarks = [("asv19", "protocol_asv19.txt"), ("df21", "protocol_df21.txt"),
                  ("itw", "protocol_itw.txt")]
    merge_files = {
        ("aasist", "baseline"): "AASIST-baseline_merge.txt",
        ("conformertcm", "baseline"): "Conformertcm-baseline_merge.txt",
        ("aasist", "finetune"): "FT-seperate/aasist-fineune_merge.txt",
        ("conformertcm", "finetune"): "FT-seperate/conformertcm-finetune_merge.txt",
    }
    rows = []
    for key, name in MODELS.items():
        for method, folder in [("baseline", "final"), ("finetune", "FT-seperate")]:
            row = {"model": name, "method": method,
                   "params_M": "" if method == "baseline" else FT_PARAMS[key]}
            for bench, protocol in benchmarks:
                value, _ = eer(os.path.join(protocols, protocol),
                               os.path.join(scores, folder, f"{key}-{method}-{bench}.txt"))
                row[bench] = None if value is None else round(value, 2)
            value, _ = eer(os.path.join(protocols, "protocol_all.txt"),
                           os.path.join(scores, merge_files[(key, method)]))
            row["pooled"] = None if value is None else round(value, 2)
            rows.append(row)
        rows.append({"model": name, "method": "dna_multi_lora",
                     "params_M": OURS_OVERALL[key]["params"],
                     **{k: OURS_OVERALL[key][k] for k in ["asv19", "df21", "itw", "pooled"]}})
    return pd.DataFrame(rows)[["model", "method", "params_M", "asv19", "df21", "itw", "pooled"]]


# Noise class -> paper domain. D1 groups the two background conditions, D5 groups
# pitch shift and time stretch (stored as the single class "manipulation").
DOMAINS = [("D0", "clean"), ("D1", "background_music_noise"), ("D2", "auto_tune"),
           ("D3", "band_pass_filter"), ("D4", "echo"), ("D5", "manipulation"),
           ("D6", "gaussian_noise"), ("D7", "reverberation")]


def table_per_domain(protocols, scores):
    """Table 4 -- baseline vs noise-specific LoRA on ASV19*, per noise domain."""
    rows = []
    for key, name in MODELS.items():
        tag = "baseline" if key == "conformertcm" else "asv19"  # naming differs per model
        row = {"model": name, "method": "baseline"}
        for domain, noise in DOMAINS:
            value, _ = eer(os.path.join(protocols, "noise_type_asv19", f"{noise}.txt"),
                           os.path.join(scores, "noise", f"{key}-{tag}-{noise}.txt"))
            row[domain] = None if value is None else round(value, 2)
        rows.append(row)
        rows.append({"model": name, "method": "noise_specific_lora", "D0": None,
                     **OURS_PER_DOMAIN[key]})
    return pd.DataFrame(rows)[["model", "method"] + [d for d, _ in DOMAINS]]


def table_sequential_ft(protocols, scores):
    """Figure 2 -- catastrophic forgetting under sequential fine-tuning.

    Full grid: after fine-tuning on domain D<stage>, EER on every earlier domain.
    The curve plotted in the paper is the `eval_domain == stage - 1` diagonal.
    """
    rows = []
    for key, name in MODELS.items():
        for stage in range(1, 8):
            for prev in range(0, stage):
                value, n = eer(os.path.join(protocols, "catastropy", f"g{prev}.txt"),
                               os.path.join(scores, "FT-cata", f"{key}-g{stage}-g{prev}.txt"))
                if value is None:
                    continue
                rows.append({"model": name, "ft_stage": f"D{stage}", "eval_domain": f"D{prev}",
                             "eer": round(value, 2), "n": n,
                             "on_paper_curve": prev == stage - 1})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- output

def write(df, out_dir, stem, note=""):
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"{stem}.csv"), index=False)
    with open(os.path.join(out_dir, f"{stem}.md"), "w") as f:
        if note:
            f.write(note.rstrip() + "\n\n")
        pretty = df.copy()
        for column in pretty.columns:
            if pretty[column].isna().any():
                pretty[column] = pretty[column].map(lambda v: "-" if pd.isna(v) else v)
        f.write(pretty.to_markdown(index=False, floatfmt=".2f") + "\n")
    print(f"wrote {out_dir}/{stem}.{{csv,md}}  ({len(df)} rows)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocols", required=True, help="protocol root directory")
    parser.add_argument("--scores", required=True, help="raw score root directory")
    parser.add_argument("--out", default="results/tables", help="output directory")
    args = parser.parse_args()

    write(table_overall(args.protocols, args.scores), args.out, "table2_overall",
          "# Table 2 -- overall EER (%)\n"
          "`baseline` and `finetune` are recomputed from raw scores; `dna_multi_lora` is\n"
          "quoted from the paper (score files not included in this snapshot).")
    write(table_per_domain(args.protocols, args.scores), args.out, "table4_per_domain",
          "# Table 4 -- EER (%) per noise domain on ASV19*\n"
          "`baseline` is recomputed from raw scores; `noise_specific_lora` is quoted\n"
          "from the paper.")
    write(table_sequential_ft(args.protocols, args.scores), args.out, "fig2_sequential_ft",
          "# Figure 2 -- sequential fine-tuning D1 -> D7\n"
          "Rows with `on_paper_curve = True` are the values plotted in the paper.")


if __name__ == "__main__":
    main()
