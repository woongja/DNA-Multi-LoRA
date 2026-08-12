"""Score the noise classification module from a prediction file.

`train.py --is_eval` writes one line per utterance:

    <audio_path> <predicted_class>

The ground-truth class is recovered from the filename, which the augmentation
pipeline stamps as `<utt>__<noise>.wav` (clean files carry no suffix).

    python noise_classification/compute_accuracy.py --predictions results/NC_timit.txt
"""

import argparse
import os

import pandas as pd

# Raw augmentation names -> the 9 classes the module is trained on. Pitch shift
# and time stretch are a single "manipulation" class; the two filter variants
# collapse into the band-limited class.
CLASS_MAP = {
    "pitch_shift": "manipulation",
    "time_stretch": "manipulation",
    "high_pass_filter": "band_pass_filter",
    "low_pass_filter": "band_pass_filter",
}


def truth_from_path(path):
    stem = os.path.basename(path).replace(".wav", "")
    raw = stem.split("__")[1] if "__" in stem else "clean"
    return CLASS_MAP.get(raw, raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True, help="prediction file from train.py --is_eval")
    parser.add_argument("--confusion", action="store_true", help="also print the confusion matrix")
    args = parser.parse_args()

    df = pd.read_csv(args.predictions, sep=" ", header=None, names=["path", "pred"])
    df["truth"] = df["path"].map(truth_from_path)

    print(f"utterances : {len(df)}")
    print(f"accuracy   : {100 * (df['truth'] == df['pred']).mean():.2f}%\n")
    print("per-class recall (%)")
    recall = df.groupby("truth").apply(lambda g: 100 * (g["truth"] == g["pred"]).mean())
    print(recall.round(2).to_string())

    if args.confusion:
        print("\nconfusion matrix (rows = truth)")
        print(pd.crosstab(df["truth"], df["pred"]).to_string())


if __name__ == "__main__":
    main()
