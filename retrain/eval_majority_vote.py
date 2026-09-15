"""
SignalScope - quick majority-vote evaluation (no training).

Runs the 5 frozen ensemble members from model/ensemble.py over a balanced test
sample and reports accuracy / precision / recall / F1 for every member and for
the hard majority vote. Positive class = AI-generated.

Test data: CIFAKE *test* split (birdy654/cifake-real-and-ai-generated-synthetic-images).
  test/FAKE -> AI-generated (1)
  test/REAL -> real (0)
CIFAKE images are 32x32; each member upsamples them with its own preprocessing.

Scores are batched, image decoding runs in threads, and per-image member
probabilities are saved after every batch so an interrupted run resumes.

Usage:
  python retrain/eval_majority_vote.py --data-dir retrain/data/test --n 1000
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.ensemble import ENSEMBLE_MEMBERS, majority_vote, prepare_image, score_members_batch  # noqa: E402

CIFAKE_CLASS_IS_AI = {"REAL": 0, "FAKE": 1}


def load_balanced_sample(data_dir, n, seed):
    rows = [(os.path.join(data_dir, cls, f), is_ai)
            for cls, is_ai in CIFAKE_CLASS_IS_AI.items()
            for f in sorted(os.listdir(os.path.join(data_dir, cls)))]
    df = pd.DataFrame(rows, columns=["path", "is_ai"])
    per_class = n // 2
    counts = df["is_ai"].value_counts()
    if counts.min() < per_class:
        raise SystemExit(f"Not enough images per class for n={n}: {counts.to_dict()}")
    return pd.concat([
        df[df["is_ai"] == 1].sample(per_class, random_state=seed),
        df[df["is_ai"] == 0].sample(per_class, random_state=seed),
    ]).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def metrics(y, preds, scores=None):
    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
    m = {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "macro_f1": float(f1_score(y, preds, average="macro", zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    if scores is not None and len(set(y)) == 2:
        m["auc"] = float(roc_auc_score(y, scores))
    return m


def vote(P):
    preds, fracs = [], []
    for row in P:
        valid = [p for p in row if not np.isnan(p)]
        is_ai, votes, n = majority_vote(valid)
        preds.append(int(is_ai))
        fracs.append(votes / max(1, n))
    return np.array(preds), np.array(fracs)


def decode(path):
    return prepare_image(Image.open(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="CIFAKE test split folder containing REAL/ and FAKE/")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="retrain/majority_vote_results.json")
    args = ap.parse_args()

    sample = load_balanced_sample(args.data_dir, args.n, args.seed)
    y = sample["is_ai"].to_numpy()
    names = [m["id"] for m in ENSEMBLE_MEMBERS]
    print(f"test set: {len(sample)} images ({int(y.sum())} AI, {int(len(y) - y.sum())} real)", flush=True)

    cache = os.path.splitext(args.out)[0] + f"_n{args.n}_s{args.seed}_scores.npy"
    P = np.load(cache) if os.path.exists(cache) else np.full((len(sample), len(names)), np.nan)
    done = int((~np.isnan(P).all(axis=1)).sum())
    if done:
        print(f"resuming from {cache}: {done} images already scored", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        for start in range(done, len(sample), args.batch):
            stop = min(start + args.batch, len(sample))
            prepared = list(pool.map(decode, sample["path"].iloc[start:stop]))
            P[start:stop] = score_members_batch(prepared)
            np.save(cache, P)

            scored = stop - done
            rate = (time.time() - t0) / scored
            if stop % 100 < args.batch or stop == len(sample):
                preds, _ = vote(P[:stop])
                m = metrics(y[:stop], preds)
                print(f"  {stop}/{len(sample)}  {rate:.2f}s/img  eta {rate * (len(sample) - stop) / 60:.1f} min  |  "
                      f"running majority: acc={m['accuracy']:.3f} prec={m['precision']:.3f} "
                      f"rec={m['recall']:.3f} f1={m['f1']:.3f}", flush=True)

    results = {"members": {}}
    print(f"\n{'model':28s} {'acc':>7s} {'prec':>7s} {'recall':>7s} {'f1':>7s} {'auc':>7s} {'fpr':>7s}")

    def show(name, m):
        auc = f"{m['auc']:.4f}" if "auc" in m else "   -  "
        print(f"{name:28s} {m['accuracy']:.4f} {m['precision']:.4f} {m['recall']:.4f} "
              f"{m['f1']:.4f} {auc:>7s} {m['fpr']:.4f}")

    for j, name in enumerate(names):
        ok = ~np.isnan(P[:, j])
        if not ok.any():
            print(f"{name:28s} failed to load - excluded from vote")
            continue
        m = metrics(y[ok], (P[ok, j] >= 0.5).astype(int), P[ok, j])
        m["n_scored"] = int(ok.sum())
        results["members"][name] = m
        show(name, m)

    preds, vote_frac = vote(P)
    results["majority_vote"] = metrics(y, preds, vote_frac)
    print("-" * 74)
    show("MAJORITY VOTE", results["majority_vote"])

    results.update({
        "dataset": "CIFAKE (test split)",
        "n_images": int(len(sample)),
        "seed": args.seed,
        "positive_class": "AI-generated",
        "tie_rule": "even split resolves to real",
    })
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(P, columns=names).assign(label_is_ai=y, majority_pred=preds).to_csv(
        os.path.splitext(args.out)[0] + "_per_image.csv", index=False)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
