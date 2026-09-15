"""
SignalScope - train the stacking meta-learner for model/ensemble.py.

True stacking: the 3 ensemble members are frozen level-0 models, and a level-1
L2-regularized (ridge) logistic regression is fitted on their logit(P(AI)).

  1. Score every image with every member (batched, resumable cache).
  2. Stratified split into a meta-train set and a held-out test set.
  3. On meta-train, K-fold CV picks the L2 strength C (LogisticRegressionCV,
     log-loss). A second K-fold pass with that C produces out-of-fold (OOF)
     meta-predictions, so the reported CV metrics and the decision threshold
     never come from a model that saw the row it is scoring.
  4. The final meta-learner is refit on all of meta-train and compared on the
     held-out test set against every member and the majority vote.
  5. Weights are exported as JSON (no pickle, no sklearn needed at inference).

The members must not have been trained on the images used here, otherwise their
scores are in-sample and the meta-learner will over-trust them. The dual-stream
member trains on CIFAKE train/, so stack on the CIFAKE test/ split.

Positive class = AI-generated (label 1).

Data: a CIFAKE split folder containing REAL/ and FAKE/
(birdy654/cifake-real-and-ai-generated-synthetic-images). A balanced sample of
--n images is drawn from it.

Usage:
  python retrain/train_stacker.py --data-dir retrain/data/test --n 4000
  python retrain/train_stacker.py --data-dir retrain/data/test --n 4000 --target-fpr 0.02
"""

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.ensemble import (  # noqa: E402
    ENSEMBLE_MEMBERS, STACKER_PATH, logit_features, prepare_image, score_members_batch,
)
from retrain.eval_majority_vote import CIFAKE_CLASS_IS_AI, load_balanced_sample, metrics, vote  # noqa: E402


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_sources(args):
    """Returns (image paths, labels, decode_fn, fingerprint)."""
    missing = [c for c in CIFAKE_CLASS_IS_AI if not os.path.isdir(os.path.join(args.data_dir, c))]
    if missing:
        raise SystemExit(f"{args.data_dir} is not a CIFAKE split folder: missing {missing}")
    sample = load_balanced_sample(args.data_dir, args.n, args.seed)
    paths = list(sample["path"])
    decode = lambda p: prepare_image(Image.open(p))  # noqa: E731
    key = f"cifake|{os.path.abspath(args.data_dir)}|{args.n}|{args.seed}"
    return paths, sample["is_ai"].to_numpy(), decode, key


def score_all(items, decode, cache, fingerprint, batch, workers):
    """Scores every item with every member, resuming from `cache` (.npz) when it matches."""
    names = [m["id"] for m in ENSEMBLE_MEMBERS]
    fp = hashlib.sha1(fingerprint.encode()).hexdigest()
    P = np.full((len(items), len(names)), np.nan)
    if os.path.exists(cache):
        saved = np.load(cache, allow_pickle=False)
        if str(saved["fingerprint"]) == fp and list(saved["members"]) == names:
            P = saved["P"]
        else:
            print(f"ignoring stale cache {cache} (different data or members)", flush=True)
    done = int((~np.isnan(P).all(axis=1)).sum())
    if done:
        print(f"resuming from {cache}: {done}/{len(items)} images already scored", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        for start in range(done, len(items), batch):
            stop = min(start + batch, len(items))
            P[start:stop] = score_members_batch(list(pool.map(decode, items[start:stop])))
            np.savez(cache, P=P, fingerprint=fp, members=np.array(names))
            if stop % 200 < batch or stop == len(items):
                rate = (time.time() - t0) / (stop - done)
                print(f"  scored {stop}/{len(items)}  {rate:.2f}s/img  "
                      f"eta {rate * (len(items) - stop) / 60:.1f} min", flush=True)
    return P


# ---------------------------------------------------------------------------
# Meta-learner
# ---------------------------------------------------------------------------
def threshold_for_fpr(p_real, target_fpr):
    """Smallest threshold whose false-positive rate on real images is <= target_fpr."""
    return float(np.quantile(p_real, 1.0 - target_fpr, method="higher")) + 1e-9


def export_json(pipe, C, threshold, cv_metrics, test_metrics, baseline, args, n_train, n_test):
    scaler, lr = pipe.named_steps["standardscaler"], pipe.named_steps["logisticregression"]
    return {
        "type": "stacking_logistic_regression",
        "penalty": "l2",
        "C": float(C),
        "features": "logit(clip(P(AI), eps, 1 - eps)) per member, standardized",
        "eps": 1e-6,
        "members": [m["id"] for m in ENSEMBLE_MEMBERS],
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficients": lr.coef_[0].tolist(),
        "intercept": float(lr.intercept_[0]),
        "threshold": float(threshold),
        "threshold_rule": f"OOF FPR <= {args.target_fpr}" if args.target_fpr else "0.5",
        "training": {
            "source": {"dataset": "CIFAKE", "data_dir": args.data_dir, "n": args.n},
            "n_meta_train": n_train,
            "n_test": n_test,
            "cv_folds": args.folds,
            "seed": args.seed,
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "oof_cv_metrics": cv_metrics,
        "test_metrics": test_metrics,
        "test_baselines": baseline,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="CIFAKE split folder containing REAL/ and FAKE/ (use test/)")
    ap.add_argument("--n", type=int, default=4000, help="balanced sample size drawn from --data-dir")
    ap.add_argument("--test-size", type=float, default=0.2, help="held-out fraction for the final comparison")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--Cs", type=float, nargs="+", default=list(np.logspace(-4, 2, 13)),
                    help="L2 strengths to search (smaller C = stronger regularization)")
    ap.add_argument("--target-fpr", type=float, default=None,
                    help="pick the decision threshold on OOF predictions for this FPR (default: 0.5)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache", default="retrain/checkpoints/stacker_member_scores.npz")
    ap.add_argument("--out", default=STACKER_PATH)
    args = ap.parse_args()

    names = [m["id"] for m in ENSEMBLE_MEMBERS]
    items, y, decode, fingerprint = load_sources(args)
    print(f"dataset: {len(y)} images ({int(y.sum())} AI, {int(len(y) - y.sum())} real)", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.cache)), exist_ok=True)
    P = score_all(items, decode, args.cache, fingerprint, args.batch, args.workers)

    missing = np.isnan(P).all(axis=0)
    if missing.any():
        raise SystemExit(f"members failed to load: {[n for n, m in zip(names, missing) if m]} - "
                         "the meta-learner needs every member")
    ok = ~np.isnan(P).any(axis=1)
    if (~ok).any():
        print(f"dropping {int((~ok).sum())} images a member could not score", flush=True)
    P, y = P[ok], y[ok]

    P_tr, P_te, y_tr, y_te = train_test_split(P, y, test_size=args.test_size, stratify=y, random_state=args.seed)
    X_tr, X_te = logit_features(P_tr), logit_features(P_te)
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    # 1) choose C by CV log-loss
    search = make_pipeline(StandardScaler(), LogisticRegressionCV(
        Cs=args.Cs, cv=cv, penalty="l2", solver="lbfgs", scoring="neg_log_loss", max_iter=5000))
    search.fit(X_tr, y_tr)
    C = float(search.named_steps["logisticregressioncv"].C_[0])
    print(f"\nselected C = {C:.4g} (L2 penalty strength 1/C = {1 / C:.4g})")

    # 2) out-of-fold meta-predictions with that C
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=5000))
    p_oof = cross_val_predict(clone(pipe), X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    threshold = threshold_for_fpr(p_oof[y_tr == 0], args.target_fpr) if args.target_fpr else 0.5
    cv_metrics = metrics(y_tr, (p_oof >= threshold).astype(int), p_oof)

    # 3) final meta-learner on all meta-train rows
    pipe.fit(X_tr, y_tr)
    p_te = pipe.predict_proba(X_te)[:, 1]
    test_metrics = metrics(y_te, (p_te >= threshold).astype(int), p_te)

    header = f"{'model':34s} {'acc':>7s} {'prec':>7s} {'recall':>7s} {'f1':>7s} {'auc':>7s} {'fpr':>7s}"

    def show(name, m):
        print(f"{name:34s} {m['accuracy']:.4f} {m['precision']:.4f} {m['recall']:.4f} "
              f"{m['f1']:.4f} {m.get('auc', float('nan')):7.4f} {m['fpr']:.4f}")

    baseline = {n: metrics(y_te, (P_te[:, j] >= 0.5).astype(int), P_te[:, j]) for j, n in enumerate(names)}
    maj_preds, maj_frac = vote(P_te)
    baseline["majority_vote"] = metrics(y_te, maj_preds, maj_frac)

    print(f"\nthreshold = {threshold:.4f}\n\nOUT-OF-FOLD on meta-train ({len(y_tr)} images)\n{header}")
    show("STACKED meta-learner (OOF)", cv_metrics)
    print(f"\nHELD-OUT test ({len(y_te)} images)\n{header}")
    for n in names:
        show(n, baseline[n])
    show("MAJORITY VOTE", baseline["majority_vote"])
    print("-" * len(header))
    show("STACKED meta-learner", test_metrics)

    lr = pipe.named_steps["logisticregression"]
    print("\nmeta-learner weights (on standardized member logits):")
    for n, w in zip(names, lr.coef_[0]):
        print(f"  {n:32s} {w:+.4f}")
    print(f"  {'intercept':32s} {lr.intercept_[0]:+.4f}")

    spec = export_json(pipe, C, threshold, cv_metrics, test_metrics, baseline, args, len(y_tr), len(y_te))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
