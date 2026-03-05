#!/usr/bin/env python3
"""
Run NAffinity classifier on a single complex folder and write:
  naffinity_predicted_binding_class.txt

Conventions:
- Expects naffinity.joblib to be in the SAME directory as this script.
- Input is a directory whose basename is the complex name (folder_name).
- The directory must contain:
    descriptors.txt
    rdkit.txt
    receptor_descriptors.txt
    electro_hydro.txt

Usage:
  python3 naffinity_predict.py /path/to/complex_folder
  python3 naffinity_predict.py /path/to/complex_folder --model /path/to/naffinity.joblib
"""

import argparse
import os
import sys
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# -------------------------------------------------------------------
# Custom transformers must exist at import time so joblib can unpickle
# the trained pipeline. Keep class names identical to training.
# -------------------------------------------------------------------
class NumericCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, clip_to_float32=True):
        self.clip_to_float32 = clip_to_float32

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if self.clip_to_float32:
            f32max = np.finfo(np.float32).max
            X = X.clip(lower=-f32max, upper=f32max)
        return X


class DropConstantColumns(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.keep_cols_ = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        nunique = X.nunique(dropna=False)
        self.keep_cols_ = nunique[nunique > 1].index.tolist()
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X.reindex(columns=self.keep_cols_, fill_value=0.0)


class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.8, method="pearson"):
        self.threshold = float(threshold)
        self.method = method
        self.keep_cols_ = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        if X.shape[1] <= 1:
            self.keep_cols_ = X.columns.tolist()
            return self

        corr = X.corr(method=self.method).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if (upper[c] > self.threshold).any()]
        self.keep_cols_ = [c for c in X.columns if c not in set(to_drop)]
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X.reindex(columns=self.keep_cols_, fill_value=0.0)


def script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def default_model_path() -> str:
    return os.path.join(script_dir(), "naffinity.joblib")


def read_feature_txt(path: str) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip()
            try:
                x = float(v)
                feats[k] = float(x) if np.isfinite(x) else 0.0
            except Exception:
                feats[k] = 0.0
    return feats


def load_features_from_folder(folder: str, feature_files: List[str]) -> pd.DataFrame:
    merged: Dict[str, float] = {}
    missing = []
    for fn in feature_files:
        p = os.path.join(folder, fn)
        if not os.path.exists(p):
            missing.append(fn)
            continue
        merged.update(read_feature_txt(p))

    if missing:
        raise FileNotFoundError(
            "Missing required feature files in folder:\n  - " + "\n  - ".join(missing)
        )

    return pd.DataFrame([merged])


def predict(model, X: pd.DataFrame):
    pred = model.predict(X)
    pred_int = int(pred[0])

    prob_strong = float("nan")
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        classes = list(getattr(model, "classes_", [0, 1]))
        if 1 in classes:
            idx1 = classes.index(1)
            prob_strong = float(proba[0][idx1])
        else:
            prob_strong = float(proba[0][-1])

    return pred_int, prob_strong


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="Complex folder containing feature txt files")
    ap.add_argument("--model", default=None, help="Optional path to naffinity.joblib (defaults to script directory)")
    ap.add_argument(
        "--out",
        default="naffinity_predicted_binding_class.txt",
        help="Output filename written inside the folder",
    )
    ap.add_argument(
        "--feature-files",
        nargs="+",
        default=["descriptors.txt", "rdkit.txt", "receptor_descriptors.txt", "electro_hydro.txt"],
        help="Feature txt files to read/merge",
    )
    args = ap.parse_args()

    folder = os.path.abspath(args.dir)
    if not os.path.isdir(folder):
        print(f"ERROR: Not a directory: {folder}", file=sys.stderr)
        sys.exit(1)

    model_path = os.path.abspath(args.model) if args.model else default_model_path()
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    try:
        X = load_features_from_folder(folder, feature_files=args.feature_files)
    except Exception as e:
        print(f"ERROR loading features: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"ERROR loading model: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        pred_int, prob_strong = predict(model, X)
    except Exception as e:
        print(f"ERROR during prediction: {e}", file=sys.stderr)
        sys.exit(1)

    pred_label = "Strong binder" if pred_int == 1 else "Weak/moderate binder"

    out_path = os.path.join(folder, args.out)
    with open(out_path, "w") as f:
        f.write(f"PredictedClass: {pred_label}\n")
        if np.isfinite(prob_strong):
            f.write(f"ProbabilityStrongBinder: {prob_strong:.2f}\n")
        else:
            f.write("ProbabilityStrongBinder: NA\n")

    folder_name = os.path.basename(os.path.normpath(folder))
    print(f"✅ Predicted for {folder_name}")
    print(f"  Class: {pred_label}")
    if np.isfinite(prob_strong):
        print(f"  P(strong): {prob_strong:.2f}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()