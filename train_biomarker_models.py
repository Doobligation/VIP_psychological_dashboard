#!/usr/bin/env python3
import argparse
import json
import os
import re
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


LABEL_ORDER = ["low", "medium", "high"]


def extract_label(file_path: str):
    stem = Path(file_path).stem
    for token in reversed(re.split(r"[_\s]+", stem)):
        token = token.lower()
        if token in {"low", "medium", "high"}:
            return token
    return None


def choose_signal_col(df: pd.DataFrame, signal: str):
    aliases = {
        "eda": ["eda"],
        "bvp": ["bvp"],
        "temp": ["temp", "temperature"],
    }
    cols = [c for c in df.columns if any(alias in c.lower() for alias in aliases[signal])]
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cols = numeric or cols
    if not cols:
        return None

    best = cols[-1]
    best_score = -1e18

    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 5:
            continue

        diffs = np.diff(s.values)
        sign_changes = np.sum(np.sign(diffs[1:]) != np.sign(diffs[:-1])) if len(diffs) > 1 else 0
        corr = abs(np.corrcoef(np.arange(len(s)), s.values)[0, 1]) if len(s) > 2 else 1.0
        score = sign_changes - 5 * corr + 0.01 * s.nunique()

        if score > best_score:
            best_score = score
            best = col

    return best


def load_and_resample(csv_path: Path):
    label = extract_label(str(csv_path))
    if label is None:
        return None

    df = pd.read_csv(csv_path)
    if "timestamp_local" not in df.columns:
        return None

    eda_col = choose_signal_col(df, "eda")
    bvp_col = choose_signal_col(df, "bvp")
    temp_col = choose_signal_col(df, "temp")

    if eda_col is None or temp_col is None:
        return None

    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(df["timestamp_local"], errors="coerce"),
            "eda": pd.to_numeric(df[eda_col], errors="coerce"),
            "temp": pd.to_numeric(df[temp_col], errors="coerce"),
            "bvp": pd.to_numeric(df[bvp_col], errors="coerce") if bvp_col is not None else np.nan,
        }
    ).dropna(subset=["timestamp"]).sort_values("timestamp")

    out = (
        out.set_index("timestamp")
        .resample("1min")
        .mean(numeric_only=True)
        .dropna(how="all")
        .reset_index()
    )

    out["file"] = csv_path.name
    out["label"] = label
    return out


def build_frustration_dataset(task_dir: Path):
    rows = []
    used_files = []
    for csv_path in sorted(task_dir.glob("*.csv")):
        d = load_and_resample(csv_path)
        if d is None or len(d) < 2:
            continue
        used_files.append(csv_path.name)

        for end in range(1, len(d)):
            prev = d.iloc[end - 1]
            cur = d.iloc[end]
            rows.append(
                {
                    "eda_t_minus_1": prev["eda"],
                    "temp_t_minus_1": prev["temp"],
                    "eda_t_0": cur["eda"],
                    "temp_t_0": cur["temp"],
                    "eda_delta_t_0": cur["eda"] - prev["eda"],
                    "temp_delta_t_0": cur["temp"] - prev["temp"],
                    "minute_index": end,
                    "file": d["file"].iloc[0],
                    "label": d["label"].iloc[0],
                }
            )

    return pd.DataFrame(rows), used_files


def build_mental_demand_dataset(task_dir: Path):
    rows = []
    used_files = []
    for csv_path in sorted(task_dir.glob("*.csv")):
        d = load_and_resample(csv_path)
        if d is None or len(d) < 1:
            continue
        used_files.append(csv_path.name)

        for idx, row in d.iterrows():
            rows.append(
                {
                    "eda": row["eda"],
                    "temp": row["temp"],
                    "minute_index": idx,
                    "file": d["file"].iloc[0],
                    "label": d["label"].iloc[0],
                }
            )

    return pd.DataFrame(rows), used_files


def evaluate_group_cv(df: pd.DataFrame, model):
    X = df.drop(columns=["file", "label"])
    y = df["label"]
    groups = df["file"]

    cv = GroupKFold(n_splits=min(5, groups.nunique()))
    preds = cross_val_predict(model, X, y, cv=cv, groups=groups)

    return {
        "samples": int(len(df)),
        "sessions": int(groups.nunique()),
        "class_counts": {k: int(v) for k, v in y.value_counts().to_dict().items()},
        "accuracy": float(accuracy_score(y, preds)),
        "macro_f1": float(f1_score(y, preds, average="macro")),
        "confusion_matrix": confusion_matrix(y, preds, labels=LABEL_ORDER).tolist(),
        "labels": LABEL_ORDER,
    }


def maybe_unzip_dataset(dataset_path: Path, work_dir: Path):
    if dataset_path.is_dir():
        return dataset_path

    if dataset_path.suffix.lower() != ".zip":
        raise ValueError(f"Expected a folder or .zip file, got: {dataset_path}")

    extract_dir = work_dir / "extracted_dataset"
    if not extract_dir.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dataset_path, "r") as zf:
            zf.extractall(extract_dir)

    nested_root = extract_dir / "Final Files to Use for the Script"
    if nested_root.exists():
        return nested_root

    dirs = [p for p in extract_dir.iterdir() if p.is_dir()]
    if len(dirs) == 1:
        return dirs[0]

    return extract_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to the dataset folder or zip file")
    parser.add_argument("--outdir", required=True, help="Output directory for trained models")
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    dataset_root = maybe_unzip_dataset(dataset_path, outdir)
    frustration_dir = dataset_root / "Frustration"
    mental_demand_dir = dataset_root / "Mental Demand"

    frustration_df, frustration_files = build_frustration_dataset(frustration_dir)
    mental_demand_df, mental_demand_files = build_mental_demand_dataset(mental_demand_dir)

    frustration_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, class_weight="balanced")),
        ]
    )

    mental_demand_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42)),
        ]
    )

    frustration_eval = evaluate_group_cv(frustration_df, frustration_model)
    mental_demand_eval = evaluate_group_cv(mental_demand_df, mental_demand_model)

    frustration_X = frustration_df.drop(columns=["file", "label"])
    frustration_y = frustration_df["label"]
    mental_demand_X = mental_demand_df.drop(columns=["file", "label"])
    mental_demand_y = mental_demand_df["label"]

    frustration_model.fit(frustration_X, frustration_y)
    mental_demand_model.fit(mental_demand_X, mental_demand_y)

    joblib.dump(frustration_model, outdir / "frustration_model.joblib")
    joblib.dump(mental_demand_model, outdir / "mental_demand_model.joblib")

    metadata = {
        "dataset_root": str(dataset_root),
        "notes": [
            "The live integration uses EDA + temperature + minute_index.",
            "Pulse rate is collected by the live script but is not used by the trained models because the uploaded labeled files do not contain a directly matching pulse-rate feature.",
            "Minute index is counted from the beginning of the current live session, so predictions are most meaningful when the stream starts near the beginning of a task/session.",
        ],
        "frustration": {
            "features": list(frustration_X.columns),
            "evaluation": frustration_eval,
            "used_files": frustration_files,
        },
        "mental_demand": {
            "features": list(mental_demand_X.columns),
            "evaluation": mental_demand_eval,
            "used_files": mental_demand_files,
        },
    }

    with open(outdir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
