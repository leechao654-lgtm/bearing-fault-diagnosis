"""
Paderborn 跨工况域适应对比
==========================
对每个测试工况，比较：
  1. Baseline        : StandardScaler（只用训练集统计）
  2. Domain z-score  : 训练域/测试域分别 z-score（对齐一阶/二阶矩）
  3. CORAL           : 对齐测试域协方差到训练域（域适应）

按轴承留出，多个 seed 取平均。只用 numpy/sklearn，不需要 torch/shap/mlflow。
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def parse_args():
    p = argparse.ArgumentParser(description="Paderborn cross-condition domain adaptation")
    p.add_argument("--repo", default=r"C:\Users\19185\bearing-fault-diagnosis")
    p.add_argument("--set", choices=["minimal", "full"], default="minimal")
    p.add_argument("--bearings", nargs="*", default=None)
    p.add_argument("--per-condition", type=int, default=10)
    p.add_argument("--vibration-only", action="store_true")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--model", choices=["rf", "gbt"], default="rf")
    p.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    p.add_argument("--coral-lambda", type=float, default=1e-2)
    p.add_argument("--out", default=str(Path(__file__).parent / "results_adapt"))
    return p.parse_args()


def make_model(name: str, seed: int):
    if name == "gbt":
        return GradientBoostingClassifier(n_estimators=150, random_state=seed)
    return RandomForestClassifier(n_estimators=300, random_state=seed,
                                  class_weight="balanced", n_jobs=-1)


def zscore_domain(Xtr, Xte):
    mu_tr = Xtr.mean(axis=0); sd_tr = Xtr.std(axis=0)
    mu_te = Xte.mean(axis=0); sd_te = Xte.std(axis=0)
    sd_tr[sd_tr == 0] = 1.0; sd_te[sd_te == 0] = 1.0
    sd_tr[~np.isfinite(sd_tr)] = 1.0; sd_te[~np.isfinite(sd_te)] = 1.0
    return (Xtr - mu_tr) / sd_tr, (Xte - mu_te) / sd_te


def _matrix_sqrt(C, inverse=False, eps=1e-8):
    w, V = np.linalg.eigh(C)
    w = np.clip(w, eps, None)
    if inverse:
        d = 1.0 / np.sqrt(w)
    else:
        d = np.sqrt(w)
    return (V * d) @ V.T


def coral_align(Xtr, Xte, lam: float = 1e-2):
    mu_tr = Xtr.mean(axis=0); mu_te = Xte.mean(axis=0)
    Xtr_c = Xtr - mu_tr
    Xte_c = Xte - mu_te
    d = Xtr.shape[1]
    Ctr = (Xtr_c.T @ Xtr_c) / max(len(Xtr) - 1, 1) + lam * np.eye(d)
    Cte = (Xte_c.T @ Xte_c) / max(len(Xte) - 1, 1) + lam * np.eye(d)
    Cte_inv_sqrt = _matrix_sqrt(Cte, inverse=True)
    Ctr_sqrt = _matrix_sqrt(Ctr, inverse=False)
    Xte_aligned = Xte_c @ Cte_inv_sqrt @ Ctr_sqrt + mu_tr
    return Xtr, Xte_aligned


def evaluate(Xtr, Xte, ytr, yte, model_name, seed):
    clf = make_model(model_name, seed)
    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)
    return accuracy_score(yte, ypred), f1_score(yte, ypred, average="macro"), ypred


def pick_test_bearings(meta, rng):
    chosen = []
    for label in np.unique(meta["label"]):
        candidates = np.unique(meta.loc[meta["label"] == label, "bearing"].values)
        chosen.append(rng.choice(candidates))
    return list(chosen)


def main():
    args = parse_args()
    repo = Path(args.repo)
    if not repo.exists():
        raise SystemExit(f"仓库不存在：{repo}")
    sys.path.insert(0, str(repo))

    from utils.download_dataset import ensure_data, MINIMAL_SET, FULL_SET
    from utils.data_loader import (load_mat_file, calc_characteristic_frequencies,
                                   OPERATING_CONDITIONS, parse_filename)
    from utils.dsp_features import extract_features_from_bearing

    bearings = args.bearings or (FULL_SET if args.set == "full" else MINIMAL_SET)
    mat_dir = repo / "paderborn_data" / "mat" if args.no_download else ensure_data(bearings)

    mat_files = sorted(mat_dir.rglob("*.mat"))
    if args.bearings:
        mat_files = [f for f in mat_files if f.parent.name in set(bearings)]
    grouped = {}
    for f in mat_files:
        try:
            setting, bearing_code, _ = parse_filename(f.name)
        except Exception:
            continue
        grouped.setdefault((bearing_code, setting), []).append(f)
    mat_files = [f for key in sorted(grouped) for f in grouped[key][:args.per_condition]]
    print(f"使用 {len(mat_files)} 个 .mat 文件")

    rows, meta_rows = [], []
    for i, f in enumerate(mat_files, 1):
        try:
            sig = load_mat_file(str(f))
            rpm = OPERATING_CONDITIONS.get(sig.setting, {}).get("speed_rpm", 1500)
            cf = calc_characteristic_frequencies(rpm)
            feats = extract_features_from_bearing(
                sig,
                use_current=not args.vibration_only,
                use_vibration=True,
                characteristic_freqs=cf,
                envelope_band=(500.0, 10000.0),
            )
            rows.append(feats)
            meta_rows.append({"bearing": sig.bearing_code,
                              "setting": sig.setting,
                              "label": sig.label_3class})
            print(f"[{i}/{len(mat_files)}] {f.name}")
        except Exception as e:
            print(f"[skip] {f}: {e}")

    X = pd.DataFrame(rows).fillna(0.0).values
    meta = pd.DataFrame(meta_rows)
    y = meta["label"].values
    settings = sorted(meta["setting"].unique())
    print("\n工况:", dict(Counter(meta["setting"])))
    print(f"特征矩阵：X={X.shape}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = []

    for test_setting in settings:
        other_settings = [s for s in settings if s != test_setting]
        methods = {"Baseline": [], "Domain z-score": [], "CORAL": []}
        last_ypred = {}
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            test_bearings = pick_test_bearings(meta, rng)
            test_mask = (meta["setting"] == test_setting) & (meta["bearing"].isin(test_bearings))
            train_mask = (meta["setting"].isin(other_settings)) & (~meta["bearing"].isin(test_bearings))
            if test_mask.sum() == 0 or train_mask.sum() == 0:
                continue
            Xtr, Xte = X[train_mask], X[test_mask]
            ytr, yte = y[train_mask], y[test_mask]

            # Baseline
            sc = StandardScaler(); Xtr_b = sc.fit_transform(Xtr); Xte_b = sc.transform(Xte)
            acc, f1, ypred = evaluate(Xtr_b, Xte_b, ytr, yte, args.model, seed)
            methods["Baseline"].append((acc, f1)); last_ypred["Baseline"] = (yte, ypred)

            # Domain z-score
            Xtr_z, Xte_z = zscore_domain(Xtr, Xte)
            acc, f1, ypred = evaluate(Xtr_z, Xte_z, ytr, yte, args.model, seed)
            methods["Domain z-score"].append((acc, f1)); last_ypred["Domain z-score"] = (yte, ypred)

            # CORAL
            try:
                Xtr_c, Xte_c = coral_align(Xtr, Xte, args.coral_lambda)
                acc, f1, ypred = evaluate(Xtr_c, Xte_c, ytr, yte, args.model, seed)
                methods["CORAL"].append((acc, f1)); last_ypred["CORAL"] = (yte, ypred)
            except Exception as e:
                print(f"  CORAL failed (seed={seed}): {e}")

        row = {"test_setting": test_setting}
        for name, vals in methods.items():
            if vals:
                row[f"{name}_acc"] = np.mean([v[0] for v in vals])
                row[f"{name}_f1"] = np.mean([v[1] for v in vals])
        summary.append(row)
        print(f"\n测试工况 {test_setting}:")
        for name, vals in methods.items():
            if vals:
                print(f"  {name:16s} Accuracy={np.mean([v[0] for v in vals]):.4f}  F1={np.mean([v[1] for v in vals]):.4f}")

    df = pd.DataFrame(summary)
    df.to_csv(out / "adaptation_summary.csv", index=False)
    print(f"\n结果保存到：{out.resolve()}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
