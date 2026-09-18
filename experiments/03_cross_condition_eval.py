"""
Paderborn 跨工况泛化评估
=========================
对比两种场景（都按轴承留出，避免同一轴承泄漏）：
  1. Same-condition : 训练/测试同一工况（参考上限）
  2. Cross-condition: 训练 N15_* 工况，测试 N09_M07_F10（真正考验泛化）

对每个测试工况、多个随机轴承划分（默认 3 个 seed）取平均。

只用 numpy/pandas/scipy/sklearn/matplotlib，不需要 torch/shap/mlflow。
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
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix)


def parse_args():
    p = argparse.ArgumentParser(description="Paderborn cross-condition evaluation")
    p.add_argument("--repo", default=r"C:\Users\19185\bearing-fault-diagnosis")
    p.add_argument("--set", choices=["minimal", "full"], default="minimal")
    p.add_argument("--bearings", nargs="*", default=None)
    p.add_argument("--per-condition", type=int, default=10,
                   help="每个轴承、每个工况取 N 个文件")
    p.add_argument("--vibration-only", action="store_true")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--model", choices=["rf", "gbt"], default="rf")
    p.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    p.add_argument("--out", default=str(Path(__file__).parent / "results_cross"))
    return p.parse_args()


def make_model(name: str, seed: int):
    if name == "gbt":
        return GradientBoostingClassifier(n_estimators=150, random_state=seed)
    return RandomForestClassifier(n_estimators=300, random_state=seed,
                                  class_weight="balanced", n_jobs=-1)


def eval_split(X, y, meta, train_settings, test_setting, test_bearings,
               model_name, seed):
    test_mask = (meta["setting"] == test_setting) & (meta["bearing"].isin(test_bearings))
    train_mask = (meta["setting"].isin(train_settings)) & (~meta["bearing"].isin(test_bearings))
    if test_mask.sum() == 0 or train_mask.sum() == 0:
        return None
    Xtr, Xte = X[train_mask], X[test_mask]
    ytr, yte = y[train_mask], y[test_mask]
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)
    clf = make_model(model_name, seed)
    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)
    return {
        "accuracy": accuracy_score(yte, ypred),
        "f1_macro": f1_score(yte, ypred, average="macro"),
        "confusion": confusion_matrix(yte, ypred),
        "y_true": yte,
        "y_pred": ypred,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
    }


def pick_test_bearings(meta, rng):
    """每个类别随机选一个轴承作为测试轴承（共 3 个）。"""
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
    print(f"特征矩阵：X={X.shape}, 类别分布={dict(Counter(y))}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = []

    for test_setting in settings:
        other_settings = [s for s in settings if s != test_setting]
        same_scores, cross_scores = [], []
        last_eval = None
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            test_bearings = pick_test_bearings(meta, rng)
            same = eval_split(X, y, meta, [test_setting], test_setting, test_bearings,
                              args.model, seed)
            cross = eval_split(X, y, meta, other_settings, test_setting, test_bearings,
                               args.model, seed)
            if same:
                same_scores.append((same["accuracy"], same["f1_macro"]))
            if cross:
                cross_scores.append((cross["accuracy"], cross["f1_macro"]))
                last_eval = cross
        if same_scores and cross_scores:
            same_acc = np.mean([s[0] for s in same_scores])
            same_f1 = np.mean([s[1] for s in same_scores])
            cross_acc = np.mean([c[0] for c in cross_scores])
            cross_f1 = np.mean([c[1] for c in cross_scores])
            print(f"\n测试工况 {test_setting}:")
            print(f"  Same-condition : Accuracy={same_acc:.4f}  F1={same_f1:.4f}")
            print(f"  Cross-condition: Accuracy={cross_acc:.4f}  F1={cross_f1:.4f}")
            summary.append({
                "test_setting": test_setting,
                "same_accuracy": same_acc, "same_f1": same_f1,
                "cross_accuracy": cross_acc, "cross_f1": cross_f1,
            })
            if last_eval is not None:
                with open(out / f"cross_{test_setting}_report.txt", "w", encoding="utf-8") as fh:
                    fh.write(f"Test setting: {test_setting}\n")
                    fh.write(f"Cross-condition F1={cross_f1:.4f}\n\n")
                    fh.write(classification_report(last_eval["y_true"], last_eval["y_pred"],
                                                   target_names=["Healthy", "OR_damage", "IR_damage"],
                                                   zero_division=0))
                cm = last_eval["confusion"]
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.imshow(cm, cmap="Blues")
                ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
                ax.set_xticklabels(["Healthy", "OR", "IR"])
                ax.set_yticklabels(["Healthy", "OR", "IR"])
                ax.set_xlabel("Predicted"); ax.set_ylabel("True")
                for i in range(3):
                    for j in range(3):
                        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                                color="white" if cm[i, j] > cm.max() / 2 else "black")
                fig.tight_layout()
                fig.savefig(out / f"cross_{test_setting}_confusion.png", dpi=150)

    df = pd.DataFrame(summary)
    df.to_csv(out / "cross_condition_summary.csv", index=False)
    if not df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        xpos = np.arange(len(df))
        ax.bar(xpos - 0.2, df["same_f1"], width=0.4, label="Same-condition")
        ax.bar(xpos + 0.2, df["cross_f1"], width=0.4, label="Cross-condition")
        ax.set_xticks(xpos); ax.set_xticklabels(df["test_setting"], rotation=15)
        ax.set_ylabel("F1-macro"); ax.legend(); fig.tight_layout()
        fig.savefig(out / "cross_condition_f1.png", dpi=150)
    print(f"\n结果保存到：{out.resolve()}")


if __name__ == "__main__":
    main()
