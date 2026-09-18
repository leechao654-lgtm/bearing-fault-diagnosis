"""
Paderborn 增强版 baseline：按工况归一化 + 轴频阶次转换 + 特征选择

对比 4 种配置（按轴承分组 5 折交叉验证）：
  1. Raw features                  原始特征
  2. Per-condition norm            按工况 z-score（只用训练折统计）
  3. Norm + shaft-order            再加轴频阶次转换
  4. Norm + shaft-order + top-K    再加 Top-K 特征选择

只用 numpy/pandas/scipy/sklearn/matplotlib，不需要 torch/shap/mlflow/xgboost。

用法（venv 激活后）：
  python baseline_rf_norm.py --repo C:\\Users\\19185\\bearing-fault-diagnosis --set minimal --no-download --per-condition 5
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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix)


def parse_args():
    p = argparse.ArgumentParser(description="Paderborn per-condition normalization baseline")
    p.add_argument("--repo", default=r"C:\Users\19185\bearing-fault-diagnosis")
    p.add_argument("--set", choices=["minimal", "full"], default="minimal")
    p.add_argument("--bearings", nargs="*", default=None)
    p.add_argument("--per-condition", type=int, default=5,
                   help="每个轴承、每个工况取 N 个文件")
    p.add_argument("--vibration-only", action="store_true",
                   help="只用振动特征；默认电流+振动")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--model", choices=["rf", "gbt"], default="rf")
    p.add_argument("--select-k", type=int, default=40)
    p.add_argument("--no-shaft-order", action="store_true",
                   help="不做轴频阶次转换")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(Path(__file__).parent / "results_norm"))
    return p.parse_args()


def make_model(name: str, seed: int):
    if name == "gbt":
        return GradientBoostingClassifier(n_estimators=150, random_state=seed)
    return RandomForestClassifier(n_estimators=300, random_state=seed,
                                  class_weight="balanced", n_jobs=-1)


def normalize_per_condition(Xtr: np.ndarray, set_tr: np.ndarray,
                            Xte: np.ndarray, set_te: np.ndarray):
    """按工况做 z-score，均值/方差只从训练折计算，避免泄漏。"""
    Xtr = Xtr.copy()
    Xte = Xte.copy()
    for s in np.unique(set_tr):
        mtr = set_tr == s
        mte = set_te == s
        if not mte.any():
            continue
        mu = Xtr[mtr].mean(axis=0)
        sd = Xtr[mtr].std(axis=0)
        sd[~np.isfinite(sd)] = 1.0
        sd[sd == 0] = 1.0
        Xtr[mtr] = (Xtr[mtr] - mu) / sd
        Xte[mte] = (Xte[mte] - mu) / sd
    # 测试集出现了训练集没有的工况：用全局训练统计兜底
    unseen = ~np.isin(set_te, np.unique(set_tr))
    if unseen.any():
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0)
        sd[~np.isfinite(sd)] = 1.0
        sd[sd == 0] = 1.0
        Xte[unseen] = (Xte[unseen] - mu) / sd
    return Xtr, Xte


def select_topk(Xtr, ytr, Xte, k, seed):
    """用随机森林重要性选 Top-K 特征（只在训练折上拟合）。"""
    sel = RandomForestClassifier(n_estimators=200, random_state=seed,
                                 class_weight="balanced", n_jobs=-1)
    sel.fit(Xtr, ytr)
    idx = np.argsort(sel.feature_importances_)[::-1][:k]
    return Xtr[:, idx], Xte[:, idx], idx


def run_cv(X, y, groups, settings, args, use_norm: bool, do_select: bool):
    cv = StratifiedGroupKFold(n_splits=args.n_splits)
    y_pred = np.zeros(len(y), dtype=int)
    for tr, te in cv.split(X, y, groups):
        Xtr, Xte = X[tr].copy(), X[te].copy()
        if use_norm:
            Xtr, Xte = normalize_per_condition(Xtr, settings[tr], Xte, settings[te])
        if do_select and args.select_k and args.select_k < Xtr.shape[1]:
            Xtr, Xte, _ = select_topk(Xtr, y[tr], Xte, args.select_k, args.seed)
        clf = make_model(args.model, args.seed)
        clf.fit(Xtr, y[tr])
        y_pred[te] = clf.predict(Xte)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="macro")
    return acc, f1, y_pred


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
    if args.no_download:
        mat_dir = repo / "paderborn_data" / "mat"
    else:
        mat_dir = ensure_data(bearings)

    # 选择文件：每个轴承、每个工况各取 N 个
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
    print(f"使用 {len(mat_files)} 个 .mat 文件，来自 {len(set(f.parent.name for f in mat_files))} 个轴承")

    # 提取特征（只提取一次）
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
            meta_rows.append({
                "bearing": sig.bearing_code,
                "setting": sig.setting,
                "label": sig.label_3class,
                "shaft_freq": rpm / 60.0,
            })
            print(f"[{i}/{len(mat_files)}] {f.name}  {sig.label_name}  {sig.bearing_code}")
        except Exception as e:
            print(f"[skip] {f}: {e}")

    if not rows:
        raise SystemExit("特征提取全部失败")

    X_df = pd.DataFrame(rows).fillna(0.0)
    meta = pd.DataFrame(meta_rows)
    y = meta["label"].values
    groups = meta["bearing"].values
    settings = meta["setting"].values
    sf = meta["shaft_freq"].values
    print("\n工况分布:", dict(Counter(settings)))
    print(f"特征矩阵：X={X_df.shape}, 类别分布={dict(Counter(y))}")

    # 轴频阶次转换：频谱类特征除以轴频（方差除以轴频平方）
    X_shaft = X_df.copy()
    if not args.no_shaft_order:
        for col in X_df.columns:
            if any(k in col for k in ["fd_spectral_centroid", "fd_spectral_std", "fd_peak_frequency"]):
                X_shaft[col] = X_df[col].values / sf
            elif "fd_spectral_variance" in col:
                X_shaft[col] = X_df[col].values / (sf ** 2)

    # 对比配置
    configs = [
        ("Raw features", X_df.values, False, False),
        ("Per-condition norm", X_df.values, True, False),
    ]
    if not args.no_shaft_order:
        configs.append(("Norm + shaft-order", X_shaft.values, True, False))
        configs.append((f"Norm + shaft-order + top{args.select_k}", X_shaft.values, True, True))
    else:
        configs.append((f"Norm + top{args.select_k}", X_df.values, True, True))

    print("\n=== 对比实验（按轴承分组交叉验证）===")
    results = []
    best = None
    for name, Xc, use_norm, do_select in configs:
        acc, f1, y_pred = run_cv(Xc, y, groups, settings, args, use_norm, do_select)
        results.append((name, acc, f1))
        print(f"{name:38s} Accuracy={acc:.4f}  F1-macro={f1:.4f}")
        if best is None or f1 > best[2]:
            best = (name, acc, f1, y_pred)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results, columns=["config", "accuracy", "f1_macro"]).to_csv(
        out / "comparison.csv", index=False)

    name, acc, f1, y_pred = best
    with open(out / "best_classification_report.txt", "w", encoding="utf-8") as fh:
        fh.write(f"Best config: {name}\nAccuracy={acc:.4f}\nF1-macro={f1:.4f}\n\n")
        fh.write(classification_report(y, y_pred,
                                       target_names=["Healthy", "OR_damage", "IR_damage"],
                                       zero_division=0))
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(["Healthy", "OR", "IR"])
    ax.set_yticklabels(["Healthy", "OR", "IR"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "best_confusion_matrix.png", dpi=150)

    # 全量拟合一次，保存特征重要性（用于解释）
    X_imp = X_shaft.values if not args.no_shaft_order else X_df.values
    X_imp_norm = X_imp.copy()
    for s in np.unique(settings):
        m = settings == s
        mu = X_imp[m].mean(axis=0)
        sd = X_imp[m].std(axis=0)
        sd[~np.isfinite(sd)] = 1.0
        sd[sd == 0] = 1.0
        X_imp_norm[m] = (X_imp[m] - mu) / sd
    clf = make_model(args.model, args.seed)
    clf.fit(X_imp_norm, y)
    pd.Series(clf.feature_importances_, index=X_df.columns) \
        .sort_values(ascending=False).to_csv(out / "feature_importance.csv", header=["importance"])

    print(f"\n最佳配置：{name}  Accuracy={acc:.4f}  F1-macro={f1:.4f}")
    print(f"结果已保存到：{out.resolve()}")
    print("  comparison.csv")
    print("  best_classification_report.txt")
    print("  best_confusion_matrix.png")
    print("  feature_importance.csv")


if __name__ == "__main__":
    main()
