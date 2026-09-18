"""
目标域数据预算实验（无监督域适应）
==================================
问题：Domain z-score 需要目标域的均值/方差。实际部署时，需要多少目标域"无标签"数据？
做法：
  - 只用一部分目标域样本（无标签）估计均值/方差；
  - 在剩余目标域样本上评估；
  - 扫描 0%、10%、20%、50%、80%；
  - 每个设置用多个 seed 取平均。

只用 numpy/sklearn/matplotlib。
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
from sklearn.metrics import accuracy_score, f1_score


def parse_args():
    p = argparse.ArgumentParser(description="Target-domain data budget for domain z-score")
    p.add_argument("--repo", default=r"C:\Users\19185\bearing-fault-diagnosis")
    p.add_argument("--set", choices=["minimal", "full"], default="minimal")
    p.add_argument("--bearings", nargs="*", default=None)
    p.add_argument("--per-condition", type=int, default=10)
    p.add_argument("--vibration-only", action="store_true")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--model", choices=["rf", "gbt"], default="rf")
    p.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44, 45, 46])
    p.add_argument("--fractions", nargs="*", type=float, default=[0.0, 0.1, 0.2, 0.5, 0.8])
    p.add_argument("--out", default=str(Path(__file__).parent / "results_budget"))
    return p.parse_args()


def make_model(name, seed):
    if name == "gbt":
        return GradientBoostingClassifier(n_estimators=150, random_state=seed)
    return RandomForestClassifier(n_estimators=300, random_state=seed,
                                  class_weight="balanced", n_jobs=-1)


def pick_test_bearings(meta, rng):
    chosen = []
    for label in np.unique(meta["label"]):
        cand = np.unique(meta.loc[meta["label"] == label, "bearing"].values)
        chosen.append(rng.choice(cand))
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
                sig, use_current=not args.vibration_only, use_vibration=True,
                characteristic_freqs=cf, envelope_band=(500.0, 10000.0))
            rows.append(feats)
            meta_rows.append({"bearing": sig.bearing_code, "setting": sig.setting,
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

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    summary = []

    for test_setting in settings:
        other = [s for s in settings if s != test_setting]
        for frac in args.fractions:
            scores = []
            for seed in args.seeds:
                rng = np.random.default_rng(seed)
                test_bearings = pick_test_bearings(meta, rng)
                test_mask = (meta["setting"] == test_setting) & (meta["bearing"].isin(test_bearings))
                train_mask = (meta["setting"].isin(other)) & (~meta["bearing"].isin(test_bearings))
                if test_mask.sum() == 0 or train_mask.sum() == 0:
                    continue
                Xs, ys = X[train_mask], y[train_mask]
                Xt, yt = X[test_mask], y[test_mask]
                idx = np.arange(len(Xt))
                rng.shuffle(idx)

                sc_s = StandardScaler().fit(Xs)
                Xs_s = sc_s.transform(Xs)

                if frac <= 0:
                    Xt_eval = sc_s.transform(Xt)
                    y_eval = yt
                else:
                    n_adapt = max(5, int(len(Xt) * frac))
                    n_adapt = min(n_adapt, len(Xt) - 5)  # 至少留 5 个评估样本
                    adapt_idx = idx[:n_adapt]
                    eval_idx = idx[n_adapt:]
                    mu_t = Xt[adapt_idx].mean(axis=0)
                    sd_t = Xt[adapt_idx].std(axis=0)
                    sd_t[sd_t == 0] = 1.0
                    sd_t[~np.isfinite(sd_t)] = 1.0
                    Xt_eval = (Xt[eval_idx] - mu_t) / sd_t
                    y_eval = yt[eval_idx]

                clf = make_model(args.model, seed)
                clf.fit(Xs_s, ys)
                ypred = clf.predict(Xt_eval)
                scores.append((accuracy_score(y_eval, ypred),
                               f1_score(y_eval, ypred, average="macro")))

            if scores:
                summary.append({
                    "test_setting": test_setting,
                    "target_fraction": frac,
                    "n_adapt_approx": int(round(frac * args.per_condition * 5 * 3)),  # 粗估
                    "accuracy": float(np.mean([s[0] for s in scores])),
                    "f1_macro": float(np.mean([s[1] for s in scores])),
                })
                print(f"{test_setting:16s} frac={frac:4.2f}  "
                      f"F1={np.mean([s[1] for s in scores]):.4f}")

    df = pd.DataFrame(summary)
    df.to_csv(out / "target_budget_summary.csv", index=False)

    if not df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        for setting, g in df.groupby("test_setting"):
            g = g.sort_values("target_fraction")
            ax.plot(g["target_fraction"], g["f1_macro"], marker="o", label=setting)
        ax.set_xlabel("Fraction of target-domain unlabeled data used for normalization")
        ax.set_ylabel("F1-macro on held-out target data")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "target_budget_f1.png", dpi=150)

    print(f"\n结果保存到：{out.resolve()}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
