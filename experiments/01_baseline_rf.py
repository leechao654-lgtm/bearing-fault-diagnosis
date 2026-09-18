"""
Paderborn 轴承故障诊断 —— 最小 RandomForest baseline
不依赖 torch / shap / mlflow / xgboost，先跑通数据 → 特征 → 分类 → 混淆矩阵。

用法（在已激活的 venv 里）：
    python baseline_rf.py --repo C:\\Users\\19185\\bearing-fault-diagnosis --set minimal --max-per-bearing 2 --vibration-only
    python baseline_rf.py --repo ... --bearings K001 KA04 KI04 --no-download --max-per-bearing 2

说明：
- 第一次运行会调用仓库自带的 utils/download_dataset.py 下载数据（支持断点续传）；
- 3 个轴承约 0.5GB，MINIMAL_SET 15 个约 2.4GB；
- 需要 Windows 上安装 7-Zip（用于解压 .rar）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut, cross_val_predict


def parse_args():
    p = argparse.ArgumentParser(description="Minimal RF baseline for Paderborn bearing diagnosis")
    p.add_argument("--repo", default=r"C:\Users\19185\bearing-fault-diagnosis",
                   help="仓库根目录")
    p.add_argument("--set", choices=["minimal", "full"], default="minimal",
                   help="minimal=15个真实损伤轴承(~2.4GB)，full=32个(~5GB)")
    p.add_argument("--bearings", nargs="*", default=None,
                   help="自定义轴承列表，例如 K001 KA04 KI04")
    p.add_argument("--max-per-bearing", type=int, default=2,
                   help="每个轴承最多用几个 .mat 文件（快速迭代）")
    p.add_argument("--per-condition", type=int, default=None,
                   help="每个轴承、每个工况各取 N 个文件（推荐 2 或 5）；设置后优先于 --max-per-bearing")
    p.add_argument("--vibration-only", action="store_true",
                   help="只用振动特征（更快）；默认同时用电流+振动")
    p.add_argument("--no-download", action="store_true",
                   help="跳过自动下载（数据已存在时使用）")
    p.add_argument("--env-low", type=float, default=500.0)
    p.add_argument("--env-high", type=float, default=10000.0)
    p.add_argument("--out", default=str(Path(__file__).parent / "results"),
                   help="输出目录")
    return p.parse_args()


def main():
    args = parse_args()
    repo = Path(args.repo)
    if not repo.exists():
        raise SystemExit(f"仓库不存在：{repo}")
    sys.path.insert(0, str(repo))

    # 延迟导入，确保 sys.path 已包含仓库根目录
    from utils.download_dataset import ensure_data, MINIMAL_SET, FULL_SET
    from utils.data_loader import (load_mat_file, calc_characteristic_frequencies,
                                   OPERATING_CONDITIONS, parse_filename)
    from utils.dsp_features import extract_features_from_bearing

    # 轴承集合
    if args.bearings:
        bearings = args.bearings
    elif args.set == "full":
        bearings = FULL_SET
    else:
        bearings = MINIMAL_SET

    # 数据目录
    if args.no_download:
        mat_dir = repo / "paderborn_data" / "mat"
    else:
        mat_dir = ensure_data(bearings)

    # 收集 .mat 文件（mat_dir/<bearing>/*.mat）
    mat_files = sorted(mat_dir.rglob("*.mat"))
    if args.bearings:
        mat_files = [f for f in mat_files if f.parent.name in set(bearings)]
    if args.per_condition:
        # 每个轴承、每个工况各取 N 个文件，保证 4 个工况都覆盖
        grouped = {}
        for f in mat_files:
            try:
                setting, bearing_code, _ = parse_filename(f.name)
            except Exception:
                continue
            grouped.setdefault((bearing_code, setting), []).append(f)
        mat_files = [f for key in sorted(grouped) for f in grouped[key][:args.per_condition]]
    elif args.max_per_bearing:
        grouped = {}
        for f in mat_files:
            grouped.setdefault(f.parent.name, []).append(f)
        mat_files = [f for b in sorted(grouped) for f in grouped[b][:args.max_per_bearing]]

    print(f"\n使用 {len(mat_files)} 个 .mat 文件，来自 {len(set(f.parent.name for f in mat_files))} 个轴承")
    if not mat_files:
        raise SystemExit("没有找到 .mat 文件；请先下载数据，或检查 --repo / --bearings")

    # 特征提取
    rows, labels, groups = [], [], []
    settings_used = []
    for i, f in enumerate(mat_files, 1):
        try:
            sig = load_mat_file(str(f))
            settings_used.append(sig.setting)
            rpm = OPERATING_CONDITIONS.get(sig.setting, {}).get("speed_rpm", 1500)
            cf = calc_characteristic_frequencies(rpm)
            feats = extract_features_from_bearing(
                sig,
                use_current=not args.vibration_only,
                use_vibration=True,
                characteristic_freqs=cf,
                envelope_band=(args.env_low, args.env_high),
            )
            rows.append(feats)
            labels.append(sig.label_3class)
            groups.append(sig.bearing_code)
            print(f"[{i}/{len(mat_files)}] {f.name}  label={sig.label_name}  bearing={sig.bearing_code}")
        except Exception as e:
            print(f"[skip] {f}: {e}")

    if not rows:
        raise SystemExit("特征提取全部失败，检查依赖和数据格式")

    if settings_used:
        from collections import Counter
        print("工况分布:", dict(Counter(settings_used)))
    X = pd.DataFrame(rows).fillna(0.0)
    y = np.array(labels)
    groups = np.array(groups)
    print(f"\n特征矩阵：X={X.shape}, 类别分布={dict(zip(*np.unique(y, return_counts=True)))}")

    # 按轴承分组交叉验证（避免同一轴承同时出现在训练和验证）
    n_groups = len(np.unique(groups))
    min_class = min(np.bincount(y))
    if n_groups >= 6 and min_class >= 3:
        n_splits = min(5, n_groups, min_class)
        cv = StratifiedGroupKFold(n_splits=n_splits)
        cv_name = f"StratifiedGroupKFold({n_splits})"
    else:
        # 冒烟测试：样本/类别太少时，按轴承留一验证（每个轴承轮流当测试集）
        cv = LeaveOneGroupOut()
        cv_name = "LeaveOneGroupOut（冒烟测试）"
    clf = RandomForestClassifier(n_estimators=200, random_state=42,
                                 class_weight="balanced", n_jobs=-1)
    y_pred = cross_val_predict(clf, X, y, cv=cv, groups=groups, method="predict")

    acc = accuracy_score(y, y_pred)
    f1m = f1_score(y, y_pred, average="macro")
    print(f"\n=== 交叉验证结果（按轴承分组：{cv_name}）===")
    print(f"Accuracy = {acc:.4f}")
    print(f"F1-macro = {f1m:.4f}")
    print(classification_report(y, y_pred,
                                target_names=["Healthy", "OR_damage", "IR_damage"],
                                zero_division=0))

    # 用全部数据拟合一次，用于特征重要性
    clf.fit(X, y)
    cm = confusion_matrix(y, y_pred)
    print("混淆矩阵：")
    print(cm)

    # 输出
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    X.to_csv(out / "features.csv", index=False)
    pd.DataFrame({"y_true": y, "y_pred": y_pred}).to_csv(out / "predictions.csv", index=False)
    pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False) \
        .to_csv(out / "feature_importance.csv", header=["importance"])
    with open(out / "classification_report.txt", "w", encoding="utf-8") as fh:
        fh.write(f"Accuracy={acc:.4f}\nF1-macro={f1m:.4f}\n\n")
        fh.write(classification_report(y, y_pred,
                                       target_names=["Healthy", "OR_damage", "IR_damage"],
                                       zero_division=0))

    # 混淆矩阵图
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(["Healthy", "OR", "IR"])
    ax.set_yticklabels(["Healthy", "OR", "IR"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    print(f"\n输出已保存到：{out.resolve()}")
    print("  features.csv / predictions.csv / feature_importance.csv / confusion_matrix.png")


if __name__ == "__main__":
    main()



