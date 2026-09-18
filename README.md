# Paderborn 电机轴承故障诊断：跨工况泛化与域适应

> 本项目基于开源项目 [Dingding1996/bearing-fault-diagnosis](https://github.com/Dingding1996/bearing-fault-diagnosis)（MIT License）完成扩展实验。
> 原项目提供数据下载、信号加载、DSP 特征提取和基础 ML 流程；我在此基础上完成了跨工况评估、域适应对比和目标域数据预算实验。

## 一、我的贡献

- 设计并实现**按轴承分组**的跨工况评估，量化同工况与跨工况的性能差距；
- 对比 **Domain z-score** 与 **CORAL** 两种域适应方法；
- 设计**目标域数据预算实验**，量化"需要多少无标签目标数据"；
- 完成结果分析、结论与本文档。

原项目提供：数据下载脚本、`.mat` 信号加载、特征提取（时域/频域/WPD/包络）、基础 ML pipeline。

## 二、数据与任务

- 数据：Paderborn University KAt-DataCenter（Zenodo DOI 10.5281/zenodo.15845309）
- 信号：两相电流 + 振动，采样率 64 kHz
- 轴承：15 个真实损伤轴承（5 健康、5 外圈、5 内圈）
- 工况：N09_M07_F10、N15_M01_F10、N15_M07_F04、N15_M07_F10
- 任务：正常 / 外圈故障 / 内圈故障 三分类

## 三、方法

1. 特征：时域（RMS、峭度等）、频域（谱重心、PSD、主频）、小波包能量、包络（BPFO/BPFI）；
2. 分类器：随机森林（class_weight=balanced）；
3. 验证：按轴承分组交叉验证，避免同一轴承泄漏；
4. 域适应：Domain z-score（训练域/目标域分别标准化）、CORAL（协方差对齐）；
5. 数据预算：仅用 10%/20%/50%/80% 无标签目标数据估计统计量，在剩余目标数据上评估。

## 四、主要结果

### 4.1 同工况 vs 跨工况（F1-macro）

| 测试工况 | Same-condition | Cross-condition |
|---|---|---|
| N09_M07_F10 | 0.894 | **0.318** |
| N15_M01_F10 | 0.954 | **0.466** |
| N15_M07_F04 | 0.920 | 1.000（工况相近） |
| N15_M07_F10 | 0.930 | 0.841 |

### 4.2 域适应对比（跨工况 F1）

| 测试工况 | Baseline | Domain z-score | CORAL |
|---|---|---|---|
| N09_M07_F10 | 0.318 | **0.717** | 0.304 |
| N15_M01_F10 | 0.466 | **0.884** | 0.549 |
| N15_M07_F04 | 1.000 | 0.966 | 0.415 |
| N15_M07_F10 | 0.841 | **0.860** | 0.405 |

### 4.3 目标域数据预算（F1）

| 测试工况 | 0% | 10% | 20% | 50% | 80% |
|---|---|---|---|---|---|
| N09_M07_F10 | 0.335 | **0.542** | **0.558** | 0.545 | 0.532 |
| N15_M01_F10 | 0.410 | **0.768** | **0.785** | 0.801 | 0.756 |
| N15_M07_F04 | **0.878** | 0.804 | 0.829 | 0.749 | 0.838 |
| N15_M07_F10 | **0.785** | 0.807 | 0.752 | 0.754 | 0.735 |

## 五、结论

1. 同工况 F1≈0.90，但跨工况显著退化（最低 0.32），说明工况偏移是主要瓶颈；
2. Domain z-score 对困难跨工况有效，将 F1 从 0.32→0.72、0.47→0.88；
3. CORAL 在本设置下失效：145 维、目标样本少，协方差估计不稳定；
4. 目标域数据预算实验表明：困难工况只需 10–20% 无标签目标数据即可恢复大部分性能；
5. 对相似工况，盲目适配反而有害，应采用基于域偏移的按需适配。

## 六、如何复现

```bash
git clone https://github.com/leechao654-lgtm/bearing-fault-diagnosis.git
cd bearing-fault-diagnosis
python -m venv .venv
# Windows
.venv\Scripts\activate.bat
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

下载数据（冒烟测试 3 个轴承，约 0.5GB）：

```bash
python experiments/00_download_paderborn.py --repo . --bearings K001 KA04 KI04
```

下载最小集（15 个轴承，约 2.4GB）：

```bash
python experiments/00_download_paderborn.py --repo . --minimal
```

运行实验：

```bash
python experiments/01_baseline_rf.py --repo . --set minimal --no-download --per-condition 10
python experiments/03_cross_condition_eval.py --repo . --set minimal --no-download --per-condition 10
python experiments/04_domain_adaptation.py --repo . --set minimal --no-download --per-condition 10
python experiments/05_target_budget.py --repo . --set minimal --no-download --per-condition 10
```

## 七、文件结构

```
experiments/
  00_download_paderborn.py
  01_baseline_rf.py
  02_normalization_experiments.py
  03_cross_condition_eval.py
  04_domain_adaptation.py
  05_target_budget.py
  results/            # CSV + PNG 结果
docs/
  experiment_log.md   # 实验记录
  interview_qa.md     # 面试问答
NOTICE.md
requirements.txt
```

## 八、引用与许可

- 原项目：[Dingding1996/bearing-fault-diagnosis](https://github.com/Dingding1996/bearing-fault-diagnosis)，MIT License；
- 数据集：Lessmeier et al., PHME 2016；Zenodo DOI 10.5281/zenodo.15845309；
- 本仓库扩展代码沿用 MIT License，请保留原项目版权声明和数据集引用。
