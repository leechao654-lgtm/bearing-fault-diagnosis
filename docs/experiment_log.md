# 实验日志（可直接放进 GitHub docs/）

## 实验 1：基础 baseline（3 轴承冒烟测试）
- 命令：`python experiments/01_baseline_rf.py --bearings K001 KA04 KI04 --per-condition 2 --vibration-only`
- 数据：6 个文件，3 个轴承
- 结果：Accuracy 0.00，F1 0.00
- 结论：每类只有 1 个轴承，LeaveOneGroupOut 时训练集看不到该类，指标无意义；仅验证流程可跑通。

## 实验 2：最小集 baseline（15 轴承，每工况 2 文件）
- 命令：`--per-condition 2 --vibration-only`
- 数据：120 个样本，4 个工况
- 结果：Accuracy 0.633，F1-macro 0.635
- 混淆：IR→OR 14、OR→IR 7、Healthy→OR 10
- 结论：覆盖 4 工况后提升；主要错误是 OR/IR 混淆和 Healthy/OR 混淆。

## 实验 3：每工况 5 文件（300 样本）
- 命令：`--per-condition 5`
- 结果：Accuracy 0.743，F1-macro 0.750
- 混淆：Healthy→OR 23、IR→OR 30、OR→IR 18
- 结论：更多数据 + 电流特征提升明显；OR 被过度预测。

## 实验 4：每工况 10 文件（600 样本）
- 命令：`--per-condition 10`
- 结果：Accuracy 0.703，F1-macro 0.712
- 结论：继续加同类文件没有提升，说明瓶颈是工况差异/预处理，不是数据量。

## 实验 5：归一化与特征选择对比
- 命令：`python experiments/02_normalization_experiments.py --per-condition 5`
- 结果：
  - Raw features: F1 0.7374
  - Per-condition norm: F1 0.7296
  - Norm + shaft-order: F1 0.7395
  - Norm + shaft-order + top40: F1 0.7297
- 结论：特征级归一化帮助有限；需要信号级归一化。

## 实验 6：同工况 vs 跨工况
- 命令：`python experiments/03_cross_condition_eval.py --per-condition 10`
- 结果：
  - N09_M07_F10: same 0.894 → cross 0.318
  - N15_M01_F10: same 0.954 → cross 0.466
  - N15_M07_F04: same 0.920 → cross 1.000（工况相近）
  - N15_M07_F10: same 0.930 → cross 0.841
- 结论：域偏移严重，尤其转速变化时。

## 实验 7：域适应对比
- 命令：`python experiments/04_domain_adaptation.py --per-condition 10`
- 结果：
  - N09: Baseline 0.318 → Domain z-score 0.717；CORAL 0.304
  - N15_M01: 0.466 → 0.884；CORAL 0.549
  - N15_M07_F04: 1.000 → 0.966；CORAL 0.415
  - N15_M07_F10: 0.841 → 0.860；CORAL 0.405
- 结论：Domain z-score 对难工况提升显著；CORAL 因高维小样本失效。

## 实验 8：目标域数据预算
- 命令：`python experiments/05_target_budget.py --per-condition 10`
- 结果（F1）：
  - N09: 0.335 → 10% 0.542 → 20% 0.558
  - N15_M01: 0.410 → 10% 0.768 → 20% 0.785
  - N15_M07_F04: 0.878 → 10% 0.804（适配反而变差）
  - N15_M07_F10: 0.785 → 10% 0.807
- 结论：困难工况只需 10–20% 无标签目标数据；相似工况不应盲目适配；应做基于域偏移的按需适配。
