# 引用与许可说明

## 原仓库
本项目基于 [Dingding1996/bearing-fault-diagnosis](https://github.com/Dingding1996/bearing-fault-diagnosis)（MIT License）。
原仓库提供：
- Paderborn 数据集下载脚本；
- `.mat` 信号加载与标签映射；
- DSP 特征提取（时域/频域/WPD/包络）；
- 基础 ML pipeline 与 notebook。

请保留原仓库的 LICENSE 和版权声明。

## 数据集
- Paderborn University KAt-DataCenter
- Zenodo DOI: 10.5281/zenodo.15845309
- 引用：Lessmeier, C., Kimotho, J. K., Zimmer, D., Sextro, W. (2016). Condition Monitoring of Bearing Damage in Electromechanical Drive Systems by Using Motor Current Signals of Electric Motors: A Benchmark Data Set for Data-Driven Classification. PHME 2016.

## 我的扩展
`experiments/` 下的脚本和 `experiments/results/` 下的结果为本人扩展实验，包括：
- 跨工况评估；
- Domain z-score / CORAL 域适应对比；
- 目标域数据预算实验。
