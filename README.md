# Optimization Methods（最优化方法）

本仓库用于整理最优化方法课程的作业与实验，包括算法实现、数学分析、实验结果及相关文档。

## 实验作业

| 编号 | 内容 | 目录 | 状态 |
| --- | --- | --- | --- |
| 01 | 凸集、凸函数与数值反例 | [exp01](exp01/) | 初始代码与实验说明已归档，待完成 |

每次实验使用独立目录，按 `exp01`、`exp02`、`exp03` 等编号组织。实验目录中的 `src` 存放源代码，`doc` 存放题目与参考文档，`results` 存放运行结果，`report` 存放实验报告。

## 实验一

在仓库根目录打开终端，进入实验目录后管理依赖和运行代码：

```bash
cd exp01
python -m pip install -r requirements.txt
python src/check_progress.py --labs 01
```

当前代码保留三个待补全函数；进度检查发现未完成项时退出码为 `2`，不执行数值实验。

完成代码后运行：

```bash
python src/lab01_convexity.py
```

默认将数据、图像与指标写入当前实验的 `results` 目录，也可通过 `--output-dir` 指定输出位置。

## 许可证

本仓库的原创内容采用 [MIT 许可证](LICENSE)。
