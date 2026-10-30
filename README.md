# Optimization Methods（最优化方法）

本仓库用于整理最优化方法课程的作业与实验，包括算法实现、数学分析、实验结果及相关文档。

## 实验作业

| 编号 | 内容 | 目录 | 状态 |
| --- | --- | --- | --- |
| 01 | 凸集、凸函数与数值反例 | [exp01](exp01/) | 已完成，含参数对照与数值验证 |

每次实验使用独立目录，按 `exp01`、`exp02`、`exp03` 等编号组织。实验目录中的 `src` 存放源代码，`doc` 存放题目与参考文档，`results` 存放运行结果，`report` 存放实验报告。

## 实验一

实验使用 Python、NumPy 和 matplotlib。本机已在现有 Anaconda `pg` 环境中验证通过，未修改其依赖。进入实验目录后运行：

```bash
cd exp01
conda run -n pg python src/check_progress.py --labs 01
conda run -n pg python -m unittest discover -s tests -v
conda run -n pg python src/run_studies.py
```

在其他机器上，可使用实验目录中的 `environment.yml` 从南京大学镜像创建独立环境：

```bash
conda env create -f environment.yml
conda activate optimization-methods
python src/run_studies.py
```

也可在已激活的 Python 环境中通过 `python -m pip install --index-url https://mirror.nju.edu.cn/pypi/web/simple -r requirements.txt` 安装对应版本依赖。镜像仅在项目配置或当前命令中指定。

`src/lab01_convexity.py` 运行默认实验，支持 `--output-dir` 指定输出位置。`src/run_studies.py` 完成默认实验、样本量对照、内半径对照、Hessian 扩展及保存数据复核；结果写入当前实验的 `results` 目录。

默认设置为随机种子 `20261001`、样本量 `4000`、内半径 `0.6`、容差 `1e-10`。圆盘与凸二次函数的违例数均为 `0`，圆环成员违例数为 `1398`，鞍函数 Jensen 违例数为 `1957`。抽样用于发现反例，全局凸性结论须结合数学证明。

| 结果文件 | 内容 |
| --- | --- |
| `results/data.csv`、`results/metrics.json` | 默认实验原始数值与指标 |
| `results/sample_comparison.csv` | 样本量 100、1000、10000 的对照 |
| `results/radius_comparison.csv` | 内半径 0、0.3、0.6、0.8 的对照 |
| `results/hessian_comparison.csv` | 正定、半正定与不定矩阵的对照 |
| `results/csv_audit.csv` | 默认数据前三组的独立数值复核 |
| `results/*.png` | 默认实验、参数变化、二次曲面与明确反例的 matplotlib 图形 |
| `results/study_summary.json` | 全部对照指标与运行版本 |

实验一的 7 项自动测试均通过。实验报告与课程文档仅在本地保存。

## 许可证

本仓库的原创内容采用 [MIT 许可证](LICENSE)。
