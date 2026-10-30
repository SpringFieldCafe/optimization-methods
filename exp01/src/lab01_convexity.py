"""实验01学生填空版：凸集、凸函数与数值反例。

请独立补全 TODO 01-1 至 TODO 01-3，再运行整份实验。

在 exp01 目录运行：python src/lab01_convexity.py [--output-dir 输出目录]
有限抽样用于寻找反例；本例凸性的证明来自定义和解析 Hessian。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED = 20261001
N_SAMPLES = 4000
Q = np.array([[3.0, 1.0], [1.0, 2.0]])
INNER_RADIUS = 0.6
TOL = 1e-10


def _sample_disk(rng, n, inner_radius=0.0):
    # 面积均匀采样：半径不能直接取 U(0,1)，应对半径平方均匀采样。
    angle = rng.uniform(0.0, 2.0 * np.pi, n)
    radius = np.sqrt(rng.uniform(inner_radius**2, 1.0, n))
    return np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))


def convex_combination(x, y, theta):
    """TODO 01-1：逐组计算二维点的凸组合。

    输入：x、y 为 shape=(N, 2) 的 NumPy 数组；theta 为 shape=(N,)，值在 [0, 1]。
    返回：shape=(N, 2) 的数组，第 i 行是第 i 组端点的凸组合。
    数学提示：z_i = theta_i*x_i + (1-theta_i)*y_i。
    数组提示：为逐行加权，可将 theta 扩展为 shape=(N, 1)。
    """
    raise NotImplementedError("TODO 01-1 convex_combination：补全逐组凸组合")


def _quadratic(x):
    """TODO 01-2：计算已给定矩阵 Q 对应的二次函数值。

    输入：x 为 shape=(N, 2) 的二维点数组；Q 是脚本顶部给定的 shape=(2, 2) 矩阵。
    返回：shape=(N,) 的浮点数组，每个元素是一行点的函数值。
    数学提示：f(x_i) = 0.5*x_i.T@Q@x_i。注意最终按行求和，不要把不同样本混在一起。
    可先计算 x@Q，再与 x 逐元素相乘并按列求和；也可使用 einsum。
    """
    raise NotImplementedError("TODO 01-2 _quadratic：补全二次函数值")


def jensen_residual(function, x, y, theta):
    """TODO 01-3：计算给定函数的 Jensen 残差。

    输入：function 接收 shape=(N, 2) 数组并返回 shape=(N,)；
    x、y 的 shape=(N, 2)，theta 的 shape=(N,)。
    返回：shape=(N,) 的浮点数组，保留残差正负号，不要取绝对值或截断。
    数学提示：先调用 convex_combination 得到 z，再计算函数值的“左边减右边”。
    对凸函数应有 function(z) <= theta*function(x)+(1-theta)*function(y)。
    """
    raise NotImplementedError("TODO 01-3 jensen_residual：补全左边减右边的残差")


def _saddle(x):
    return x[..., 0] ** 2 - x[..., 1] ** 2


def run(output_dir=None) -> dict:
    if not (0 <= INNER_RADIUS < 1) or N_SAMPLES < 1:
        raise RuntimeError("内半径须满足0≤r<1，样本数须为正整数。")
    out = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent.parent / "results"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    theta = rng.uniform(0.0, 1.0, N_SAMPLES)
    disk_x, disk_y = _sample_disk(rng, N_SAMPLES), _sample_disk(rng, N_SAMPLES)
    ring_x = _sample_disk(rng, N_SAMPLES, INNER_RADIUS)
    ring_y = _sample_disk(rng, N_SAMPLES, INNER_RADIUS)
    disk_mix = convex_combination(disk_x, disk_y, theta)
    ring_mix = convex_combination(ring_x, ring_y, theta)
    disk_norm, ring_norm = np.linalg.norm(disk_mix, axis=1), np.linalg.norm(ring_mix, axis=1)
    # Jensen 残差为左边减右边；凸函数应当满足 residual <= 0。
    convex_gap = jensen_residual(_quadratic, disk_x, disk_y, theta)
    saddle_gap = jensen_residual(_saddle, disk_x, disk_y, theta)
    eigenvalues = np.linalg.eigvalsh(Q)
    disk_violations = int(np.count_nonzero(disk_norm > 1.0 + TOL))
    ring_violations = int(np.count_nonzero((ring_norm < INNER_RADIUS - TOL) | (ring_norm > 1.0 + TOL)))
    convex_violations = int(np.count_nonzero(convex_gap > TOL))
    saddle_violations = int(np.count_nonzero(saddle_gap > TOL))
    # 构造反例：圆环两端点的中点是原点；鞍函数两端点的 Jensen 残差为 1。
    p, q = np.array([0.0, 1.0]), np.array([0.0, -1.0])
    explicit_gap = float(jensen_residual(_saddle, p[None, :], q[None, :], np.array([0.5]))[0])
    if disk_violations != 0 or convex_violations != 0 or eigenvalues.min() <= 0:
        raise RuntimeError("凸集或凸二次函数的数值检查失败，请检查公式和容差。")
    if (INNER_RADIUS > 0 and ring_violations == 0) or saddle_violations == 0 or explicit_gap <= 0:
        raise RuntimeError("未识别预设非凸反例，请检查抽样和残差方向。")
    if INNER_RADIUS == 0 and ring_violations != 0:
        raise RuntimeError("内半径为0时集合退化为圆盘，混合点应当仍在圆盘内。")
    with (out / "data.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample", "theta", "disk_x1", "disk_x2", "disk_y1", "disk_y2", "ring_x1", "ring_x2", "ring_y1", "ring_y2", "disk_mix_norm", "ring_mix_norm", "convex_jensen_residual", "saddle_jensen_residual"])
        for i in range(N_SAMPLES):
            writer.writerow([i, theta[i], *disk_x[i], *disk_y[i], *ring_x[i], *ring_y[i], disk_norm[i], ring_norm[i], convex_gap[i], saddle_gap[i]])
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    angles = np.linspace(0, 2 * np.pi, 300)
    for ax in axes[0]:
        ax.plot(np.cos(angles), np.sin(angles), color="black", linewidth=1)
        ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), xlabel="x1", ylabel="x2", aspect="equal")
        ax.grid(alpha=0.2)
    axes[0, 0].scatter(*disk_mix[:500].T, s=7, alpha=0.35, label="sampled mixtures")
    axes[0, 0].set_title("Unit disk: convex set")
    axes[0, 1].plot(INNER_RADIUS * np.cos(angles), INNER_RADIUS * np.sin(angles), color="black", linewidth=1)
    bad = ring_norm[:500] < INNER_RADIUS
    axes[0, 1].scatter(*ring_mix[:500].T, c=np.where(bad, "tab:red", "tab:blue"), s=7, alpha=0.5)
    axes[0, 1].plot([0, 0], [-1, 1], "k--", linewidth=1)
    axes[0, 1].scatter([0], [0], marker="x", color="red", s=70)
    axes[0, 1].set_title("Annulus: red mixtures violate membership")
    axes[1, 0].scatter(theta[:800], convex_gap[:800], s=7, alpha=0.35, label="convex quadratic")
    axes[1, 0].scatter(theta[:800], saddle_gap[:800], s=7, alpha=0.35, label="saddle function")
    axes[1, 0].axhline(0, color="black", linewidth=1)
    axes[1, 0].set(xlabel="theta", ylabel="Jensen residual", title="Positive residual is a counterexample")
    axes[1, 0].legend(fontsize=8)
    xx = np.linspace(-1, 1, 200)
    axes[1, 1].plot(xx, _quadratic(np.column_stack((xx, np.zeros_like(xx)))), label="convex slice: 1.5*x1^2")
    axes[1, 1].plot(xx, -xx**2, label="saddle slice: -x2^2")
    axes[1, 1].set(xlabel="coordinate", ylabel="function value", title="Curvature along coordinate directions")
    axes[1, 1].legend(fontsize=8)
    fig.savefig(out / "figure.png", dpi=170)
    plt.close(fig)
    metrics = {
        "experiment": "lab01", "seed": SEED, "samples": N_SAMPLES,
        "inner_radius": INNER_RADIUS,
        "disk_membership_violations": disk_violations,
        "annulus_membership_violations": ring_violations,
        "annulus_violation_rate": ring_violations / N_SAMPLES,
        "convex_jensen_violations": convex_violations,
        "max_convex_jensen_residual": float(convex_gap.max()),
        "saddle_jensen_violations": saddle_violations,
        "max_saddle_jensen_residual": float(saddle_gap.max()),
        "quadratic_hessian_eigenvalues": eigenvalues.tolist(),
        "saddle_hessian_eigenvalues": [-2.0, 2.0],
        "explicit_saddle_counterexample_residual": explicit_gap,
        "explicit_ring_midpoint_norm": 0.0,
        "explicit_ring_counterexample_valid": bool(INNER_RADIUS > 0),
        "tolerance": TOL,
        "interpretation": "Sampling finds counterexamples; convexity is established analytically by the PSD Hessian and the norm inequality.",
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        result = run(args.output_dir)
    except NotImplementedError as exc:
        parser.exit(2, f"实验尚未完成：{exc}\n请补全对应函数后重新运行。\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
