from __future__ import annotations

import csv
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import lab01_convexity as lab


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"


def save_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(samples, radius, q=None):
    rng = np.random.default_rng(lab.SEED)
    theta = rng.uniform(0.0, 1.0, samples)
    x = lab._sample_disk(rng, samples)
    y = lab._sample_disk(rng, samples)
    ring_x = lab._sample_disk(rng, samples, radius)
    ring_y = lab._sample_disk(rng, samples, radius)
    norm = np.linalg.norm(lab.convex_combination(ring_x, ring_y, theta), axis=1)
    matrix = lab.Q if q is None else q
    quadratic = lambda points: 0.5 * np.sum((points @ matrix) * points, axis=1)
    gap = lab.jensen_residual(quadratic, x, y, theta)
    saddle = lab.jensen_residual(lab._saddle, x, y, theta)
    count = int(np.count_nonzero((norm < radius - lab.TOL) | (norm > 1 + lab.TOL)))
    return {
        "samples": samples,
        "inner_radius": radius,
        "seed": lab.SEED,
        "tolerance": lab.TOL,
        "annulus_violations": count,
        "annulus_violation_rate": count / samples,
        "quadratic_violations": int(np.count_nonzero(gap > lab.TOL)),
        "max_quadratic_residual": float(gap.max()),
        "saddle_violations": int(np.count_nonzero(saddle > lab.TOL)),
        "saddle_violation_rate": float(np.mean(saddle > lab.TOL)),
    }


def audit_saved_data():
    with (OUT / "data.csv").open(encoding="utf-8", newline="") as stream:
        rows = [next(stream) for _ in range(4)]
    records = list(csv.DictReader(rows))
    checks = []
    for row in records:
        t = float(row["theta"])
        x1, x2 = float(row["disk_x1"]), float(row["disk_x2"])
        y1, y2 = float(row["disk_y1"]), float(row["disk_y2"])
        z1, z2 = t * x1 + (1 - t) * y1, t * x2 + (1 - t) * y2
        f = lambda a, b: 1.5 * a * a + a * b + b * b
        g = lambda a, b: a * a - b * b
        expected = {
            "disk_mix_norm": (z1 * z1 + z2 * z2) ** 0.5,
            "convex_jensen_residual": f(z1, z2) - t * f(x1, x2) - (1 - t) * f(y1, y2),
            "saddle_jensen_residual": g(z1, z2) - t * g(x1, x2) - (1 - t) * g(y1, y2),
        }
        a1, a2 = float(row["ring_x1"]), float(row["ring_x2"])
        b1, b2 = float(row["ring_y1"]), float(row["ring_y2"])
        expected["ring_mix_norm"] = ((t * a1 + (1 - t) * b1) ** 2 + (t * a2 + (1 - t) * b2) ** 2) ** 0.5
        for name, value in expected.items():
            saved = float(row[name])
            error = abs(saved - value)
            if error > lab.TOL:
                raise AssertionError((row["sample"], name, error))
            checks.append({"sample": int(row["sample"]), "quantity": name, "saved": saved, "recomputed": value, "absolute_error": error, "passed": True})
    save_csv(OUT / "csv_audit.csv", checks)
    return max(row["absolute_error"] for row in checks)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    defaults = lab.run(OUT)
    sample_rows = [evaluate(n, 0.6) for n in (100, 1000, 10000)]
    radius_rows = [evaluate(4000, r) for r in (0.0, 0.3, 0.6, 0.8)]
    save_csv(OUT / "sample_comparison.csv", sample_rows)
    save_csv(OUT / "radius_comparison.csv", radius_rows)
    matrices = [("positive_definite", lab.Q), ("positive_semidefinite", np.diag([1.0, 0.0])), ("indefinite", np.diag([1.0, -0.2]))]
    matrix_rows = []
    for name, matrix in matrices:
        values = np.linalg.eigvalsh(matrix)
        row = evaluate(4000, 0.6, matrix)
        matrix_rows.append({"case": name, "eigenvalue_min": float(values[0]), "eigenvalue_max": float(values[1]), "violations": row["quadratic_violations"], "max_residual": row["max_quadratic_residual"], "direction_e2_curvature": float(matrix[1, 1])})
    save_csv(OUT / "hessian_comparison.csv", matrix_rows)
    assert all(row["quadratic_violations"] == 0 for row in sample_rows + radius_rows)
    assert radius_rows[0]["annulus_violations"] == 0
    assert all(row["annulus_violations"] > 0 for row in radius_rows[1:])
    assert matrix_rows[1]["violations"] == 0 and matrix_rows[2]["violations"] > 0
    error = audit_saved_data()

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    axes[0].semilogx([r["samples"] for r in sample_rows], [r["annulus_violation_rate"] for r in sample_rows], "o-", label="Annulus")
    axes[0].semilogx([r["samples"] for r in sample_rows], [r["saddle_violation_rate"] for r in sample_rows], "s-", label="Saddle")
    axes[0].set(xlabel="Sample count", ylabel="Violation fraction", title="Sample-size comparison", ylim=(0, 1))
    axes[0].legend()
    axes[1].plot([r["inner_radius"] for r in radius_rows], [r["annulus_violation_rate"] for r in radius_rows], "o-", color="tab:red")
    axes[1].set(xlabel="Inner radius", ylabel="Violation fraction", title="Changing the hole", ylim=(0, 1))
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.savefig(OUT / "parameter_comparison.png", dpi=180)
    plt.close(fig)

    coordinates = np.linspace(-1, 1, 90)
    xx, yy = np.meshgrid(coordinates, coordinates)
    fig = plt.figure(figsize=(12, 3.8), constrained_layout=True)
    for i, (name, matrix) in enumerate(matrices):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        zz = 0.5 * (matrix[0, 0] * xx ** 2 + 2 * matrix[0, 1] * xx * yy + matrix[1, 1] * yy ** 2)
        ax.plot_surface(xx, yy, zz, cmap="viridis", linewidth=0, alpha=0.9)
        ax.set(xlabel="x1", ylabel="x2", zlabel="f(x)", title=name.replace("_", " "))
        ax.view_init(elev=25, azim=-60)
    fig.savefig(OUT / "hessian_surfaces.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    t = np.linspace(-1, 1, 100)
    for name, matrix in matrices:
        axes[0].plot(t, 0.5 * matrix[1, 1] * t ** 2, label=name.replace("_", " "))
    axes[0].set(xlabel="x2", ylabel="f(0, x2)", title="Curvature in the e2 direction")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[1].plot([0, 0], [-1, 1], "k--", label="Segment between endpoints")
    angle = np.linspace(0, 2 * np.pi, 300)
    axes[1].plot(np.cos(angle), np.sin(angle), color="tab:blue")
    axes[1].plot(0.6 * np.cos(angle), 0.6 * np.sin(angle), color="tab:red")
    axes[1].scatter([0, 0], [-1, 1], s=55, label="Endpoints in annulus")
    axes[1].scatter([0], [0], marker="x", s=70, color="red", label="Midpoint outside annulus")
    axes[1].set(xlabel="x1", ylabel="x2", title="Explicit annulus counterexample", aspect="equal", xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))
    axes[1].annotate("p = (0, 1)", (0, 1), (0.12, 0.85), fontsize=8)
    axes[1].annotate("q = (0, -1)", (0, -1), (0.12, -1.05), fontsize=8)
    axes[1].annotate("midpoint (0, 0)", (0, 0), (0.12, -0.15), fontsize=8)
    fig.savefig(OUT / "analytic_counterexamples.png", dpi=180)
    plt.close(fig)

    summary = {"defaults": defaults, "sample_comparison": sample_rows, "radius_comparison": radius_rows, "hessian_comparison": matrix_rows, "csv_audit_max_error": error, "environment": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__}}
    (OUT / "study_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
