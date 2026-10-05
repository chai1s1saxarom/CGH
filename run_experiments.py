"""Сравнительный синтез CGH и экспорт рисунков / таблиц для презентации."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cgh.algorithms import SynthesisResult, all_algorithms
from cgh.fields import TargetField, make_initial_phase, make_input_beam, make_targets
from cgh.metrics import iterations_to_converge


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
N_PIXELS = 512
N_ITER = 120
SEED = 42

ALG_COLORS = {
    "GS": "#6B7280",
    "WGS": "#2563EB",
    "MRAF": "#D97706",
    "AW-IFTA": "#059669",
}


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    input_amp = make_input_beam(N_PIXELS)
    phase0 = make_initial_phase(N_PIXELS, rng)
    targets = make_targets(N_PIXELS)

    catalog: dict[str, dict[str, SynthesisResult]] = {}
    rows: list[dict[str, object]] = []

    for target in targets:
        catalog[target.name] = {}
        print(f"\n=== {target.title} ===")
        for alg_name, solver in all_algorithms():
            result = solver(target, input_amp, N_ITER, phase0)
            catalog[target.name][alg_name] = result
            row = {
                "target": target.name,
                "title": target.title,
                "algorithm": alg_name,
                "eta_pct": round(100.0 * result.metrics.eta, 2),
                "uniformity_pct": round(100.0 * result.metrics.uniformity, 2),
                "rmse": round(result.metrics.rmse, 6),
                "ptv_pct": round(100.0 * result.metrics.peak_to_valley, 2),
                "iterations": result.metrics.iterations,
                "iters_to_converge": iterations_to_converge(result.rmse_history),
            }
            rows.append(row)
            print(
                f"  {alg_name:8s}  eta={row['eta_pct']:6.2f}%  "
                f"U={row['uniformity_pct']:6.2f}%  "
                f"RMSE={row['rmse']:.4f}  "
                f"PV={row['ptv_pct']:6.2f}%  "
                f"N*={row['iters_to_converge']}"
            )

    _save_table(rows)
    _save_json(rows)
    _plot_all(targets, catalog, input_amp)
    print(f"\nГотово: таблицы в {RESULTS}, рисунки в {FIGURES}")


def _save_table(rows: list[dict[str, object]]) -> None:
    path = RESULTS / "metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_json(rows: list[dict[str, object]]) -> None:
    path = RESULTS / "metrics.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Segoe UI",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": 200,
        }
    )


def _plot_all(
    targets: list[TargetField],
    catalog: dict[str, dict[str, SynthesisResult]],
    input_amp: np.ndarray,
) -> None:
    _setup_style()
    _plot_targets(targets)
    _plot_input(input_amp)
    _plot_compare_tophat(targets[0], catalog[targets[0].name])
    _plot_hologram(catalog[targets[0].name]["AW-IFTA"])
    _plot_profiles(targets[0], catalog[targets[0].name])
    _plot_convergence(targets[0], catalog[targets[0].name])
    _plot_metrics_bars(catalog)
    _plot_gallery(targets, catalog)
    _plot_multispot(targets[3], catalog[targets[3].name])
    _plot_metrics_table(catalog)


def _norm(image: np.ndarray) -> np.ndarray:
    peak = float(np.max(image))
    return image / peak if peak > 0 else image


def _imshow(ax, image: np.ndarray, cmap: str, title: str, vmax: float | None = 1.0) -> None:
    ax.imshow(_norm(image), cmap=cmap, origin="lower", vmin=0.0, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def _plot_targets(targets: list[TargetField]) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.3))
    for ax, target in zip(axes, targets):
        _imshow(ax, target.amplitude**2, "inferno", target.title)
    fig.suptitle("Целевые распределения интенсивности", y=1.03)
    fig.savefig(FIGURES / "01_targets.png")
    plt.close(fig)


def _plot_input(input_amp: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    _imshow(ax, input_amp, "viridis", "Входной гауссов пучок (плоскость CGH)")
    fig.savefig(FIGURES / "02_input_beam.png")
    plt.close(fig)


def _center_crop(image: np.ndarray, fraction: float = 0.42) -> np.ndarray:
    n = image.shape[0]
    half = max(int(n * fraction / 2), 1)
    c = n // 2
    return image[c - half : c + half, c - half : c + half]


def _plot_compare_tophat(target: TargetField, results: dict[str, SynthesisResult]) -> None:
    images = [
        (_center_crop(target.amplitude**2), "Целевое поле"),
        (_center_crop(results["GS"].intensity), "GS, базовый"),
        (_center_crop(results["WGS"].intensity), "WGS"),
        (_center_crop(results["AW-IFTA"].intensity), "AW-IFTA, предложенный"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.6))
    for ax, (image, title) in zip(axes, images):
        _imshow(ax, image, "inferno", title)
    fig.suptitle("Круглый top-hat: цель и восстановленные поля", y=1.04)
    fig.savefig(FIGURES / "03_tophat_compare.png")
    plt.close(fig)


def _plot_hologram(result: SynthesisResult) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    axes[0].imshow(result.phase, cmap="twilight", origin="lower", vmin=-np.pi, vmax=np.pi)
    axes[0].set_title("Фаза CGH, AW-IFTA")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    _imshow(axes[1], result.intensity, "inferno", "Восстановленная интенсивность")
    fig.savefig(FIGURES / "04_hologram_phase.png")
    plt.close(fig)


def _plot_profiles(target: TargetField, results: dict[str, SynthesisResult]) -> None:
    mid = target.amplitude.shape[0] // 2
    core = target.core_mask[mid]
    x = np.linspace(-1.0, 1.0, target.amplitude.shape[0], endpoint=False)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    tgt = target.amplitude[mid] ** 2
    ax.plot(x, tgt / (np.mean(tgt[core]) + 1e-15), color="#111827", lw=2.2, label="Цель")
    for name in ("GS", "WGS", "MRAF", "AW-IFTA"):
        profile = results[name].intensity[mid]
        ax.plot(x, profile / (np.mean(profile[core]) + 1e-15), color=ALG_COLORS[name], lw=1.6, label=name)
    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(0.0, 2.4)
    ax.set_xlabel("Поперечная координата, отн. ед.")
    ax.set_ylabel("Интенсивность / среднее в ядре")
    ax.set_title("Поперечный профиль круглого top-hat")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.savefig(FIGURES / "05_tophat_profile.png")
    plt.close(fig)


def _plot_convergence(target: TargetField, results: dict[str, SynthesisResult]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    for name, result in results.items():
        it = np.arange(1, len(result.rmse_history) + 1)
        axes[0].plot(it[2:], result.rmse_history[2:], color=ALG_COLORS[name], lw=1.8, label=name)
        axes[1].plot(it, 100.0 * np.array(result.eta_history), color=ALG_COLORS[name], lw=1.8, label=name)
    axes[0].set_title("Сходимость RMSE")
    axes[0].set_xlabel("Итерация")
    axes[0].set_ylabel("RMSE")
    axes[0].set_ylim(0.15, 0.55)
    axes[1].set_title("Дифракционная эффективность")
    axes[1].set_xlabel("Итерация")
    axes[1].set_ylabel("η, %")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle(f"Сходимость: {target.title}", y=1.02)
    fig.savefig(FIGURES / "06_convergence.png")
    plt.close(fig)


def _plot_metrics_bars(catalog: dict[str, dict[str, SynthesisResult]]) -> None:
    target_order = list(catalog.keys())
    titles = {
        "tophat_circle": "Круг",
        "tophat_square": "Квадрат",
        "ring": "Кольцо",
        "spots_4x4": "4×4",
    }
    algorithms = ["GS", "WGS", "MRAF", "AW-IFTA"]
    x = np.arange(len(target_order))
    width = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    metrics = [
        (axes[0], "eta", "Дифракционная эффективность η, %", True),
        (axes[1], "uniformity", "Однородность U = 1 − σ/μ, %", True),
    ]
    for ax, key, title, percent in metrics:
        for i, alg in enumerate(algorithms):
            values = []
            for name in target_order:
                value = getattr(catalog[name][alg].metrics, key)
                values.append(100.0 * value if percent else value)
            ax.bar(x + (i - 1.5) * width, values, width, label=alg, color=ALG_COLORS[alg])
        ax.set_xticks(x, [titles[name] for name in target_order])
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False, ncol=2)
    fig.savefig(FIGURES / "07_metrics_bars.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for i, alg in enumerate(algorithms):
        values = [catalog[name][alg].metrics.rmse for name in target_order]
        ax.bar(x + (i - 1.5) * width, values, width, label=alg, color=ALG_COLORS[alg])
    ax.set_xticks(x, [titles[name] for name in target_order])
    ax.set_title("Среднеквадратичная ошибка RMSE")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.savefig(FIGURES / "08_rmse_bars.png")
    plt.close(fig)


def _plot_gallery(targets: list[TargetField], catalog: dict[str, dict[str, SynthesisResult]]) -> None:
    fig, axes = plt.subplots(len(targets), 3, figsize=(9.6, 11.0))
    for row, target in enumerate(targets):
        _imshow(axes[row, 0], target.amplitude**2, "inferno", f"{target.title}: цель")
        _imshow(axes[row, 1], catalog[target.name]["GS"].intensity, "inferno", "GS")
        _imshow(axes[row, 2], catalog[target.name]["AW-IFTA"].intensity, "inferno", "AW-IFTA")
    fig.suptitle("Сравнение восстановленных полей: GS vs AW-IFTA", y=0.995)
    fig.tight_layout()
    fig.savefig(FIGURES / "09_gallery.png")
    plt.close(fig)


def _plot_multispot(target: TargetField, results: dict[str, SynthesisResult]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.5))
    _imshow(axes[0], target.amplitude**2, "inferno", "Цель: массив 4x4")
    _imshow(axes[1], results["GS"].intensity, "inferno", "GS")
    _imshow(axes[2], results["AW-IFTA"].intensity, "inferno", "AW-IFTA")
    fig.savefig(FIGURES / "10_multispot.png")
    plt.close(fig)


def _plot_metrics_table(catalog: dict[str, dict[str, SynthesisResult]]) -> None:
    titles = {
        "tophat_circle": "Круг",
        "tophat_square": "Квадрат",
        "ring": "Кольцо",
        "spots_4x4": "4x4",
    }
    algorithms = ["GS", "WGS", "MRAF", "AW-IFTA"]
    cell = [["Поле / метод", "Метрика"] + algorithms]
    for name, title in titles.items():
        eta = [f"{100.0 * catalog[name][alg].metrics.eta:.1f}" for alg in algorithms]
        uni = [f"{100.0 * catalog[name][alg].metrics.uniformity:.1f}" for alg in algorithms]
        rmse = [f"{catalog[name][alg].metrics.rmse:.3f}" for alg in algorithms]
        cell.append([title, "η, %"] + eta)
        cell.append(["", "U, %"] + uni)
        cell.append(["", "RMSE"] + rmse)

    fig, ax = plt.subplots(figsize=(10.6, 5.6))
    ax.axis("off")
    table = ax.table(cellText=cell, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.15, 1.45)
    for (row, col), cell_obj in table.get_celld().items():
        if row == 0:
            cell_obj.set_facecolor("#111827")
            cell_obj.set_text_props(color="white", weight="bold")
        elif col <= 1:
            cell_obj.set_facecolor("#F3F4F6")
        elif col == 1 + algorithms.index("AW-IFTA"):
            cell_obj.set_facecolor("#D1FAE5")
    ax.set_title("Сводная таблица метрик: GS, WGS, MRAF и AW-IFTA")
    fig.savefig(FIGURES / "11_metrics_table.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
