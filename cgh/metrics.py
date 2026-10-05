"""Метрики качества фазовой голограммы."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class QualityMetrics:
    eta: float
    uniformity: float
    rmse: float
    peak_to_valley: float
    iterations: int

    def as_dict(self) -> dict[str, float | int]:
        data = asdict(self)
        data["eta_pct"] = 100.0 * self.eta
        data["uniformity_pct"] = 100.0 * self.uniformity
        data["peak_to_valley_pct"] = 100.0 * self.peak_to_valley
        return data


def _masked(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected = np.asarray(values[mask], dtype=np.float64)
    return selected if selected.size else np.array([0.0])


def diffraction_efficiency(intensity: np.ndarray, signal_mask: np.ndarray) -> float:
    total = float(np.sum(intensity))
    if total <= 0.0:
        return 0.0
    return float(np.sum(intensity[signal_mask]) / total)


def uniformity(intensity: np.ndarray, core_mask: np.ndarray) -> float:
    vals = _masked(intensity, core_mask)
    mean = float(np.mean(vals))
    if mean <= 0.0:
        return 0.0
    return float(np.clip(1.0 - np.std(vals) / mean, 0.0, 1.0))


def spot_uniformity(intensity: np.ndarray, spots: tuple[tuple[float, float], ...], n: int) -> float:
    if not spots:
        return 0.0
    axis = np.linspace(-1.0, 1.0, n, endpoint=False)
    means = []
    radius = 4
    for xc, yc in spots:
        ix = int(np.argmin(np.abs(axis - xc)))
        iy = int(np.argmin(np.abs(axis - yc)))
        patch = intensity[max(iy - radius, 0) : iy + radius + 1, max(ix - radius, 0) : ix + radius + 1]
        means.append(float(np.mean(patch)))
    vals = np.array(means)
    mean = float(np.mean(vals))
    if mean <= 0.0:
        return 0.0
    return float(np.clip(1.0 - np.std(vals) / mean, 0.0, 1.0))


def peak_to_valley(intensity: np.ndarray, core_mask: np.ndarray) -> float:
    vals = _masked(intensity, core_mask)
    vmax = float(np.max(vals))
    vmin = float(np.min(vals))
    denom = vmax + vmin
    if denom <= 0.0:
        return 0.0
    return float(np.clip(1.0 - (vmax - vmin) / denom, 0.0, 1.0))


def rmse(intensity: np.ndarray, target_intensity: np.ndarray, signal_mask: np.ndarray) -> float:
    rec = _masked(intensity, signal_mask)
    tgt = _masked(target_intensity, signal_mask)
    rec = rec / (np.mean(rec) + 1e-15)
    tgt = tgt / (np.mean(tgt) + 1e-15)
    return float(np.sqrt(np.mean((rec - tgt) ** 2)))


def evaluate(
    intensity: np.ndarray,
    target_amplitude: np.ndarray,
    signal_mask: np.ndarray,
    core_mask: np.ndarray,
    iterations: int,
    spots: tuple[tuple[float, float], ...] = (),
) -> QualityMetrics:
    target_intensity = target_amplitude**2
    uni = (
        spot_uniformity(intensity, spots, intensity.shape[0])
        if spots
        else uniformity(intensity, core_mask)
    )
    return QualityMetrics(
        eta=diffraction_efficiency(intensity, signal_mask),
        uniformity=uni,
        rmse=rmse(intensity, target_intensity, signal_mask),
        peak_to_valley=peak_to_valley(intensity, core_mask),
        iterations=iterations,
    )


def iterations_to_converge(history: list[float], tol: float = 0.05) -> int:
    if not history:
        return 0
    best = min(history[len(history) // 2 :])
    threshold = best * (1.0 + tol) if best > 0 else best + 1e-6
    for index, value in enumerate(history, start=1):
        if value <= threshold:
            return index
    return len(history)
