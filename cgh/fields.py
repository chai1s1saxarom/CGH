"""Сетки, входной пучок и целевые поля в плоскости изображения."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class TargetField:
    name: str
    title: str
    amplitude: np.ndarray
    signal_mask: np.ndarray
    core_mask: np.ndarray
    spots: tuple[tuple[float, float], ...] = field(default_factory=tuple)


def coordinates(n: int) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(-1.0, 1.0, n, endpoint=False)
    return np.meshgrid(axis, axis)


def _smooth_disk(x: np.ndarray, y: np.ndarray, radius: float, edge: float) -> np.ndarray:
    rho = np.hypot(x, y)
    return 0.5 * (1.0 - np.tanh((rho - radius) / max(edge, 1e-6)))


def _smooth_rect(x: np.ndarray, y: np.ndarray, half: float, edge: float) -> np.ndarray:
    ax = 0.5 * (1.0 - np.tanh((np.abs(x) - half) / max(edge, 1e-6)))
    ay = 0.5 * (1.0 - np.tanh((np.abs(y) - half) / max(edge, 1e-6)))
    return ax * ay


def make_input_beam(n: int, waist: float = 0.42) -> np.ndarray:
    """Гауссова амплитуда на круглой апертуре фазового модулятора."""
    x, y = coordinates(n)
    rho = np.hypot(x, y)
    gauss = np.exp(-(rho / waist) ** 2)
    beam = gauss * (rho <= 0.98)
    return beam / (np.sqrt(np.sum(beam**2)) + 1e-15)


def make_initial_phase(n: int, rng: np.random.Generator) -> np.ndarray:
    """Слабая квадратичная фаза плюс небольшой шум — меньше оптических вихрей, чем у чисто случайной фазы."""
    x, y = coordinates(n)
    return 1.8 * (x**2 + y**2) + rng.uniform(-0.35, 0.35, size=(n, n))


def make_targets(n: int) -> list[TargetField]:
    x, y = coordinates(n)
    edge = 2.2 / n
    return [
        _circular_tophat(x, y, edge),
        _square_tophat(x, y, edge),
        _ring(x, y, edge),
        _spot_array(x, y, n),
    ]


def _normalize_amp(amp: np.ndarray) -> np.ndarray:
    return amp / (np.sqrt(np.sum(amp**2)) + 1e-15)


def _circular_tophat(x: np.ndarray, y: np.ndarray, edge: float) -> TargetField:
    amp = _normalize_amp(_smooth_disk(x, y, 0.22, edge))
    signal = _smooth_disk(x, y, 0.235, edge) > 0.5
    core = _smooth_disk(x, y, 0.185, edge) > 0.5
    return TargetField("tophat_circle", "Круглый top-hat", amp, signal, core)


def _square_tophat(x: np.ndarray, y: np.ndarray, edge: float) -> TargetField:
    amp = _normalize_amp(_smooth_rect(x, y, 0.20, edge))
    signal = _smooth_rect(x, y, 0.215, edge) > 0.5
    core = _smooth_rect(x, y, 0.165, edge) > 0.5
    return TargetField("tophat_square", "Квадратный top-hat", amp, signal, core)


def _ring(x: np.ndarray, y: np.ndarray, edge: float) -> TargetField:
    amp = _normalize_amp(np.clip(_smooth_disk(x, y, 0.28, edge) - _smooth_disk(x, y, 0.16, edge), 0.0, 1.0))
    signal = (_smooth_disk(x, y, 0.30, edge) - _smooth_disk(x, y, 0.14, edge)) > 0.5
    core = (_smooth_disk(x, y, 0.265, edge) - _smooth_disk(x, y, 0.175, edge)) > 0.5
    return TargetField("ring", "Кольцо", amp, signal, core)


def _spot_array(x: np.ndarray, y: np.ndarray, n: int) -> TargetField:
    sigma = 6.5 / n
    coords = (-0.30, -0.10, 0.10, 0.30)
    spots = tuple((xc, yc) for xc in coords for yc in coords)
    amp = np.zeros_like(x)
    core = np.zeros_like(x, dtype=bool)
    for xc, yc in spots:
        amp += np.exp(-((x - xc) ** 2 + (y - yc) ** 2) / (2.0 * sigma**2))
        core |= (x - xc) ** 2 + (y - yc) ** 2 <= (1.6 * sigma) ** 2
    amp = _normalize_amp(amp)
    signal = amp > 0.08 * amp.max()
    return TargetField("spots_4x4", "Массив 4x4", amp, signal, core, spots)
