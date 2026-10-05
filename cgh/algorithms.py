"""Итерационные алгоритмы синтеза фазовых CGH в фурье-плоскости."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .fields import TargetField, coordinates
from .metrics import QualityMetrics, evaluate


@dataclass
class SynthesisResult:
    name: str
    phase: np.ndarray
    intensity: np.ndarray
    metrics: QualityMetrics
    rmse_history: list[float] = field(default_factory=list)
    eta_history: list[float] = field(default_factory=list)


def fft(field: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))


def ifft(field: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(field)))


def reconstruct(phase: np.ndarray, input_amp: np.ndarray) -> np.ndarray:
    return np.abs(fft(input_amp * np.exp(1j * phase))) ** 2


def _record(intensity: np.ndarray, target: TargetField, rmse_history: list[float], eta_history: list[float]) -> None:
    metrics = evaluate(
        intensity,
        target.amplitude,
        target.signal_mask,
        target.core_mask,
        iterations=len(rmse_history) + 1,
        spots=target.spots,
    )
    rmse_history.append(metrics.rmse)
    eta_history.append(metrics.eta)


def _match_signal_energy(amp: np.ndarray, shape: np.ndarray, mask: np.ndarray) -> np.ndarray:
    current = np.sum(amp[mask] ** 2)
    desired = np.sum(shape[mask] ** 2)
    scale = np.sqrt(current / (desired + 1e-15))
    return shape * scale


def _apply_eta_floor(amp: np.ndarray, mask: np.ndarray, eta_floor: float) -> np.ndarray:
    signal = np.where(mask, amp, 0.0)
    noise = np.where(~mask, amp, 0.0)
    energy_s = np.sum(signal**2)
    energy_n = np.sum(noise**2)
    total = energy_s + energy_n
    if total <= 0.0 or energy_s <= 0.0:
        return amp
    if energy_s / total >= eta_floor or energy_n <= 0.0:
        return amp
    target_noise = energy_s * (1.0 / eta_floor - 1.0)
    noise *= np.sqrt(target_noise / energy_n)
    return signal + noise


def _final(name: str, phase: np.ndarray, input_amp: np.ndarray, target: TargetField, n_iter: int, rmse_h: list[float], eta_h: list[float]) -> SynthesisResult:
    intensity = reconstruct(phase, input_amp)
    metrics = evaluate(
        intensity,
        target.amplitude,
        target.signal_mask,
        target.core_mask,
        n_iter,
        spots=target.spots,
    )
    return SynthesisResult(name, phase, intensity, metrics, rmse_h, eta_h)


def gerchberg_saxton(
    target: TargetField,
    input_amp: np.ndarray,
    n_iter: int,
    phase0: np.ndarray,
) -> SynthesisResult:
    """Классический алгоритм Герчберга–Секстона (базовый метод)."""
    phase = phase0.copy()
    rmse_h: list[float] = []
    eta_h: list[float] = []
    target_amp = target.amplitude

    for _ in range(n_iter):
        far = fft(input_amp * np.exp(1j * phase))
        _record(np.abs(far) ** 2, target, rmse_h, eta_h)
        far = _match_signal_energy(np.abs(far), target_amp, np.ones(target_amp.shape, dtype=bool)) * np.exp(1j * np.angle(far))
        phase = np.angle(ifft(far))

    return _final("GS", phase, input_amp, target, n_iter, rmse_h, eta_h)


def weighted_gs(
    target: TargetField,
    input_amp: np.ndarray,
    n_iter: int,
    phase0: np.ndarray,
    gamma: float = 0.45,
) -> SynthesisResult:
    """Взвешенный GS: веса компенсируют провалы интенсивности."""
    phase = phase0.copy()
    weights = np.ones_like(target.amplitude)
    rmse_h: list[float] = []
    eta_h: list[float] = []
    mask = target.signal_mask
    target_amp = target.amplitude
    eps = 1e-8

    for _ in range(n_iter):
        far = fft(input_amp * np.exp(1j * phase))
        amp = np.abs(far)
        _record(amp**2, target, rmse_h, eta_h)
        if target.spots:
            weights = _update_spot_weights(weights, amp**2, target, gamma)
        else:
            weights[mask] *= (target_amp[mask] / (amp[mask] + eps)) ** gamma
            weights = np.clip(weights, 0.25, 6.0)
        desired = _match_signal_energy(amp, weights * target_amp, np.ones_like(mask))
        phase = np.angle(ifft(desired * np.exp(1j * np.angle(far))))

    return _final("WGS", phase, input_amp, target, n_iter, rmse_h, eta_h)


def mraf(
    target: TargetField,
    input_amp: np.ndarray,
    n_iter: int,
    phase0: np.ndarray,
    mix: float = 0.55,
    damp: float = 0.82,
    eta_floor: float = 0.80,
) -> SynthesisResult:
    """IFTA со свободой амплитуды в шумовом окне (MRAF) и порогом эффективности."""
    phase = phase0.copy()
    rmse_h: list[float] = []
    eta_h: list[float] = []
    mask = target.signal_mask
    target_amp = target.amplitude

    for _ in range(n_iter):
        far = fft(input_amp * np.exp(1j * phase))
        amp = np.abs(far)
        _record(amp**2, target, rmse_h, eta_h)
        desired = _match_signal_energy(amp, target_amp, mask)
        new_amp = np.where(mask, mix * desired + (1.0 - mix) * amp, damp * amp)
        new_amp = _apply_eta_floor(new_amp, mask, eta_floor)
        phase = np.angle(ifft(new_amp * np.exp(1j * np.angle(far))))

    return _final("MRAF", phase, input_amp, target, n_iter, rmse_h, eta_h)


def aw_ifta(
    target: TargetField,
    input_amp: np.ndarray,
    n_iter: int,
    phase0: np.ndarray,
    mix_start: float = 0.78,
    mix_end: float = 0.42,
    damp_start: float = 0.70,
    damp_end: float = 0.90,
    eta_start: float = 0.88,
    eta_end: float = 0.80,
    gamma: float = 0.28,
) -> SynthesisResult:
    """Адаптивный IFTA: MRAF + весовое выравнивание + отжиг mix/damp/eta.

    Спекл вытесняется в шумовое окно, а адаптивные веса выглаживают сигнальную
    область. Порог eta не даёт эффективности просесть ниже заданного уровня.
    """
    phase = phase0.copy()
    weights = np.ones_like(target.amplitude)
    rmse_h: list[float] = []
    eta_h: list[float] = []
    mask = target.signal_mask
    target_amp = target.amplitude
    eps = 1e-8

    for step in range(n_iter):
        t = step / max(n_iter - 1, 1)
        mix = mix_start + (mix_end - mix_start) * t
        damp = damp_start + (damp_end - damp_start) * t
        eta_floor = eta_start + (eta_end - eta_start) * t
        far = fft(input_amp * np.exp(1j * phase))
        amp = np.abs(far)
        _record(amp**2, target, rmse_h, eta_h)
        if target.spots:
            weights = _update_spot_weights(weights, amp**2, target, gamma)
        else:
            weights[mask] *= (target_amp[mask] / (amp[mask] + eps)) ** gamma
            weights = np.clip(weights, 0.3, 5.0)
        desired = _match_signal_energy(amp, weights * target_amp, mask)
        new_amp = np.where(mask, mix * desired + (1.0 - mix) * amp, damp * amp)
        new_amp = _apply_eta_floor(new_amp, mask, eta_floor)
        phase = np.angle(ifft(new_amp * np.exp(1j * np.angle(far))))

    return _final("AW-IFTA", phase, input_amp, target, n_iter, rmse_h, eta_h)


def _update_spot_weights(
    weights: np.ndarray,
    intensity: np.ndarray,
    target: TargetField,
    gamma: float,
) -> np.ndarray:
    x, y = coordinates(intensity.shape[0])
    updated = weights.copy()
    sigma = 7.5 / intensity.shape[0]
    means = []
    masks = []
    for xc, yc in target.spots:
        spot = ((x - xc) ** 2 + (y - yc) ** 2) <= (1.8 * sigma) ** 2
        masks.append(spot)
        means.append(float(np.mean(intensity[spot])) if np.any(spot) else 0.0)
    global_mean = float(np.mean([m for m in means if m > 0]) or 1.0)
    for spot, mean in zip(masks, means):
        factor = (global_mean / (mean + 1e-8)) ** gamma
        updated[spot] = np.clip(updated[spot] * factor, 0.25, 8.0)
    return updated


Algorithm = Callable[[TargetField, np.ndarray, int, np.ndarray], SynthesisResult]


def all_algorithms() -> list[tuple[str, Algorithm]]:
    return [
        ("GS", gerchberg_saxton),
        ("WGS", weighted_gs),
        ("MRAF", mraf),
        ("AW-IFTA", aw_ifta),
    ]
