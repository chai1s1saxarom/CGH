"""Синтез фазовых компьютерно-генерируемых голограмм (CGH / ДОЭ)."""

from .algorithms import aw_ifta, gerchberg_saxton, mraf, reconstruct, weighted_gs
from .fields import TargetField, make_initial_phase, make_input_beam, make_targets
from .metrics import evaluate

__all__ = [
    "TargetField",
    "aw_ifta",
    "evaluate",
    "gerchberg_saxton",
    "make_initial_phase",
    "make_input_beam",
    "make_targets",
    "mraf",
    "reconstruct",
    "weighted_gs",
]
