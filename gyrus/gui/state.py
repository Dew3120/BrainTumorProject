"""Explicit application states and guarded transitions for the Gyrus shell."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AppState(StrEnum):
    BOOT_VERIFYING = "BOOT_VERIFYING"
    BOOT_BLOCKED = "BOOT_BLOCKED"
    READY_EMPTY = "READY_EMPTY"
    INPUT_VALIDATING = "INPUT_VALIDATING"
    READY_TO_ANALYZE = "READY_TO_ANALYZE"
    ANALYZING = "ANALYZING"
    OOD_REVIEW_GATE = "OOD_REVIEW_GATE"
    RESULT_STANDARD = "RESULT_STANDARD"
    RESULT_REVIEW_REQUIRED = "RESULT_REVIEW_REQUIRED"
    EDITING_MASK = "EDITING_MASK"
    EXPORT_PREVIEW = "EXPORT_PREVIEW"
    HISTORY = "HISTORY"
    RECOVERABLE_ERROR = "RECOVERABLE_ERROR"
    INTEGRITY_ERROR_LOCKED = "INTEGRITY_ERROR_LOCKED"


_TRANSITIONS: dict[AppState, frozenset[AppState]] = {
    AppState.BOOT_VERIFYING: frozenset({
        AppState.BOOT_BLOCKED,
        AppState.READY_EMPTY,
        AppState.INTEGRITY_ERROR_LOCKED,
    }),
    AppState.BOOT_BLOCKED: frozenset({AppState.BOOT_VERIFYING}),
    AppState.READY_EMPTY: frozenset({
        AppState.INPUT_VALIDATING,
        AppState.HISTORY,
        AppState.INTEGRITY_ERROR_LOCKED,
    }),
    AppState.INPUT_VALIDATING: frozenset({
        AppState.READY_EMPTY,
        AppState.READY_TO_ANALYZE,
        AppState.RECOVERABLE_ERROR,
        AppState.INTEGRITY_ERROR_LOCKED,
    }),
    AppState.READY_TO_ANALYZE: frozenset({
        AppState.ANALYZING,
        AppState.READY_EMPTY,
        AppState.HISTORY,
        AppState.INTEGRITY_ERROR_LOCKED,
    }),
    AppState.ANALYZING: frozenset({
        AppState.OOD_REVIEW_GATE,
        AppState.RESULT_STANDARD,
        AppState.RESULT_REVIEW_REQUIRED,
        AppState.RECOVERABLE_ERROR,
        AppState.INTEGRITY_ERROR_LOCKED,
    }),
    AppState.OOD_REVIEW_GATE: frozenset({
        AppState.RESULT_REVIEW_REQUIRED,
        AppState.READY_EMPTY,
        AppState.INTEGRITY_ERROR_LOCKED,
    }),
    AppState.RESULT_STANDARD: frozenset({
        AppState.EDITING_MASK,
        AppState.EXPORT_PREVIEW,
        AppState.READY_EMPTY,
        AppState.HISTORY,
        AppState.INTEGRITY_ERROR_LOCKED,
    }),
    AppState.RESULT_REVIEW_REQUIRED: frozenset({
        AppState.EDITING_MASK,
        AppState.EXPORT_PREVIEW,
        AppState.READY_EMPTY,
        AppState.HISTORY,
        AppState.INTEGRITY_ERROR_LOCKED,
    }),
    AppState.EDITING_MASK: frozenset({
        AppState.RESULT_STANDARD,
        AppState.RESULT_REVIEW_REQUIRED,
        AppState.RECOVERABLE_ERROR,
        AppState.INTEGRITY_ERROR_LOCKED,
    }),
    AppState.EXPORT_PREVIEW: frozenset({
        AppState.RESULT_STANDARD,
        AppState.RESULT_REVIEW_REQUIRED,
        AppState.RECOVERABLE_ERROR,
        AppState.INTEGRITY_ERROR_LOCKED,
    }),
    AppState.HISTORY: frozenset({
        AppState.READY_EMPTY,
        AppState.RESULT_STANDARD,
        AppState.RESULT_REVIEW_REQUIRED,
        AppState.INTEGRITY_ERROR_LOCKED,
    }),
    AppState.RECOVERABLE_ERROR: frozenset({
        AppState.READY_EMPTY,
        AppState.READY_TO_ANALYZE,
        AppState.ANALYZING,
        AppState.INTEGRITY_ERROR_LOCKED,
    }),
    AppState.INTEGRITY_ERROR_LOCKED: frozenset({AppState.BOOT_VERIFYING}),
}


class InvalidStateTransition(ValueError):
    """The requested user interface transition violates the workflow."""


@dataclass(frozen=True, slots=True)
class StateTransition:
    source: AppState
    target: AppState
    reason: str


class ShellStateMachine:
    """Keep state history without importing Qt or model dependencies."""

    def __init__(self) -> None:
        self._state = AppState.BOOT_VERIFYING
        self._history: list[StateTransition] = []

    @property
    def state(self) -> AppState:
        return self._state

    @property
    def history(self) -> tuple[StateTransition, ...]:
        return tuple(self._history)

    def can_enter(self, target: AppState) -> bool:
        return target in _TRANSITIONS[self._state]

    def enter(self, target: AppState, *, reason: str) -> StateTransition:
        if not reason.strip():
            raise InvalidStateTransition("every state transition needs a reason")
        if not self.can_enter(target):
            raise InvalidStateTransition(f"{self._state.value} cannot enter {target.value}")
        event = StateTransition(self._state, target, reason)
        self._state = target
        self._history.append(event)
        return event


assert set(_TRANSITIONS) == set(AppState)
