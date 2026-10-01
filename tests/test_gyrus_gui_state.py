import pytest

from gyrus.gui.state import AppState, InvalidStateTransition, ShellStateMachine


def test_all_fourteen_states_are_registered_and_transitions_are_guarded():
    assert len(AppState) == 14
    machine = ShellStateMachine()
    assert machine.state == AppState.BOOT_VERIFYING
    with pytest.raises(InvalidStateTransition):
        machine.enter(AppState.RESULT_STANDARD, reason="no final response")
    machine.enter(AppState.READY_EMPTY, reason="verified")
    machine.enter(AppState.INPUT_VALIDATING, reason="file selected")
    machine.enter(AppState.READY_TO_ANALYZE, reason="input accepted")
    machine.enter(AppState.ANALYZING, reason="complete job submitted")
    machine.enter(AppState.OOD_REVIEW_GATE, reason="valid OOD review")
    machine.enter(AppState.RESULT_REVIEW_REQUIRED, reason="user opened reviewed output")
    assert len(machine.history) == 6


def test_integrity_lock_requires_explicit_reverification():
    machine = ShellStateMachine()
    machine.enter(AppState.INTEGRITY_ERROR_LOCKED, reason="manifest mismatch")
    with pytest.raises(InvalidStateTransition):
        machine.enter(AppState.READY_EMPTY, reason="skip checks")
    machine.enter(AppState.BOOT_VERIFYING, reason="explicit retry")
    machine.enter(AppState.READY_EMPTY, reason="all hashes verified")


def test_every_transition_requires_a_nonempty_reason():
    machine = ShellStateMachine()
    with pytest.raises(InvalidStateTransition, match="needs a reason"):
        machine.enter(AppState.READY_EMPTY, reason="  ")
