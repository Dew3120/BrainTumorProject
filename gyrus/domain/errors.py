"""Fail-closed exceptions with stable machine-readable codes."""

from __future__ import annotations


class DomainError(RuntimeError):
    """Base error safe to transport from the domain layer to a presentation layer."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class ManifestVerificationError(DomainError):
    """The frozen stack cannot be trusted or used."""


class InputValidationError(DomainError):
    """The selected local image violates the registered input contract."""


class PolicyError(DomainError):
    """A policy input or state transition is inconsistent."""


class ResultValidationError(DomainError):
    """A worker result violates the immutable result schema."""


class ProvenanceError(DomainError):
    """A mask-edit chain is invalid or attempts to overwrite model evidence."""


class HistoryRepositoryError(DomainError):
    """A local opt-in history operation failed safely."""
