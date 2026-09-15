"""Typed, fail-closed errors for the Phase 11 training pipeline."""

from __future__ import annotations


class Phase11Error(ValueError):
    """Base class for bounded Phase 11 contract and pipeline failures."""


class ContractError(Phase11Error):
    pass


class UnknownVersionError(ContractError):
    pass


class InvalidStatusError(ContractError):
    pass


class Phase10InvalidError(Phase11Error):
    pass


class EmptyEligibleSetError(Phase11Error):
    pass


class SequenceTooLongError(Phase11Error):
    pass
