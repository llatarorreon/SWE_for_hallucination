"""
Verifier package for hallucination detection in LLM agents.
Contains base verifier class and specific implementation for different verification scenarios.
"""

from .base_verifier import BaseVerifier
from .verify_repetitive_swebench import VerifyRepetitiveSWEbench
from .verify_erroneous_swebench import VerifyErroneousSWEbench
from .verify_misleading_swebench import VerifyMisleadingSWEbench

__all__ = [
    "BaseVerifier",
    "VerifyUnexpectedTransitionTAC",
    "VerifyUnexpectedTransitionWebarena",
    "VerifyRepetitive",
    "VerifyUsersQuestionsTAC",
    "VerifyUsersQuestionsTaubench",
    "VerifyPopupWebarena",
    "VerifyUnderspecifiedWebarena",
    "VerifyMisleadingWebarena",
    "VerifyUnachievableWebarena",
    "VerifyErroneous",
    "VerifyRepetitiveSWEbench",
    "VerifyErroneousSWEbench",
    "VerifyMisleadingSWEbench",
]
