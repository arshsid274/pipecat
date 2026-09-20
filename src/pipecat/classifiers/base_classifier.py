#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Classifiers answer a typed question about some state.

A classifier is a plain object. Whoever needs answers creates one, keeps it,
and calls it. It does not sit in a pipeline and no frames flow into it. It
answers three kinds of question: whether the state satisfies a criteria
(:meth:`BaseClassifier.yes_no`), which of several options fits it
(:meth:`BaseClassifier.choice`), and where it falls on an ordered rubric
(:meth:`BaseClassifier.score`).
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeAlias

from pydantic import BaseModel

if TYPE_CHECKING:
    from pipecat.workers.base_worker import BaseWorker

#: What a question is about: plain text, or structured data such as a
#: transcript with speaker labels or a trimmed screen snapshot.
ClassifierState: TypeAlias = str | dict[str, Any] | list[Any]


class ClassifierError(Exception):
    """A classifier could not answer a question."""

    pass


class YesNoResult(BaseModel):
    """Answer to :meth:`BaseClassifier.yes_no`.

    Parameters:
        probability: How likely the answer is yes, from 0 to 1.
    """

    probability: float


class ChoiceResult(BaseModel):
    """Answer to :meth:`BaseClassifier.choice`.

    Parameters:
        label: The option that fits best.
        probabilities: How likely each option is, keyed by option label.
        confidence: How sure the classifier is of ``label``, from 0 to 1.
    """

    label: str
    probabilities: dict[str, float]
    confidence: float


class ScoreResult(BaseModel):
    """Answer to :meth:`BaseClassifier.score`.

    Parameters:
        score: Where the state falls on the rubric, as a position from 0 (the
            first level) to one less than the number of levels. It may fall
            between two levels.
        probabilities: How likely each level is, keyed by level description.
        confidence: How sure the classifier is of ``score``, from 0 to 1.
    """

    score: float
    probabilities: dict[str, float]
    confidence: float


class BaseClassifier(ABC):
    """Answers typed questions about a state.

    Subclasses implement the three questions. They say through
    :attr:`calibrated` whether their probabilities can be read as rates: a
    calibrated classifier that answers 0.8 is right about 80% of the time,
    so a threshold tuned on one does not carry to an uncalibrated one.

    The owner calls :meth:`setup` once before the first question and
    :meth:`cleanup` once when it is done. Both do nothing by default.
    """

    @property
    @abstractmethod
    def calibrated(self) -> bool:
        """Whether the probabilities this classifier returns are calibrated."""
        pass

    async def setup(self, worker: "BaseWorker"):
        """Prepare the classifier to answer questions.

        Args:
            worker: The worker the owner runs in, for implementations that
                need one.
        """
        pass

    async def cleanup(self):
        """Release whatever the classifier holds."""
        pass

    @abstractmethod
    async def yes_no(self, state: ClassifierState, criteria: str) -> YesNoResult:
        """Ask whether the state satisfies a criteria.

        Args:
            state: What the question is about.
            criteria: What is being checked for, as a yes or no question.

        Returns:
            How likely the answer is yes.

        Raises:
            ClassifierError: If no answer could be produced.
        """
        pass

    @abstractmethod
    async def choice(
        self, state: ClassifierState, options: dict[str, str], criteria: str
    ) -> ChoiceResult:
        """Ask which option fits the state.

        Args:
            state: What the question is about.
            options: The options to choose from, each label mapped to a
                description of when it applies.
            criteria: What is being decided.

        Returns:
            The option that fits and how likely each one is.

        Raises:
            ClassifierError: If no answer could be produced.
        """
        pass

    @abstractmethod
    async def score(self, state: ClassifierState, rubric: list[str], criteria: str) -> ScoreResult:
        """Ask where the state falls on an ordered rubric.

        Args:
            state: What the question is about.
            rubric: The levels in order, lowest first, each described in a
                few words. At least two.
            criteria: What is being rated.

        Returns:
            The position on the rubric and how likely each level is.

        Raises:
            ClassifierError: If no answer could be produced.
        """
        pass
