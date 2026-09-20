#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Classifier backed by any Pipecat LLM service.

The LLM runs the normal Pipecat way, inside a pipeline, in a worker the
classifier creates for it: a
:class:`~pipecat.workers.llm.llm_classifier_worker.LLMClassifierWorker`.
Each question is a job to that worker, and the LLM answers by calling a
tool. The worker is attached to whatever worker the classifier is set up
with, starts and stops with it, and takes jobs from nothing else.
"""

from typing import TYPE_CHECKING, Any

from pipecat.classifiers.base_classifier import (
    BaseClassifier,
    ChoiceResult,
    ClassifierError,
    ClassifierState,
    ScoreResult,
    YesNoResult,
)
from pipecat.pipeline.job_context import JobError, JobParams
from pipecat.services.llm_service import LLMService
from pipecat.workers.llm.llm_classifier_worker import CLASSIFY_JOB_NAME, LLMClassifierWorker

if TYPE_CHECKING:
    from pipecat.workers.base_worker import BaseWorker


class LLMClassifier(BaseClassifier):
    """Answers questions by asking an LLM, kept in a pipeline of its own.

    The probabilities are whatever the LLM wrote into its tool call, so they
    are not calibrated.

    Example::

        classifier = LLMClassifier(llm=OpenAILLMService(model="gpt-4o-mini"))
        detector = VoicemailDetector(classifier=classifier)
    """

    def __init__(
        self,
        *,
        llm: LLMService[Any],
        instructions: str | None = None,
        name: str | None = None,
        keep_history: bool = False,
        max_history: int = 20,
        timeout: float = 10.0,
    ):
        """Initialize the classifier.

        Args:
            llm: The LLM that answers the questions.
            instructions: System instructions for the LLM. The default tells
                it to answer only by calling the tool each question names.
            name: Name for the worker the classifier creates; auto-generated
                when omitted.
            keep_history: Whether the LLM sees its earlier questions and
                answers.
            max_history: With ``keep_history``, how many messages to keep.
            timeout: Seconds to wait for an answer, including the wait for
                the worker to start.
        """
        self._worker = LLMClassifierWorker(
            llm=llm,
            instructions=instructions,
            name=name,
            keep_history=keep_history,
            max_history=max_history,
        )
        self._timeout = timeout
        self._owner: BaseWorker | None = None

    @property
    def calibrated(self) -> bool:
        """An LLM's probabilities are not calibrated."""
        return False

    @property
    def worker(self) -> LLMClassifierWorker:
        """The worker that runs the LLM."""
        return self._worker

    async def setup(self, worker: "BaseWorker"):
        """Attach the LLM worker as a child of the worker the owner runs in.

        Args:
            worker: The owner's worker. The LLM worker starts and stops with
                it, and questions are sent from it.
        """
        if worker is None:
            raise ValueError("LLMClassifier needs the worker it runs in")
        self._owner = worker
        await worker.add_workers(self._worker)

    async def yes_no(self, state: ClassifierState, criteria: str) -> YesNoResult:
        """Ask whether the state satisfies a criteria.

        Args:
            state: What the question is about.
            criteria: What is being checked for, as a yes or no question.

        Returns:
            How likely the answer is yes.
        """
        answer = await self._ask({"kind": "yes_no", "state": state, "criteria": criteria})
        return YesNoResult(probability=_number(answer, "probability"))

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
        """
        answer = await self._ask(
            {"kind": "choice", "state": state, "criteria": criteria, "options": options}
        )
        label = str(answer.get("label", ""))
        if label not in options:
            raise ClassifierError(f"the LLM chose {label!r}, which is not an option")
        given = answer.get("probabilities") or {}
        probabilities = {o: float(given.get(o, 0.0)) for o in options}
        return ChoiceResult(
            label=label,
            probabilities=probabilities,
            confidence=probabilities[label] if probabilities[label] > 0 else 1.0,
        )

    async def score(self, state: ClassifierState, rubric: list[str], criteria: str) -> ScoreResult:
        """Ask where the state falls on an ordered rubric.

        Args:
            state: What the question is about.
            rubric: The levels in order, lowest first. At least two.
            criteria: What is being rated.

        Returns:
            The position on the rubric and how sure the LLM is. The
            probabilities put the LLM's confidence on the nearest level.
        """
        answer = await self._ask(
            {"kind": "score", "state": state, "criteria": criteria, "rubric": rubric}
        )
        score = _number(answer, "score")
        confidence = _number(answer, "confidence")
        nearest = min(range(len(rubric)), key=lambda i: abs(i - score)) if rubric else None
        return ScoreResult(
            score=score,
            probabilities={
                level: confidence if i == nearest else 0.0 for i, level in enumerate(rubric)
            },
            confidence=confidence,
        )

    async def _ask(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._owner is None:
            raise ClassifierError("LLMClassifier needs setup(worker) before it can answer")
        params = JobParams(name=CLASSIFY_JOB_NAME, payload=payload, timeout=self._timeout)
        try:
            async with self._owner.job(self._worker.name, params=params) as job:
                pass
        except JobError as e:
            raise ClassifierError(f"LLM classification failed: {e}") from e
        response = job.response or {}
        if "error" in response:
            raise ClassifierError(f"LLM classification failed: {response['error']}")
        return response


def _number(answer: dict[str, Any], key: str) -> float:
    try:
        return float(answer[key])
    except (KeyError, TypeError, ValueError) as e:
        raise ClassifierError(f"LLM answer has no usable '{key}'") from e
