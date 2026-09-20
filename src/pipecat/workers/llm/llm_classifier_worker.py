#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""LLM classifier worker: an LLM in a pipeline that answers classification questions.

:class:`~pipecat.classifiers.llm.LLMClassifier` creates one of these for the
LLM it is given and sends every question to it as a ``classify`` job. The
LLM answers by calling one of three tools, one per kind of question, and the
tool's arguments are the answer. Nothing else sends this worker jobs.
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from pipecat.bus.messages import BusJobRequestMessage
from pipecat.frames.frames import (
    ErrorFrame,
    FunctionCallResultProperties,
    LLMMessagesAppendFrame,
    LLMUpdateSettingsFrame,
)
from pipecat.pipeline.job_decorator import job
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMUserAggregatorParams,
)
from pipecat.services.llm_service import FunctionCallParams, LLMService
from pipecat.services.settings import LLMSettings
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.utils.errors import ErrorCategory
from pipecat.workers.llm.llm_context_worker import LLMContextWorker
from pipecat.workers.llm.tool_decorator import tool

#: Name of the job a :class:`LLMClassifierWorker` handles.
CLASSIFY_JOB_NAME = "classify"

DEFAULT_INSTRUCTIONS = (
    "You are a classifier. Each message asks one question about some input and "
    "names the tool to answer with. Answer only by calling that tool, with your "
    "best judgment. Never reply with text."
)


@dataclass
class _Run:
    """One question in progress."""

    job_id: str
    kind: str
    answer: dict[str, Any] | None = None
    error: str | None = None
    finished: asyncio.Event = field(default_factory=asyncio.Event)


class LLMClassifierWorker(LLMContextWorker):
    """Answers classification questions with an LLM that calls a tool per answer.

    The worker keeps the LLM in a pipeline, so tools, metrics and settings
    work as they do everywhere else. A ``classify`` job carries one question;
    the worker renders it as a user message, runs the LLM once, and answers
    the job from the tool the LLM called. A run that ends with text and no
    tool call fails the job.

    By default the context is empty between questions. With ``keep_history``
    the last ``max_history`` messages stay, so the LLM sees its earlier
    answers. The instructions are the LLM's system instruction, set when the
    worker starts.

    Job contract (``@job(name="classify")``, one question at a time):

    - request payload: ``{"kind": "yes_no" | "choice" | "score", "state": ...,
      "criteria": str, "options": {label: description}, "rubric": [level]}``,
      with ``options`` for a choice and ``rubric`` for a score.
    - response: ``{"probability": float}``, ``{"label": str, "probabilities":
      {label: float}}`` or ``{"score": float, "confidence": float}``. A run
      without a usable tool call answers ``{"error": str}`` instead.
    """

    def __init__(
        self,
        *,
        llm: LLMService[Any],
        instructions: str | None = None,
        name: str | None = None,
        keep_history: bool = False,
        max_history: int = 20,
    ):
        """Initialize the worker.

        Args:
            llm: The LLM that answers the questions.
            instructions: System instructions. The default tells the LLM to
                answer only by calling the tool each question names.
            name: Worker name; auto-generated when omitted.
            keep_history: Whether earlier questions and answers stay in the
                context.
            max_history: With ``keep_history``, how many messages to keep
                after the instructions.
        """
        self._instructions = instructions or DEFAULT_INSTRUCTIONS
        self._keep_history = keep_history
        self._max_history = max_history
        self._run: _Run | None = None
        super().__init__(
            name,  # type: ignore[arg-type]  # None selects an auto-generated name
            llm=llm,
            active=True,
            context=LLMContext(tool_choice="required"),
            user_params=LLMUserAggregatorParams(user_turn_strategies=ExternalUserTurnStrategies()),
        )

        @self.assistant_aggregator.event_handler("on_assistant_turn_stopped")
        async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
            await self._on_assistant_turn_stopped(message)

        @self.event_handler("on_pipeline_error")
        async def on_pipeline_error(worker, frame: ErrorFrame):
            await self._on_pipeline_error(frame)

    @property
    def instructions(self) -> str:
        """The system instruction the LLM answers under."""
        return self._instructions

    async def on_activated(self, args: dict | None) -> None:
        """Give the LLM its instructions, then the tools.

        Args:
            args: Optional activation arguments.
        """
        await self.queue_frame(
            LLMUpdateSettingsFrame(delta=LLMSettings(system_instruction=self._instructions))
        )
        await super().on_activated(args)

    @job(name=CLASSIFY_JOB_NAME, sequential=True)
    async def classify(self, message: BusJobRequestMessage):
        """Answer one question.

        Args:
            message: The job request; see the class docstring for the payload.
        """
        payload = message.payload or {}
        kind = payload.get("kind")
        if kind not in ("yes_no", "choice", "score"):
            await self.send_job_response(
                message.job_id, {"error": f"unknown question kind {kind!r}"}
            )
            return

        self._reset_context()
        run = self._run = _Run(job_id=message.job_id, kind=kind)
        await self.queue_frame(
            LLMMessagesAppendFrame(
                messages=[{"role": "user", "content": _render(payload)}], run_llm=True
            )
        )
        try:
            await run.finished.wait()
        finally:
            self._run = None
        if run.answer is None:
            error = run.error or "the LLM answered with text instead of calling a tool"
            logger.warning(f"Worker '{self.name}': job {message.job_id} failed: {error}")
            await self.send_job_response(message.job_id, {"error": error})
            return
        await self.send_job_response(message.job_id, run.answer)

    @tool
    async def answer_yes_no(self, params: FunctionCallParams, probability: float):
        """Answer a yes or no question.

        Args:
            params: The function call, which delivers the result.
            probability: How likely the answer is yes, from 0 to 1.
        """
        await self._answer(params, "yes_no", {"probability": _unit(probability)})

    @tool
    async def answer_choice(
        self,
        params: FunctionCallParams,
        label: str,
        probabilities: dict[str, float] | None = None,
    ):
        """Answer a choice question.

        Args:
            params: The function call, which delivers the result.
            label: The option that fits best.
            probabilities: How likely each option is, keyed by option label,
                summing to 1. Leave it out if you can only name the option.
        """
        await self._answer(
            params,
            "choice",
            {
                "label": label,
                "probabilities": {k: _unit(v) for k, v in (probabilities or {}).items()},
            },
        )

    @tool
    async def answer_score(self, params: FunctionCallParams, score: float, confidence: float):
        """Answer a score question.

        Args:
            params: The function call, which delivers the result.
            score: The position on the scale, from 0 for the first level to
                one less than the number of levels. Decimals are allowed.
            confidence: How sure you are, from 0 to 1.
        """
        await self._answer(
            params, "score", {"score": float(score), "confidence": _unit(confidence)}
        )

    async def _answer(self, params: FunctionCallParams, kind: str, answer: dict[str, Any]):
        run = self._run
        if run is not None and run.answer is None:
            if run.kind == kind:
                run.answer = answer
            else:
                run.error = f"the LLM answered a {kind} question, not a {run.kind} one"
            run.finished.set()
        # One question is one run: the answer is in hand, so the LLM does not
        # run again on the tool result.
        await params.result_callback(
            {"recorded": True}, properties=FunctionCallResultProperties(run_llm=False)
        )

    async def _on_assistant_turn_stopped(self, message: AssistantTurnStoppedMessage):
        run = self._run
        if run is None or run.finished.is_set():
            return
        if self.assistant_aggregator.has_function_calls_in_progress:
            return
        run.error = "the LLM answered with text instead of calling a tool"
        run.finished.set()

    async def _on_pipeline_error(self, frame: ErrorFrame):
        run = self._run
        if run is None or frame.category == ErrorCategory.APPLICATION:
            return
        run.error = frame.error
        run.finished.set()

    def _reset_context(self):
        if not self._keep_history:
            self.context.set_messages([])
            return
        messages = self.context.messages
        if len(messages) > self._max_history:
            self.context.set_messages(messages[-self._max_history :])


def _render(payload: dict[str, Any]) -> str:
    """Write a question as the user message the LLM answers."""
    kind = payload["kind"]
    criteria = str(payload.get("criteria") or "")
    state = payload.get("state")
    state_text = state if isinstance(state, str) else json.dumps(state, indent=2)
    parts = [criteria] if criteria else []
    if kind == "choice":
        options = payload.get("options") or {}
        parts.append("Options:\n" + "\n".join(f"- {k}: {v}" for k, v in options.items()))
    elif kind == "score":
        rubric = payload.get("rubric") or []
        parts.append(
            "Scale, lowest first:\n" + "\n".join(f"{i}: {level}" for i, level in enumerate(rubric))
        )
    parts.append(f"Input:\n{state_text}")
    if kind == "yes_no":
        parts.append(
            "Call answer_yes_no with the probability, from 0 to 1, that the answer is yes."
        )
    elif kind == "choice":
        parts.append(
            "Call answer_choice with the option that fits best and a probability for every "
            "option, summing to 1."
        )
    else:
        last = max(len(payload.get("rubric") or []) - 1, 0)
        parts.append(
            f"Call answer_score with the position on the scale, from 0 to {last} (decimals "
            "allowed), and your confidence from 0 to 1."
        )
    return "\n\n".join(parts)


def _unit(value: Any) -> float:
    """Clamp a number the LLM wrote to the 0 to 1 range."""
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0
