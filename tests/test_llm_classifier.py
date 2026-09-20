#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for LLMClassifier and LLMClassifierWorker.

A scripted LLM service stands in for the model: each LLMContextFrame plays the
next scripted response, a tool call or text, so the tests exercise the real
worker, aggregators, tool loop and job plumbing under a WorkerRunner.
"""

import asyncio
from typing import Any

import pytest

from pipecat.classifiers.base_classifier import ClassifierError
from pipecat.classifiers.llm import LLMClassifier
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import FunctionCallFromLLM, LLMService
from pipecat.services.settings import LLMSettings
from pipecat.workers.base_worker import BaseWorker
from pipecat.workers.runner import WorkerRunner


class _ScriptedLLM(LLMService):
    """Plays one scripted response per LLMContextFrame.

    A step is ``("call", tool_name, args)`` or ``("text", str)``.
    """

    def __init__(self, runs: list[list[tuple]]):
        super().__init__(
            settings=LLMSettings(
                model="scripted",
                system_instruction=None,
                temperature=None,
                max_tokens=None,
                top_p=None,
                top_k=None,
                frequency_penalty=None,
                presence_penalty=None,
                seed=None,
                filter_incomplete_user_turns=False,
                user_turn_completion_config=None,
            )
        )
        self._runs = list(runs)
        self.contexts_seen: list[list[Any]] = []
        self.tools_seen: list[Any] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return
        self.contexts_seen.append(list(frame.context.get_messages()))
        self.tools_seen.append(frame.context.tools)
        steps = self._runs.pop(0) if self._runs else []
        await self.push_frame(LLMFullResponseStartFrame())
        calls = []
        for i, step in enumerate(steps):
            if step[0] == "text":
                await self.push_frame(LLMTextFrame(step[1]))
            else:
                _, name, args = step
                calls.append(
                    FunctionCallFromLLM(
                        context=frame.context,
                        tool_call_id=f"call-{len(self.contexts_seen)}-{i}",
                        function_name=name,
                        arguments=args,
                    )
                )
        if calls:
            await self.run_function_calls(calls)
        await self.push_frame(LLMFullResponseEndFrame())


async def _with_classifier(runs: list[list[tuple]], body, **kwargs):
    """Run ``body(classifier, llm)`` with the classifier set up under a WorkerRunner."""
    llm = _ScriptedLLM(runs)
    classifier = LLMClassifier(llm=llm, timeout=5.0, **kwargs)
    owner = BaseWorker("owner")
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(owner)
    result: dict[str, Any] = {}

    async def run_body():
        try:
            # As in a bot, the classifier is set up while the runner is running.
            await classifier.setup(owner)
            result["value"] = await body(classifier, llm)
        finally:
            await classifier.cleanup()
            await runner.cancel()

    await asyncio.wait_for(asyncio.gather(runner.run(), run_body()), timeout=15)
    return result["value"]


@pytest.mark.asyncio
async def test_yes_no_answers_from_the_tool_call():
    async def body(classifier, llm):
        return await classifier.yes_no("Please leave a message.", "is this a voicemail greeting?")

    result = await _with_classifier([[("call", "answer_yes_no", {"probability": 0.93})]], body)
    assert result.probability == 0.93


@pytest.mark.asyncio
async def test_choice_answers_with_a_probability_per_option():
    options = {"complete": "finished", "short": "cut off", "long": "needs time"}

    async def body(classifier, llm):
        return await classifier.choice("I'd like to", options, "is the turn over?")

    result = await _with_classifier(
        [
            [
                (
                    "call",
                    "answer_choice",
                    {"label": "short", "probabilities": {"short": 0.8, "complete": 0.2}},
                )
            ]
        ],
        body,
    )
    assert result.label == "short"
    assert result.probabilities == {"complete": 0.2, "short": 0.8, "long": 0.0}
    assert result.confidence == 0.8


@pytest.mark.asyncio
async def test_score_answers_with_the_nearest_level():
    rubric = ["calm", "impatient", "frustrated"]

    async def body(classifier, llm):
        return await classifier.score("This is the third time!", rubric, "how upset?")

    result = await _with_classifier(
        [[("call", "answer_score", {"score": 1.8, "confidence": 0.7})]], body
    )
    assert result.score == 1.8
    assert result.confidence == 0.7
    assert result.probabilities == {"calm": 0.0, "impatient": 0.0, "frustrated": 0.7}


@pytest.mark.asyncio
async def test_the_question_is_rendered_with_the_options_and_the_tool_to_call():
    async def body(classifier, llm):
        await classifier.choice(
            {"assistant": "Where to?", "user": "japan"}, {"a": "first", "b": "second"}, "which?"
        )
        return llm.contexts_seen[0]

    messages = await _with_classifier([[("call", "answer_choice", {"label": "a"})]], body)
    assert [m["role"] for m in messages] == ["user"]
    user = messages[0]["content"]
    assert user.startswith("which?")
    assert "- a: first" in user and "- b: second" in user
    assert '"user": "japan"' in user
    assert "Call answer_choice" in user


@pytest.mark.asyncio
async def test_the_tools_and_instructions_reach_the_llm():
    async def body(classifier, llm):
        await classifier.yes_no("hi", "?")
        return llm.tools_seen[0], llm._settings.system_instruction

    tools, instruction = await _with_classifier(
        [[("call", "answer_yes_no", {"probability": 0.5})]], body, instructions="Decide."
    )
    names = {t.name for t in tools.standard_tools}
    assert names == {"answer_yes_no", "answer_choice", "answer_score"}
    assert instruction == "Decide."


@pytest.mark.asyncio
async def test_each_question_starts_from_a_fresh_context():
    async def body(classifier, llm):
        await classifier.yes_no("first", "?")
        await classifier.yes_no("second", "?")
        return llm.contexts_seen

    contexts = await _with_classifier(
        [
            [("call", "answer_yes_no", {"probability": 0.1})],
            [("call", "answer_yes_no", {"probability": 0.2})],
        ],
        body,
    )
    assert [m["role"] for m in contexts[1]] == ["user"]
    assert "second" in contexts[1][0]["content"]


@pytest.mark.asyncio
async def test_keep_history_keeps_earlier_questions_and_answers():
    async def body(classifier, llm):
        await classifier.yes_no("first", "?")
        await classifier.yes_no("second", "?")
        return llm.contexts_seen

    contexts = await _with_classifier(
        [
            [("call", "answer_yes_no", {"probability": 0.1})],
            [("call", "answer_yes_no", {"probability": 0.2})],
        ],
        body,
        keep_history=True,
    )
    roles = [m["role"] for m in contexts[1]]
    assert roles[0] == "user"
    assert "first" in contexts[1][0]["content"]
    assert roles.count("user") == 2


@pytest.mark.asyncio
async def test_a_text_reply_is_an_error():
    async def body(classifier, llm):
        with pytest.raises(ClassifierError, match="text"):
            await classifier.yes_no("hi", "?")
        return True

    assert await _with_classifier([[("text", "I think yes.")]], body)


@pytest.mark.asyncio
async def test_a_label_outside_the_options_is_an_error():
    async def body(classifier, llm):
        with pytest.raises(ClassifierError, match="not an option"):
            await classifier.choice("hi", {"a": "", "b": ""}, "?")
        return True

    assert await _with_classifier([[("call", "answer_choice", {"label": "c"})]], body)


@pytest.mark.asyncio
async def test_the_wrong_tool_for_the_question_is_an_error():
    async def body(classifier, llm):
        with pytest.raises(ClassifierError, match="score question"):
            await classifier.yes_no("hi", "?")
        return True

    assert await _with_classifier([[("call", "answer_score", {"score": 1, "confidence": 1})]], body)


@pytest.mark.asyncio
async def test_asking_before_setup_is_an_error():
    classifier = LLMClassifier(llm=_ScriptedLLM([]))
    with pytest.raises(ClassifierError, match="setup"):
        await classifier.yes_no("hi", "?")


def test_is_not_calibrated():
    assert not LLMClassifier(llm=_ScriptedLLM([])).calibrated
