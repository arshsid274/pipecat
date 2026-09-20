#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
import warnings

from pipecat.classifiers.base_classifier import (
    BaseClassifier,
    ChoiceResult,
    ClassifierError,
    ClassifierState,
    ScoreResult,
    YesNoResult,
)
from pipecat.extensions.voicemail.voicemail_detector import (
    VOICEMAIL_OPTIONS,
    VoicemailDetector,
    _LLMPromptClassifier,
)
from pipecat.frames.frames import (
    EndWorkerFrame,
    Frame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    TTSTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import LLMService
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.utils.text.base_text_aggregator import AggregationType

# Long enough for a verdict to reach the delayed voicemail handler, which
# fires VOICEMAIL_DELAY after the verdict.
VERDICT_SETTLE = 1.0
VOICEMAIL_DELAY = 0.1


class _FakeClassifier(BaseClassifier):
    """Answers the voicemail question from a scripted list of answers."""

    def __init__(self, *answers: tuple[str, float] | Exception):
        self.answers = list(answers)
        self.asked: list[ClassifierState] = []
        self.setup_worker = None
        self.cleaned_up = False

    @property
    def calibrated(self) -> bool:
        return True

    async def setup(self, worker):
        self.setup_worker = worker

    async def cleanup(self):
        self.cleaned_up = True

    async def yes_no(self, state, criteria) -> YesNoResult:
        raise NotImplementedError

    async def choice(self, state, options, criteria) -> ChoiceResult:
        self.asked.append(state)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        label, confidence = answer
        return ChoiceResult(
            label=label,
            probabilities={o: float(o == label) for o in options},
            confidence=confidence,
        )

    async def score(self, state, rubric, criteria) -> ScoreResult:
        raise NotImplementedError


class _Passthrough(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


def _names(frames) -> list[str]:
    return [type(f).__name__ for f in frames]


def _said(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text=text, user_id="", timestamp="")


def _detector(*answers) -> tuple[VoicemailDetector, _FakeClassifier]:
    classifier = _FakeClassifier(*answers)
    detector = VoicemailDetector(classifier=classifier, voicemail_response_delay=VOICEMAIL_DELAY)
    return detector, classifier


class TestVoicemailDetectorVerdicts(unittest.IsolatedAsyncioTestCase):
    async def test_voicemail_verdict_fires_handler_with_the_transcript(self):
        detector, classifier = _detector(("voicemail", 0.95))
        fired = []

        @detector.event_handler("on_voicemail_detected")
        async def _on_voicemail(processor):
            fired.append(processor)

        await run_test(
            detector,
            frames_to_send=[_said("Hi, you've reached Sam."), SleepFrame(VERDICT_SETTLE)],
            start_timeout=5.0,
        )
        self.assertEqual(fired, [detector])
        self.assertEqual(classifier.asked, ["Hi, you've reached Sam."])

    async def test_conversation_verdict_fires_handler(self):
        detector, _ = _detector(("conversation", 0.9))
        fired = []

        @detector.event_handler("on_conversation_detected")
        async def _on_conversation(processor):
            fired.append(processor)

        await run_test(
            detector,
            frames_to_send=[_said("Hello?"), SleepFrame(VERDICT_SETTLE)],
            start_timeout=5.0,
        )
        self.assertEqual(fired, [detector])

    async def test_unsure_answer_waits_for_more_speech(self):
        detector, classifier = _detector(("voicemail", 0.4), ("voicemail", 0.95))
        fired = []

        @detector.event_handler("on_voicemail_detected")
        async def _on_voicemail(processor):
            fired.append(processor)

        await run_test(
            detector,
            frames_to_send=[
                _said("Hi,"),
                SleepFrame(0.2),
                _said("you've reached Sam. Leave a message."),
                SleepFrame(VERDICT_SETTLE),
            ],
            start_timeout=5.0,
        )
        self.assertEqual(len(fired), 1)
        self.assertEqual(classifier.asked, ["Hi,", "Hi, you've reached Sam. Leave a message."])

    async def test_classifier_error_does_not_decide(self):
        detector, classifier = _detector(ClassifierError("down"), ("conversation", 0.9))
        fired = []

        @detector.event_handler("on_conversation_detected")
        async def _on_conversation(processor):
            fired.append(processor)

        await run_test(
            detector,
            frames_to_send=[
                _said("Hello?"),
                SleepFrame(0.2),
                _said("Anyone there?"),
                SleepFrame(VERDICT_SETTLE),
            ],
            start_timeout=5.0,
        )
        self.assertEqual(len(fired), 1)
        self.assertEqual(len(classifier.asked), 2)

    async def test_setup_and_cleanup_reach_the_classifier(self):
        detector, classifier = _detector(("conversation", 0.9))
        await run_test(detector, frames_to_send=[_said("Hello?")], start_timeout=5.0)
        self.assertIsNotNone(classifier.setup_worker)
        self.assertTrue(classifier.cleaned_up)


class TestVoicemailDetectorGating(unittest.IsolatedAsyncioTestCase):
    async def test_conversation_releases_held_speech(self):
        detector, _ = _detector(("conversation", 0.9))
        down, _ = await run_test(
            Pipeline([detector, detector.gate()]),
            frames_to_send=[
                TTSStartedFrame(),
                TTSTextFrame("Hi, this is Jamie.", aggregated_by=AggregationType.SENTENCE),
                SleepFrame(0.2),
                _said("Hello?"),
                SleepFrame(VERDICT_SETTLE),
            ],
            start_timeout=5.0,
        )
        names = _names(down)
        self.assertIn("TTSTextFrame", names)
        self.assertLess(names.index("TTSStartedFrame"), names.index("TTSTextFrame"))

    async def test_voicemail_drops_held_speech_and_blocks_later_input(self):
        detector, _ = _detector(("voicemail", 0.95))
        down, _ = await run_test(
            Pipeline([detector, detector.gate()]),
            frames_to_send=[
                TTSStartedFrame(),
                TTSTextFrame("Hi, this is Jamie.", aggregated_by=AggregationType.SENTENCE),
                SleepFrame(0.2),
                _said("Please leave a message."),
                SleepFrame(VERDICT_SETTLE),
                LLMTextFrame(text="should not reach the conversation"),
                SleepFrame(0.2),
            ],
            start_timeout=5.0,
        )
        names = _names(down)
        self.assertNotIn("TTSTextFrame", names)
        self.assertNotIn("LLMTextFrame", names)

    async def test_transcriptions_pass_through_before_a_verdict(self):
        detector, _ = _detector(("conversation", 0.9))
        down, _ = await run_test(
            detector,
            frames_to_send=[_said("Hello?"), SleepFrame(VERDICT_SETTLE)],
            start_timeout=5.0,
        )
        self.assertIn("TranscriptionFrame", _names(down))


class TestVoicemailDetectorEndWorkerFrame(unittest.IsolatedAsyncioTestCase):
    async def test_handler_pushing_upstream_ends_worker(self):
        detector, _ = _detector(("voicemail", 0.95))

        @detector.event_handler("on_voicemail_detected")
        async def _on_voicemail(processor: FrameProcessor):
            await processor.push_frame(
                EndWorkerFrame(reason="Voicemail detected."), FrameDirection.UPSTREAM
            )

        _down, up = await run_test(
            detector,
            frames_to_send=[_said("Please leave a message."), SleepFrame(VERDICT_SETTLE)],
            start_timeout=5.0,
        )
        self.assertTrue(
            any(isinstance(f, EndWorkerFrame) for f in up),
            f"EndWorkerFrame did not escape upstream: {_names(up)}",
        )

    async def test_handler_pushing_downstream_ends_worker(self):
        detector, _ = _detector(("voicemail", 0.95))

        @detector.event_handler("on_voicemail_detected")
        async def _on_voicemail(processor: FrameProcessor):
            await processor.push_frame(
                EndWorkerFrame(reason="Voicemail detected."), FrameDirection.DOWNSTREAM
            )

        down, _up = await run_test(
            detector,
            frames_to_send=[_said("Please leave a message."), SleepFrame(VERDICT_SETTLE)],
            start_timeout=5.0,
        )
        self.assertTrue(
            any(isinstance(f, EndWorkerFrame) for f in down),
            f"EndWorkerFrame did not escape downstream: {_names(down)}",
        )

    async def test_voicemail_verdict_lets_upstream_end_from_main_pipeline(self):
        detector, _ = _detector(("voicemail", 0.95))
        ender = _Passthrough()

        @detector.event_handler("on_voicemail_detected")
        async def _on_verdict(_processor: FrameProcessor):
            await ender.push_frame(
                EndWorkerFrame(reason="VOICEMAIL detected."), FrameDirection.UPSTREAM
            )

        _down, up = await run_test(
            Pipeline([detector, ender]),
            frames_to_send=[_said("Please leave a message."), SleepFrame(VERDICT_SETTLE)],
            start_timeout=5.0,
        )
        self.assertTrue(
            any(isinstance(f, EndWorkerFrame) for f in up),
            f"EndWorkerFrame did not escape upstream after VOICEMAIL: {_names(up)}",
        )

    async def test_conversation_verdict_lets_upstream_end_from_main_pipeline(self):
        detector, _ = _detector(("conversation", 0.9))
        ender = _Passthrough()

        @detector.event_handler("on_conversation_detected")
        async def _on_verdict(_processor: FrameProcessor):
            await ender.push_frame(
                EndWorkerFrame(reason="CONVERSATION detected."), FrameDirection.UPSTREAM
            )

        _down, up = await run_test(
            Pipeline([detector, ender]),
            frames_to_send=[_said("Hello?"), SleepFrame(VERDICT_SETTLE)],
            start_timeout=5.0,
        )
        self.assertTrue(
            any(isinstance(f, EndWorkerFrame) for f in up),
            f"EndWorkerFrame did not escape upstream after CONVERSATION: {_names(up)}",
        )


class _FakeLLM(LLMService):
    def __init__(self, reply: str):
        super().__init__()
        self.reply = reply
        self.seen = []

    async def run_inference(self, context, max_tokens=None, system_instruction=None):
        self.seen.append((context.get_messages(), system_instruction))
        return self.reply


class TestDeprecatedLLMParameter(unittest.IsolatedAsyncioTestCase):
    def test_llm_parameter_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            VoicemailDetector(llm=_FakeLLM("CONVERSATION"))
        self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))

    def test_needs_a_classifier_or_an_llm(self):
        with self.assertRaises(ValueError):
            VoicemailDetector()

    async def test_llm_reply_becomes_a_verdict(self):
        llm = _FakeLLM("VOICEMAIL")
        classifier = _LLMPromptClassifier(llm, "prompt")
        result = await classifier.choice("Leave a message.", VOICEMAIL_OPTIONS, "")

        self.assertEqual(result.label, "voicemail")
        self.assertEqual(result.confidence, 1.0)
        self.assertFalse(classifier.calibrated)
        messages, instruction = llm.seen[0]
        self.assertEqual(messages[0]["content"], "Leave a message.")
        self.assertEqual(instruction, "prompt")

    async def test_llm_reply_without_a_verdict_is_an_error(self):
        classifier = _LLMPromptClassifier(_FakeLLM("I am not sure"), "prompt")
        with self.assertRaises(ClassifierError):
            await classifier.choice("Hello?", VOICEMAIL_OPTIONS, "")
