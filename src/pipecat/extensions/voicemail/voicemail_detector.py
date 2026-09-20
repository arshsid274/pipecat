#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Voicemail detection for outbound calls.

A bot that places a call needs to know whether a person answered or the call
went to voicemail. :class:`VoicemailDetector` listens to what the other side
says and asks a classifier; until it has an answer, :class:`TTSGate` holds
the bot's speech back so a voicemail greeting is never talked over.

Note:
    The voicemail module is optimized for text LLMs only.
"""

import asyncio
import warnings

from loguru import logger

from pipecat.classifiers.base_classifier import BaseClassifier, ClassifierError
from pipecat.classifiers.llm import LLMClassifier
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    StopFrame,
    SystemFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    WorkerFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.services.llm_service import LLMService
from pipecat.utils.sync.base_notifier import BaseNotifier
from pipecat.utils.sync.event_notifier import EventNotifier
from pipecat.workers.llm.llm_classifier_worker import DEFAULT_INSTRUCTIONS

# Lifecycle frames that must still flow after voicemail is detected.
_CLOSED_GATE_ALLOWLIST = (SystemFrame, EndFrame, StopFrame, WorkerFrame)

#: The question put to the classifier, with the transcript so far as the state.
VOICEMAIL_CRITERIA = (
    "A bot has placed an outbound phone call. This is what was heard after the "
    "call connected. Decide whether a person answered or the call went to "
    "voicemail."
)

#: The two answers and what each one looks like.
VOICEMAIL_OPTIONS = {
    "conversation": (
        "a person answered: a greeting such as 'hello?', 'hi', 'yeah?' or "
        "'John speaking'; a question to the caller such as 'who is this?' or "
        "'can I help you?'; spontaneous speech that expects a reply"
    ),
    "voicemail": (
        "an automated greeting or carrier message: 'you've reached', 'leave a "
        "message', 'I'm not available right now', 'call me back', 'mailbox is "
        "full', 'not in service', 'all circuits are busy', 'our office is "
        "currently closed'"
    ),
}


class TTSGate(FrameProcessor):
    """Holds the bot's speech until the voicemail decision is made.

    Placed right after the TTS service. TTS frames are buffered while the
    decision is pending; every other frame passes through. A conversation
    verdict releases the buffered frames in order, a voicemail verdict
    discards them, since they were meant for a person.
    """

    def __init__(self, conversation_notifier: BaseNotifier, voicemail_notifier: BaseNotifier):
        """Initialize the TTS gate.

        Args:
            conversation_notifier: Signals that a person answered and the
                buffered frames should play.
            voicemail_notifier: Signals that the call went to voicemail and
                the buffered frames should be dropped.
        """
        super().__init__()
        self._conversation_notifier = conversation_notifier
        self._voicemail_notifier = voicemail_notifier
        self._frame_buffer: list[tuple[Frame, FrameDirection]] = []
        self._gating_active = True
        self._conversation_task: asyncio.Task | None = None
        self._voicemail_task: asyncio.Task | None = None

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the processor with required components.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        self._conversation_task = self.create_task(self._wait_for_conversation())
        self._voicemail_task = self.create_task(self._wait_for_voicemail())

    async def cleanup(self):
        """Clean up the processor resources."""
        await super().cleanup()
        if self._conversation_task:
            await self.cancel_task(self._conversation_task)
            self._conversation_task = None
        if self._voicemail_task:
            await self.cancel_task(self._voicemail_task)
            self._voicemail_task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Buffer TTS frames while the decision is pending; pass the rest.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if self._gating_active and isinstance(
            frame, (TTSStartedFrame, TTSStoppedFrame, TTSTextFrame, TTSAudioRawFrame)
        ):
            self._frame_buffer.append((frame, direction))
        else:
            await self.push_frame(frame, direction)

    async def _wait_for_conversation(self):
        await self._conversation_notifier.wait()
        self._gating_active = False
        for frame, direction in self._frame_buffer:
            await self.push_frame(frame, direction)
        self._frame_buffer.clear()

    async def _wait_for_voicemail(self):
        await self._voicemail_notifier.wait()
        self._gating_active = False
        self._frame_buffer.clear()


class VoicemailDetector(FrameProcessor):
    """Decides whether a person answered an outbound call or it went to voicemail.

    Placed after the STT service. It passes every frame through and collects
    what the other side says. After each transcription it asks its classifier,
    in the background, whether the call reached a person or a voicemail, and
    keeps asking as more is said until the classifier is sure. Then it acts:

    - CONVERSATION: the bot's held-back speech is released and the call goes
      on as normal.
    - VOICEMAIL: the held-back speech is dropped, the pipeline is interrupted,
      no further input reaches the conversation, and ``on_voicemail_detected``
      fires once the greeting has finished, so the handler can leave a
      message.

    Example::

        detector = VoicemailDetector(classifier=JevClassifier(api_key=...))

        @detector.event_handler("on_voicemail_detected")
        async def handle_voicemail(processor):
            await processor.push_frame(TTSSpeakFrame("Please call me back."))

        pipeline = Pipeline([
            transport.input(),
            stt,
            detector.detector(),          # Classification
            context_aggregator.user(),
            llm,
            tts,
            detector.gate(),              # TTS gating
            transport.output(),
            context_aggregator.assistant(),
        ])

    Events:
        on_conversation_detected: A person answered. The handler receives the
            detector, which it can push frames through.
        on_voicemail_detected: The call went to voicemail and the greeting has
            been quiet for ``voicemail_response_delay`` seconds. The handler
            receives the detector, which it can push frames through.
    """

    CLASSIFIER_RESPONSE_INSTRUCTION = 'Respond with ONLY "CONVERSATION" if a person answered, or "VOICEMAIL" if it\'s voicemail/recording.'

    DEFAULT_SYSTEM_PROMPT = (
        """You are a voicemail detection classifier for an OUTBOUND calling system. A bot has called a phone number and you need to determine if a human answered or if the call went to voicemail based on the provided text.

HUMAN ANSWERED - LIVE CONVERSATION (respond "CONVERSATION"):
- Personal greetings: "Hello?", "Hi", "Yeah?", "John speaking"
- Interactive responses: "Who is this?", "What do you want?", "Can I help you?"
- Conversational tone expecting back-and-forth dialogue
- Questions directed at the caller: "Hello? Anyone there?"
- Informal responses: "Yep", "What's up?", "Speaking"
- Natural, spontaneous speech patterns
- Immediate acknowledgment of the call

VOICEMAIL SYSTEM (respond "VOICEMAIL"):
- Automated voicemail greetings: "Hi, you've reached [name], please leave a message"
- Phone carrier messages: "The number you have dialed is not in service", "Please leave a message", "All circuits are busy"
- Professional voicemail: "This is [name], I'm not available right now"
- Instructions about leaving messages: "leave a message", "leave your name and number"
- References to callback or messaging: "call me back", "I'll get back to you"
- Carrier system messages: "mailbox is full", "has not been set up"
- Business hours messages: "our office is currently closed"

"""
        + CLASSIFIER_RESPONSE_INSTRUCTION
    )

    def __init__(
        self,
        *,
        classifier: BaseClassifier | None = None,
        voicemail_response_delay: float = 2.0,
        decision_threshold: float = 0.7,
        llm: LLMService | None = None,
        custom_system_prompt: str | None = None,
    ):
        """Initialize the voicemail detector.

        Args:
            classifier: What decides between a person and a voicemail. It is
                asked a ``choice`` question with the transcript so far.
            voicemail_response_delay: Seconds of silence after a voicemail
                verdict before ``on_voicemail_detected`` fires, so the message
                is left after the greeting ends and the recording starts.
            decision_threshold: How sure the classifier has to be before the
                detector acts on its answer. Below it, the detector waits for
                more speech and asks again.
            llm: LLM service used for the classification.

                .. deprecated:: 1.12.0
                    Use ``classifier`` instead. Will be removed in 2.0.0.
            custom_system_prompt: System prompt for the ``llm``.

                .. deprecated:: 1.12.0
                    Use ``classifier`` instead. Will be removed in 2.0.0.
        """
        super().__init__()
        if llm is not None:
            warnings.warn(
                "VoicemailDetector's `llm` parameter is deprecated since 1.12.0 and will be "
                "removed in 2.0.0. Use `classifier` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if custom_system_prompt is not None:
            warnings.warn(
                "VoicemailDetector's `custom_system_prompt` parameter is deprecated since 1.12.0 "
                "and will be removed in 2.0.0. Use `classifier` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if classifier is None and llm is None:
            raise ValueError("VoicemailDetector needs a classifier")
        if classifier is None:
            # The old prompt asked for a one-word reply; the classifier's own
            # instructions, which ask for a tool call, come last and win.
            instructions = None
            if custom_system_prompt:
                instructions = f"{custom_system_prompt}\n\n{DEFAULT_INSTRUCTIONS}"
            classifier = LLMClassifier(llm=llm, instructions=instructions)  # type: ignore[arg-type]
        self._classifier = classifier
        self._voicemail_response_delay = voicemail_response_delay
        self._decision_threshold = decision_threshold

        self._conversation_notifier = EventNotifier()
        self._voicemail_notifier = EventNotifier()
        self._tts_gate = TTSGate(self._conversation_notifier, self._voicemail_notifier)

        self._transcript: list[str] = []
        self._decision: str | None = None
        self._classify_task: asyncio.Task | None = None
        self._transcript_changed = False

        # The voicemail handler fires once the greeting has been quiet for
        # the response delay; speech resets the wait.
        self._voicemail_task: asyncio.Task | None = None
        self._voicemail_event = asyncio.Event()
        self._voicemail_event.set()

        self._register_event_handler("on_conversation_detected")
        self._register_event_handler("on_voicemail_detected")

    def detector(self) -> "VoicemailDetector":
        """The processor to place after the STT service.

        Returns:
            This detector.
        """
        return self

    def gate(self) -> TTSGate:
        """The processor to place after the TTS service.

        Returns:
            The gate that holds speech until the decision is made.
        """
        return self._tts_gate

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the processor and its classifier.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        await self._classifier.setup(setup.pipeline_worker)
        self._voicemail_task = self.create_task(self._delayed_voicemail_handler())

    async def cleanup(self):
        """Clean up the processor and its classifier."""
        await super().cleanup()
        if self._classify_task:
            await self.cancel_task(self._classify_task)
            self._classify_task = None
        if self._voicemail_task:
            await self.cancel_task(self._voicemail_task)
            self._voicemail_task = None
        await self._classifier.cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Collect transcriptions, keep the voicemail timer, and gate after a voicemail.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            self._transcript.append(frame.text.strip())
            self._schedule_classification()
        elif isinstance(frame, UserStartedSpeakingFrame):
            if self._decision == "voicemail":
                self._voicemail_event.set()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            if self._decision == "voicemail":
                self._voicemail_event.clear()

        # After a voicemail verdict nothing more should reach the conversation,
        # only the frames that end or control the pipeline.
        if self._decision == "voicemail" and not isinstance(frame, _CLOSED_GATE_ALLOWLIST):
            return
        await self.push_frame(frame, direction)

    def _schedule_classification(self):
        if self._decision is not None:
            return
        if self._classify_task is not None and not self._classify_task.done():
            self._transcript_changed = True
            return
        self._transcript_changed = False
        self._classify_task = self.create_task(self._classify())

    async def _classify(self):
        while self._decision is None:
            transcript = " ".join(self._transcript)
            self._transcript_changed = False
            try:
                result = await self._classifier.choice(
                    transcript, VOICEMAIL_OPTIONS, VOICEMAIL_CRITERIA
                )
            except ClassifierError as e:
                logger.warning(f"{self}: classification failed: {e}")
            else:
                logger.debug(f"{self}: {result.label} ({result.confidence:.2f}) for {transcript!r}")
                if result.confidence >= self._decision_threshold:
                    await self._decide(result.label)
                    return
            if not self._transcript_changed:
                return

    async def _decide(self, label: str):
        self._decision = label
        if label == "voicemail":
            logger.info(f"{self}: VOICEMAIL detected")
            await self._voicemail_notifier.notify()
            await self.broadcast_interruption()
            self._voicemail_event.clear()
        else:
            logger.info(f"{self}: CONVERSATION detected")
            await self._conversation_notifier.notify()
            await self._call_event_handler("on_conversation_detected")

    async def _delayed_voicemail_handler(self):
        while True:
            try:
                await asyncio.wait_for(
                    self._voicemail_event.wait(), timeout=self._voicemail_response_delay
                )
                await asyncio.sleep(0.1)
            except TimeoutError:
                await self._call_event_handler("on_voicemail_detected")
                break
