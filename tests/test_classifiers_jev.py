#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json
import os
from collections.abc import Callable

import httpx
import pytest

from pipecat.classifiers.base_classifier import ClassifierError
from pipecat.classifiers.jev import JevClassifier, JevClient

USAGE = {"input_tokens": 12, "output_tokens": 3}


def _client(handler: Callable[[httpx.Request], httpx.Response], **kwargs) -> JevClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return JevClient(api_key="key", http_client=http, **kwargs)


def _reply(answer: dict) -> httpx.Response:
    return httpx.Response(
        200, json={"model": "jev-latest", "answers": {"answer": answer}, "usage": USAGE}
    )


class TestJevClient:
    @pytest.mark.asyncio
    async def test_sends_one_question_with_auth(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["auth"] = request.headers["Authorization"]
            seen["body"] = json.loads(request.content)
            return _reply({"type": "noul", "noul": 0.9})

        client = _client(handler)
        answer = await client.ask("hello", {"type": "noul", "instructions": "a greeting?"})

        assert answer == {"type": "noul", "noul": 0.9}
        assert seen["url"] == "https://api.typesafe.ai/v1/systemone"
        assert seen["auth"] == "Bearer key"
        assert seen["body"] == {
            "model": "jev-latest",
            "state": "hello",
            "questions": {"answer": {"type": "noul", "instructions": "a greeting?"}},
        }
        await client.close()

    @pytest.mark.asyncio
    async def test_counts_tokens_over_requests(self):
        client = _client(lambda request: _reply({"type": "noul", "noul": 0.5}))
        await client.ask("a", {"type": "noul", "instructions": "?"})
        await client.ask("b", {"type": "noul", "instructions": "?"})

        assert client.usage.input_tokens == 24
        assert client.usage.output_tokens == 6
        await client.close()

    @pytest.mark.asyncio
    async def test_retries_when_busy(self, monkeypatch):
        statuses = iter([429, 529])
        waits = []

        async def no_sleep(seconds):
            waits.append(seconds)

        monkeypatch.setattr("pipecat.classifiers.jev.asyncio.sleep", no_sleep)

        def handler(request: httpx.Request) -> httpx.Response:
            status = next(statuses, None)
            if status is not None:
                return httpx.Response(status)
            return _reply({"type": "noul", "noul": 0.7})

        client = _client(handler)
        answer = await client.ask("a", {"type": "noul", "instructions": "?"})

        assert answer["noul"] == 0.7
        assert waits == [0.25, 0.5]
        await client.close()

    @pytest.mark.asyncio
    async def test_gives_up_after_max_retries(self, monkeypatch):
        async def no_sleep(seconds):
            pass

        monkeypatch.setattr("pipecat.classifiers.jev.asyncio.sleep", no_sleep)
        client = _client(lambda request: httpx.Response(429), max_retries=2)

        with pytest.raises(ClassifierError, match="busy"):
            await client.ask("a", {"type": "noul", "instructions": "?"})
        await client.close()

    @pytest.mark.asyncio
    async def test_rejected_request_is_an_error(self):
        client = _client(lambda request: httpx.Response(401))

        with pytest.raises(ClassifierError, match="401"):
            await client.ask("a", {"type": "noul", "instructions": "?"})
        await client.close()

    @pytest.mark.asyncio
    async def test_unreachable_is_an_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down")

        client = _client(handler)

        with pytest.raises(ClassifierError, match="failed"):
            await client.ask("a", {"type": "noul", "instructions": "?"})
        await client.close()

    def test_needs_an_api_key(self):
        with pytest.raises(ValueError):
            JevClient(api_key="")


class TestJevClassifier:
    @pytest.mark.asyncio
    async def test_yes_no(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["question"] = json.loads(request.content)["questions"]["answer"]
            return _reply({"type": "noul", "noul": 0.95})

        classifier = JevClassifier(client=_client(handler))
        result = await classifier.yes_no("Please leave a message", "is this a voicemail greeting?")

        assert result.probability == 0.95
        assert seen["question"] == {"type": "noul", "instructions": "is this a voicemail greeting?"}
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_choice(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["question"] = json.loads(request.content)["questions"]["answer"]
            return _reply(
                {
                    "type": "choice",
                    "choice": "short",
                    "probabilities": {"complete": 0.1, "short": 0.88},
                    "confidence": 0.8,
                }
            )

        classifier = JevClassifier(client=_client(handler))
        options = {
            "complete": "the turn is over",
            "short": "a brief pause",
            "long": "asked for time",
        }
        result = await classifier.choice("I think, um", options, "is the user's turn over?")

        assert result.label == "short"
        assert result.probabilities == {"complete": 0.1, "short": 0.88, "long": 0.0}
        assert result.confidence == 0.8
        assert seen["question"] == {
            "type": "choice",
            "instructions": "is the user's turn over?",
            "criteria": options,
        }
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_score_keys_probabilities_by_level(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["question"] = json.loads(request.content)["questions"]["answer"]
            return _reply(
                {
                    "type": "score",
                    "score": 1.05,
                    "legend": {"0": "calm", "1": "frustrated", "2": "angry"},
                    "probabilities": {"0": 0.0, "1": 0.95, "2": 0.05},
                    "confidence": 0.92,
                }
            )

        classifier = JevClassifier(client=_client(handler))
        rubric = ["calm", "frustrated", "angry"]
        result = await classifier.score("This is the third time!", rubric, "how upset is the user?")

        assert result.score == 1.05
        assert result.probabilities == {"calm": 0.0, "frustrated": 0.95, "angry": 0.05}
        assert result.confidence == 0.92
        assert seen["question"] == {
            "type": "score",
            "instructions": "how upset is the user?",
            "criteria": rubric,
        }
        await classifier.client.close()

    @pytest.mark.asyncio
    async def test_missing_answer_field_is_an_error(self):
        classifier = JevClassifier(client=_client(lambda request: _reply({"type": "noul"})))

        with pytest.raises(ClassifierError, match="noul"):
            await classifier.yes_no("a", "?")
        await classifier.client.close()

    def test_needs_a_key_or_a_client(self):
        with pytest.raises(ValueError):
            JevClassifier()

    def test_is_calibrated(self):
        assert JevClassifier(api_key="key").calibrated

    @pytest.mark.asyncio
    async def test_cleanup_closes_only_an_owned_client(self):
        owned = JevClassifier(api_key="key")
        await owned.cleanup()
        assert owned.client._http.is_closed

        shared = JevClient(api_key="key")
        classifier = JevClassifier(client=shared)
        await classifier.cleanup()
        assert not shared._http.is_closed
        await shared.close()


@pytest.mark.skipif(not os.getenv("JEV_API_KEY"), reason="JEV_API_KEY not set")
class TestJevLive:
    @pytest.mark.asyncio
    async def test_three_questions(self):
        classifier = JevClassifier(api_key=os.environ["JEV_API_KEY"])
        try:
            greeting = "Hi, you've reached Sam. I can't take your call right now, leave a message."
            yes_no = await classifier.yes_no(greeting, "is this a voicemail greeting?")
            assert yes_no.probability > 0.5

            choice = await classifier.choice(
                "I'd like to book a table for, um",
                {
                    "complete": "the user finished",
                    "short": "a brief pause",
                    "long": "asked for time",
                },
                "is the user's turn over?",
            )
            assert choice.label in ("complete", "short", "long")
            assert abs(sum(choice.probabilities.values()) - 1.0) < 0.05

            score = await classifier.score(
                "This is the third time I call and nobody helps me!",
                ["calm", "impatient", "frustrated", "asking for a person"],
                "how upset is the user?",
            )
            assert 0 <= score.score <= 3
            assert classifier.client.usage.input_tokens > 0
        finally:
            await classifier.cleanup()
