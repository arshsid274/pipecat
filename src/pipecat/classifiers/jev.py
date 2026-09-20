#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Classifier backed by Jev, TypeSafe's hosted classification model.

:class:`JevClient` is the HTTP layer: one connection pool with HTTP/2, the
auth header, timeouts, retries when Jev is busy, and token accounting.
:class:`JevClassifier` turns each question into a request and each reply
into a result. Several classifiers can share one client.
"""

import asyncio
from typing import Any

from loguru import logger
from pydantic import BaseModel

from pipecat.classifiers.base_classifier import (
    BaseClassifier,
    ChoiceResult,
    ClassifierError,
    ClassifierState,
    ScoreResult,
    YesNoResult,
)
from pipecat.utils.network import exponential_backoff_time

try:
    import httpx
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Jev, you need to `pip install pipecat-ai[jev]`.")
    raise Exception(f"Missing module: {e}")

#: Jev answers a request with this status when the caller is rate limited.
_TOO_MANY_REQUESTS = 429
#: Jev answers a request with this status when it is temporarily overloaded.
_OVERLOADED = 529


class JevUsage(BaseModel):
    """Tokens a :class:`JevClient` has used so far.

    Parameters:
        input_tokens: Tokens sent, over every request.
        output_tokens: Tokens received, over every request.
    """

    input_tokens: int = 0
    output_tokens: int = 0


class JevClient:
    """HTTP client for Jev's ``systemone`` endpoint.

    Holds one HTTP/2 connection pool, so many small requests share a
    connection and can be in flight at the same time. Retries with backoff
    when Jev answers 429 or 529. Counts the tokens every request used in
    :attr:`usage`.

    A client owns a connection and must be closed with :meth:`close`.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.typesafe.ai",
        model: str = "jev-latest",
        timeout: float = 10.0,
        max_retries: int = 3,
        http_client: httpx.AsyncClient | None = None,
    ):
        """Initialize the client.

        Args:
            api_key: Jev API key.
            base_url: Where the API is served.
            model: The Jev model to ask.
            timeout: Seconds to wait for a reply before giving up.
            max_retries: How many times to retry a request Jev refused
                because it was busy.
            http_client: An HTTP client to send requests with instead of
                the one built here. Mostly for tests.
        """
        if not api_key:
            raise ValueError("JevClient needs an API key")
        self._model = model
        self._max_retries = max_retries
        self._usage = JevUsage()
        self._http = http_client or httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            http2=True,
        )
        if http_client is not None:
            self._http.base_url = httpx.URL(base_url)
            self._http.headers["Authorization"] = f"Bearer {api_key}"

    @property
    def usage(self) -> JevUsage:
        """Tokens used so far, over every request."""
        return self._usage

    async def close(self):
        """Close the connection pool."""
        await self._http.aclose()

    async def ask(self, state: ClassifierState, question: dict[str, Any]) -> dict[str, Any]:
        """Send one question about a state and return Jev's answer.

        Args:
            state: What the question is about.
            question: The question in Jev's own format: a ``type`` of
                ``noul``, ``choice`` or ``score``, ``instructions``, and
                ``criteria``.

        Returns:
            The answer in Jev's own format.

        Raises:
            ClassifierError: If Jev rejected the request, kept refusing it
                because it was busy, or could not be reached.
        """
        body = {"model": self._model, "state": state, "questions": {"answer": question}}
        attempt = 0
        while True:
            attempt += 1
            try:
                response = await self._http.post("/v1/systemone", json=body)
            except httpx.HTTPError as e:
                raise ClassifierError(f"Jev request failed: {e}") from e
            if response.status_code in (_TOO_MANY_REQUESTS, _OVERLOADED):
                if attempt > self._max_retries:
                    raise ClassifierError(
                        f"Jev is busy (HTTP {response.status_code}) after {attempt} attempts"
                    )
                wait = exponential_backoff_time(
                    attempt, min_wait=0.25, max_wait=2.0, multiplier=0.25
                )
                logger.debug(f"Jev answered {response.status_code}, retrying in {wait}s")
                await asyncio.sleep(wait)
                continue
            if response.status_code != 200:
                raise ClassifierError(f"Jev rejected the request: HTTP {response.status_code}")
            data = response.json()
            usage = data.get("usage") or {}
            self._usage.input_tokens += usage.get("input_tokens", 0)
            self._usage.output_tokens += usage.get("output_tokens", 0)
            try:
                return data["answers"]["answer"]
            except (KeyError, TypeError) as e:
                raise ClassifierError("Jev reply has no answer") from e


class JevClassifier(BaseClassifier):
    """Answers questions by asking Jev.

    Jev's probabilities are calibrated. Build one with an API key to get a
    client of its own, or pass a :class:`JevClient` to share one between
    several classifiers. A client the classifier created is closed in
    :meth:`cleanup`; a shared one is left to whoever made it.

    Example::

        client = JevClient(api_key=os.getenv("JEV_API_KEY"))
        turn_classifier = JevClassifier(client=client)
        voicemail_classifier = JevClassifier(client=client)
    """

    def __init__(self, *, api_key: str | None = None, client: JevClient | None = None):
        """Initialize the classifier.

        Args:
            api_key: Jev API key, when the classifier should have a client
                of its own.
            client: A client to share. One of ``api_key`` and ``client`` is
                required.
        """
        if client is None and not api_key:
            raise ValueError("JevClassifier needs an API key or a JevClient")
        self._owns_client = client is None
        self._client = client or JevClient(api_key=api_key or "")

    @property
    def calibrated(self) -> bool:
        """Jev's probabilities are calibrated."""
        return True

    @property
    def client(self) -> JevClient:
        """The client this classifier asks through."""
        return self._client

    async def cleanup(self):
        """Close the client if this classifier created it."""
        if self._owns_client:
            await self._client.close()

    async def yes_no(self, state: ClassifierState, criteria: str) -> YesNoResult:
        """Ask whether the state satisfies a criteria.

        Args:
            state: What the question is about.
            criteria: What is being checked for, as a yes or no question.

        Returns:
            How likely the answer is yes.
        """
        answer = await self._client.ask(state, {"type": "noul", "instructions": criteria})
        return YesNoResult(probability=_number(answer, "noul"))

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
        answer = await self._client.ask(
            state, {"type": "choice", "instructions": criteria, "criteria": options}
        )
        probabilities = answer.get("probabilities") or {}
        return ChoiceResult(
            label=str(answer.get("choice", "")),
            probabilities={label: float(probabilities.get(label, 0.0)) for label in options},
            confidence=_number(answer, "confidence"),
        )

    async def score(self, state: ClassifierState, rubric: list[str], criteria: str) -> ScoreResult:
        """Ask where the state falls on an ordered rubric.

        Args:
            state: What the question is about.
            rubric: The levels in order, lowest first. At least two.
            criteria: What is being rated.

        Returns:
            The position on the rubric and how likely each level is.
        """
        answer = await self._client.ask(
            state, {"type": "score", "instructions": criteria, "criteria": rubric}
        )
        # Jev keys level probabilities by position; the result keys them by level.
        probabilities = answer.get("probabilities") or {}
        return ScoreResult(
            score=_number(answer, "score"),
            probabilities={
                level: float(probabilities.get(str(index), 0.0))
                for index, level in enumerate(rubric)
            },
            confidence=_number(answer, "confidence"),
        )


def _number(answer: dict[str, Any], key: str) -> float:
    try:
        return float(answer[key])
    except (KeyError, TypeError, ValueError) as e:
        raise ClassifierError(f"Jev reply has no usable '{key}'") from e
