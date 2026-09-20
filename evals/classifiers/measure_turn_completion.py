#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Measure a classifier on the turn-completion question.

Replays labeled user turns through a classifier, one at a time as the turn
path would, and reports how often it agrees with the label and how long each
answer takes. The labeled turns come from ``turn_completion.yaml`` next to
this script, plus every user line of the scripted release scenarios, which
are complete turns by construction.

Usage::

    JEV_API_KEY=... python evals/classifiers/measure_turn_completion.py
    python evals/classifiers/measure_turn_completion.py --no-context
    python evals/classifiers/measure_turn_completion.py --repeat 3
"""

import argparse
import asyncio
import os
import statistics
import sys
import time
from pathlib import Path

import yaml

from pipecat.classifiers.base_classifier import BaseClassifier, ClassifierError
from pipecat.classifiers.jev import JevClassifier

HERE = Path(__file__).parent
SCENARIOS = HERE.parent / "release" / "scenarios" / "scripted"

# The question, phrased after the turn-completion protocol the LLM follows.
CRITERIA = (
    "The user is talking to a voice assistant. Their words come from speech "
    "recognition, without punctuation, and may have been cut off. Decide whether "
    "the user's turn is complete. Complete means conversationally complete, not "
    "long: one word can be a complete answer, a question is complete, a "
    "correction is complete."
)
OPTIONS = {
    "complete": "the user has taken their turn and the assistant should answer",
    "short": (
        "the user stopped mid-sentence and will continue in a few seconds: the last "
        "words leave a phrase open, such as ending on a conjunction, a preposition, "
        "an article, or an unfinished list or number"
    ),
    "long": (
        "the user needs time to think or asked the assistant to wait, or has only "
        "acknowledged the question without answering it"
    ),
}


def load_turns(with_context: bool) -> list[dict]:
    turns = yaml.safe_load(open(HERE / "turn_completion.yaml"))["turns"]

    class Loader(yaml.SafeLoader):
        pass

    Loader.add_constructor("!include", lambda loader, node: None)
    for path in sorted(SCENARIOS.glob("*.yaml")):
        # The incomplete-turn scenarios cut lines off on purpose; the labeled
        # set above covers those.
        if path.name.startswith("filter_incomplete_turns"):
            continue
        data = yaml.load(open(path), Loader=Loader) or {}
        for scenario in data.get("scenarios") or []:
            for turn in scenario.get("turns") or []:
                user = turn.get("user")
                if isinstance(user, str) and user.strip():
                    turns.append({"user": _as_stt(user), "label": "complete", "source": path.name})
    if not with_context:
        for turn in turns:
            turn.pop("bot", None)
    return turns


def _as_stt(text: str) -> str:
    """Make a scripted line look like a transcript: lower case, no punctuation."""
    return "".join(c for c in text.lower() if c.isalnum() or c in " '").strip()


async def measure(classifier: BaseClassifier, turns: list[dict], repeat: int) -> None:
    rows = []
    for _ in range(repeat):
        for turn in turns:
            state = {"user": turn["user"]}
            if turn.get("bot"):
                state = {"assistant": turn["bot"], "user": turn["user"]}
            started = time.perf_counter()
            try:
                result = await classifier.choice(state, OPTIONS, CRITERIA)
                predicted, confidence = result.label, result.confidence
            except ClassifierError as e:
                predicted, confidence = f"error: {e}", 0.0
            rows.append(
                {
                    **turn,
                    "predicted": predicted,
                    "confidence": confidence,
                    "ms": (time.perf_counter() - started) * 1000,
                }
            )
    report(rows)


def report(rows: list[dict]) -> None:
    labels = ["complete", "short", "long"]
    latencies = sorted(r["ms"] for r in rows)
    agree = sum(r["predicted"] == r["label"] for r in rows)
    print(f"\n{len(rows)} turns")
    print(
        f"latency  p50 {statistics.median(latencies):.0f} ms   "
        f"p95 {latencies[int(len(latencies) * 0.95) - 1]:.0f} ms   "
        f"max {latencies[-1]:.0f} ms"
    )
    print(f"agreement {agree}/{len(rows)} = {agree / len(rows):.1%}")
    incomplete = [r for r in rows if r["label"] != "complete"]
    false_complete = [r for r in incomplete if r["predicted"] == "complete"]
    print(
        f"false complete {len(false_complete)}/{len(incomplete)} = "
        f"{len(false_complete) / max(len(incomplete), 1):.1%}  (incomplete turns judged complete)"
    )
    print("\nconfusion (rows: label, columns: predicted)")
    print(f"{'':10s}" + "".join(f"{p:>10s}" for p in labels + ["error"]))
    for label in labels:
        counts = []
        for predicted in labels:
            counts.append(sum(r["label"] == label and r["predicted"] == predicted for r in rows))
        counts.append(sum(r["label"] == label and r["predicted"].startswith("error") for r in rows))
        print(f"{label:10s}" + "".join(f"{c:>10d}" for c in counts))
    misses = [r for r in rows if r["predicted"] != r["label"]]
    if misses:
        print("\ndisagreements")
        for r in misses:
            bot = f"  (bot: {r['bot']})" if r.get("bot") else ""
            print(
                f"  {r['label']:8s} -> {r['predicted']:8s} conf={r['confidence']:.2f}  "
                f"{r['user']!r}{bot}"
            )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-context", action="store_true", help="drop the bot's previous line")
    parser.add_argument("--repeat", type=int, default=1, help="run the set this many times")
    args = parser.parse_args()

    api_key = os.getenv("JEV_API_KEY")
    if not api_key:
        sys.exit("JEV_API_KEY is not set")
    classifier = JevClassifier(api_key=api_key)
    try:
        turns = load_turns(with_context=not args.no_context)
        print(f"{len(turns)} labeled turns, context {'off' if args.no_context else 'on'}")
        await measure(classifier, turns, args.repeat)
        print(
            f"\ntokens  in {classifier.client.usage.input_tokens}  out {classifier.client.usage.output_tokens}"
        )
    finally:
        await classifier.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
