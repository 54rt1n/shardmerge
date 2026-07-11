# shard/loramerge/residence.py
# Copyright (C) 2026 Martin Bukowski
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.

"""On-policy residence gate for merged-model generations.

A merge is only valid if its output stays in a stable rollout basin (see
research/ADDENDUM_STABLE_POLICY_MERGING.md and the on-policy-merge principle).
This module scores generated text for the off-policy collapse modes we have
actually hit -- repetition loops ("dewdrops of dewdrops"), entropy collapse
("de de a- l-"), empty/degenerate responses ("<turn|>"), and the OPPOSITE of
repetition: word-salad ("bottle nearly tips clatters cage bars liquid sloshes"),
where function words drop out and the output is a chain of related content words
with no sentence structure -- and returns a pass/fail verdict so a merge that
lands off-policy can be rejected automatically instead of caught by eye.

The word-salad mode (multiturn, sampled) is invisible to a distinct-token floor:
it has distinct ~1.0 (every word unique), the *anti*-repetition extreme. It is
caught instead by a function-word-ratio FLOOR -- grammatical English prose holds
a stable ~0.40 ratio of function words regardless of length, while salad collapses
to <0.12. See out/rollouts/multiturn-q4-*.md.

Pure text analysis, no model required: score the transcript a rollout produced.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re


# Defaults tuned against observed failures: the qat merge (distinct ~0.1),
# the equinox+egan slerp loop (a phrase repeated back-to-back), egan's empty reply.
MIN_TOKENS = 10
DISTINCT_FLOOR = 0.40      # below this = entropy collapse (whole-output repetition)
REP2_CEILING = 0.22        # most-common bigram share above this = global loop
REP3_CEILING = 0.16        # most-common trigram share above this = global loop
LOOP_UNIGRAM = 6           # same word >= this many times back-to-back = loop
LOOP_PHRASE = 3            # a 2-4 word phrase repeated >= this many times back-to-back = loop
FUNCTION_FLOOR = 0.22      # below this share of function words = word-salad (no syntax)
SALAD_MIN_TOKENS = 40      # short replies are legitimately high-distinct; only judge salad past this

# A minimal closed-class function-word set. Grammatical English prose runs ~0.40 of
# these regardless of length/genre; word-salad (content words only) collapses <0.12.
_FUNCTION_WORDS = frozenset(
    "the a an and or but nor of to in on at for with from by as is are was were be "
    "been being am it its this that these those he she they we you i his her their our "
    "your my me him them us into over under up down out off about then there here now "
    "not no so if when while because though although than which who whom whose what "
    "has have had do does did will would shall should can could may might must".split()
)


def _function_ratio(tokens: list[str]) -> float:
    words = [w.strip(".,!?;:\"'()[]—-").lower() for w in tokens]
    words = [w for w in words if w]
    if not words:
        return 0.0
    return sum(w in _FUNCTION_WORDS for w in words) / len(words)


@dataclass
class ResidenceScore:
    tokens: int
    distinct: float        # unique / total tokens (low = repetitive)
    rep2: float            # share of the most frequent bigram
    rep3: float            # share of the most frequent trigram
    loop: int              # longest back-to-back phrase repeat (local loop length)
    func: float            # share of function words (low = word-salad, no syntax)
    verdict: str           # "ok" | "empty" | "repetition" | "entropy_collapse" | "word_salad"

    @property
    def ok(self) -> bool:
        return self.verdict == "ok"


def _top_ngram_share(tokens: list[str], n: int) -> float:
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    return Counter(grams).most_common(1)[0][1] / len(grams)


def _max_phrase_loop(tokens: list[str], n: int) -> int:
    """Longest run of an ``n``-gram repeated immediately back-to-back."""
    best = 0
    L = len(tokens)
    i = 0
    while i + n <= L:
        gram = tokens[i : i + n]
        reps = 1
        while tokens[i + reps * n : i + (reps + 1) * n] == gram:
            reps += 1
        best = max(best, reps)
        i += 1
    return best


def score_text(text: str, *, min_tokens: int = MIN_TOKENS) -> ResidenceScore:
    """Score one generation for off-policy collapse."""
    tokens = text.split()
    n = len(tokens)
    if n < min_tokens:
        return ResidenceScore(n, 0.0, 0.0, 0.0, 0, 0.0, "empty")
    distinct = len(set(tokens)) / n
    rep2 = _top_ngram_share(tokens, 2)
    rep3 = _top_ngram_share(tokens, 3)
    loop1 = _max_phrase_loop(tokens, 1)
    loop_phrase = max(_max_phrase_loop(tokens, k) for k in (2, 3, 4))
    # Function-word density is a prose-only grammar signal. Fenced source code
    # naturally has very few English function words and must not be classified
    # as word salad. Keep code in the repetition statistics, but remove it from
    # the syntax check (including a fence truncated by max_new_tokens).
    prose = re.sub(r"```.*?(?:```|$)", " ", text, flags=re.DOTALL)
    prose_tokens = prose.split()
    func = _function_ratio(prose_tokens)
    if loop1 >= LOOP_UNIGRAM or loop_phrase >= LOOP_PHRASE:
        verdict = "repetition"          # local back-to-back loop
    elif rep2 > REP2_CEILING or rep3 > REP3_CEILING:
        verdict = "repetition"          # global n-gram domination
    elif distinct < DISTINCT_FLOOR:
        verdict = "entropy_collapse"
    elif len(prose_tokens) >= SALAD_MIN_TOKENS and func < FUNCTION_FLOOR:
        verdict = "word_salad"          # anti-repetition: content words, no syntax
    else:
        verdict = "ok"
    return ResidenceScore(n, round(distinct, 3), round(rep2, 3), round(rep3, 3),
                          max(loop1, loop_phrase), round(func, 3), verdict)


def score_rollout(rows: dict[str, str]) -> tuple[dict[str, ResidenceScore], bool]:
    """Score every ``tag -> text`` generation; pass only if all are ``ok``."""
    scores = {tag: score_text(text) for tag, text in rows.items()}
    return scores, all(s.ok for s in scores.values())


def render_scores(label: str, scores: dict[str, "ResidenceScore"]) -> list[str]:
    """Markdown rows summarizing one label's per-prompt residence verdicts."""
    lines = [
        f"### {label}",
        "",
        "| prompt | tokens | distinct | rep2 | rep3 | func | verdict |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for tag, s in scores.items():
        flag = "" if s.ok else " ⚠️"
        lines.append(f"| {tag} | {s.tokens} | {s.distinct} | {s.rep2} | {s.rep3} | {s.func} | {s.verdict}{flag} |")
    return lines
