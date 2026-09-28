"""
EmptyTalkerRobot
================

A speaking robot built on top of ``MaybeRickSanchez/EmptyMicroRobot``
(imported as ``from empty import EmptyRobot``).

This is **not** a chatbot and there is **no LLM anywhere in here** -- no
pretrained encoder, no tokenizer from a model, no weights, no inference
server, no network. Every sentence this robot says is *constructed* out of
hashed statistics it derived from text files you gave it, scored by the
EmptyMicroRobot decision core, and re-weighted by the feedback it has
received. If the data is not there, the robot has nothing to say, and it
says so.

    pip install numpy            # that's it
    # plus the foundation, importable as `empty`:
    #   git clone https://github.com/MaybeRickSanchez/EmptyMicroRobot


What the foundation does, and what is added on top
--------------------------------------------------
``EmptyRobot`` is a bounded associative decision memory: feature-hashed
vectors, episodic traces, a vote, a bounded candidate generator, an
empirical world model, hard safety constraints, and ``reward()`` /
``punish()`` that actually move numbers. That is the engine. It does not
know what a word is, what a sentence is, or how long one should be.

This module supplies the four things the engine cannot invent for itself:

1. **A hashed word-position store** (``WordPositionIndex``). Text files are
   read, tokenized into words *with their positions*, and folded into
   sorted ``uint64`` columns. What is left in RAM is a positional
   transition table -- "after word *h* at position *i*, word *j* came
   next, *n* times" -- and nothing else. No Python strings per transition,
   no tuples, no per-sentence objects. The only strings kept are the capped
   lexicon needed to turn a hash back into a word at speak time; the
   original text is never retained. ``memory_report()`` prints the measured
   bytes-per-token so the claim is checkable rather than asserted.
2. **A persona** (``Persona``). Point it at a character's complete dialogue
   and it measures that character's speech -- formality, aggression,
   valence, verbosity, certainty, lexical richness -- from the data, finds
   the phrases that are *over*-represented relative to a background
   corpus, and keeps the lines themselves as quotable expressions. Give it
   a Marvel transcript and you get that character's speaking personality,
   with no character-specific code anywhere in this file.
3. **A planner** (``SpeechPlanner``). It decides, per utterance, how long
   the sentence should be (sampled from a learned length distribution or
   from the relation's own spec -- there is no hardcoded cap), which words
   to seed from, and how to walk the transition table to assemble a line,
   then hands several finished candidate lines to the foundation to pick
   between.
4. **A feedback judge** (``FeedbackJudge``). It scores whether an utterance
   was good or bad from two independent sources -- an *intrinsic*,
   self-contained check (were the required words present, was the length
   right, is it repetitive, is it new, does it fit the persona) and a
   *learned* judge (a second ``EmptyRobot`` instance trained on
   ``good``/``bad`` for the feature profile of previous utterances, read out
   through the margin between the two options and shrunk toward "no opinion"
   until it has evidence). The two are blended into one signed reward, the
   reward is pushed back into the foundation, and that reward is what moves
   the ``creativity`` scalar.

   The scalar is not a ratchet on "was this rewarded". It is driven by one
   measured comparison: **do new constructions do better than familiar
   ones?** Reward is binned by whether the line was more novel than what
   this robot usually produces, and the two running means are subtracted. In
   an environment where everything is rewarded the gap stays at zero and the
   robot does not get noisier for nothing; where new lines are rewarded and
   repeated ones are not, the gap opens and it loosens up; where new lines
   are punished it tightens. The scalar is the temperature of the word
   sampler, so "gradually becoming more creative" is one number you can
   watch in ``stats()`` under ``feedback``.

A fifth piece, ``RelationBook``, is smaller but load-bearing: it holds the
mapping from probable inputs to probable outputs (which keywords and
exemplars mean this relation, which words the reply must contain, roughly
how long it should be) and ranks them for an incoming utterance using both a
literal local score and the foundation's learned vote.

The same hashed store is also used as **memory**: ``MemoryLedger`` indexes
every exchange with ``WordPositionIndex`` and recalls it by hashed overlap
plus recency decay, so "what did I say last time I was asked about this"
costs the same as a dictionary lookup and the memory is bounded.

The JSON contract
-----------------
One ``.json`` file is both the input and the output. On the way in it
declares the persona, the relations (which probable inputs map to which
probable outputs, which words an output must contain, roughly how long it
should be) and which ``.txt`` files to digest. On the way out it carries
the persona measurements, the packed hashed index, the experience ledger,
the feedback history and the learned weights, so a robot can be saved and
resumed exactly. See ``SCHEMA`` and ``DEFAULT_CONFIG``.

    from emptytalker import EmptyTalkerRobot

    talker = EmptyTalkerRobot()                     # works with zero files
    talker.ingest_txt("thanos.txt")                 # any amount of .txt
    talker.build_persona("Thanos", dialogue_paths=["thanos.txt"])

    u = talker.speak("hello there, what do you want?")
    print(u.text, u.confidence)
    u.reward()                                      # or u.punish()
    print(talker.stats()["feedback"]["creativity"])

    print(talker.explain(u))                        # every decision, in words
    print(talker.report())                          # the whole robot, briefly

    talker.save("thanos.json")
    same = EmptyTalkerRobot.load("thanos.json")

Cost
----
Measured on one laptop, numpy only, 1 400-token corpus, a persona fitted from
it, and a warm foundation of ~700 traces:

    speak()             p50  6.2 ms   p95  9.6 ms
    speak() + reward()  p50 11.3 ms   p95 15.6 ms
    ingest              55 000 tokens/s   (single pass, no index rebuild)
    speak() by width    3.8 / 4.8 / 6.7 / 9.5 ms   at width 2 / 3 / 5 / 8

That is a turn-based-dialogue budget, not a 60 Hz one: fine for an NPC line
per player action, not for a line per rendered frame. Two knobs move it, and
they are worth knowing about. ``speak(width=...)`` sets how many finished
lines the walk builds per call, and the cost is close to linear in it. And
``limits["seed_budget"]`` bounds how many distinct lines ever get the
planner's prior written into the core as a starting belief; past that the
foundation ranks on real feedback alone, which is both cheaper and more
honest. The hashed store itself is not the bottleneck -- ingestion is fast
and the store never grows with corpus size past its caps. The foundation's
decision pass over its own memories is.

Memory, measured rather than asserted: one 12 KiB chunk of English prose
costs 38.7 bytes/token (1.8x better than holding the same text as a list of
token lists); the same text 30x over costs 1.3 bytes/token (54x better), which
is 94 KiB resident for 317 KiB of source. The store is O(distinct pairs), not
O(tokens), so the advantage grows with repetition. A corpus of entirely
*unique* words has no advantage at all, because distinct pairs then equal
tokens; ``memory_report()`` prints ``tokens_per_pair`` so you can see where on
that curve any given corpus sits rather than being handed a flattering
average.

Using it as a game / application component
------------------------------------------
    # a "voice" bound to a game event vocabulary
    talker.add_relation("player_died", {
        "input": {"keywords": ["died", "death", "kill", "dead"]},
        "output": {"must_include": ["dead"], "min_words": 3, "max_words": 9},
    })

    line = talker.speak("the player just died", relation="player_died")
    hud.say(line.text)                              # your game's text sink
    line.reward() if player_was_tamed else line.punish()

    # the foundation's own decision API is still reachable, unmodified,
    # for the non-speaking half of the brain:
    move = talker.decide({"hp": 0.1, "ammo": 0}, options=["flee", "reload", "hide"])

    # or hand the whole thing an event dict and get one line back:
    line = talker.voice({"event": "npc_greeted", "mood": "hostile"})

What this is not
----------------
* It is not a language model and it does not understand anything. It walks
  a hashed positional transition table. Ask it a question whose words never
  appeared in the data and it will either fail the similarity floor and say
  so, or produce a confident-sounding non sequitur from a bigram it has
  seen. Both are documented behaviours, not bugs to be surprised by.
* Sentences are assembled, not selected. Grammaticality comes from the
  corpus, not from a parser. Long outputs drift. ``max_words: null`` in a
  relation spec deliberately removes the length cap and lets the length come
  from the learned corpus distribution -- the response space is not
  enumerated, so it is not fixed, but it is also not infinite.
* ``Persona`` is measured, not authored. The six trait dimensions are
  computed from counts with hand-written word lists (slang markers,
  intensifiers, hedges, valence). They are crude, they are documented, and
  they are overridable in the JSON -- a Marvel character is recognised by
  *how often* those markers occur, not by understanding the character.
* The learned good/bad judge is only as good as the feedback you give it.
  With no feedback it collapses back onto the intrinsic score, which is a
  hand-written heuristic and says so.
* Confidence is the foundation's documented heuristic composition, not a
  probability. Read ``calibration_report()`` on the core before you trust
  it.
* The hashed store is lossy by construction: 64-bit hashes collide, and a
  collision silently merges two words. At 30k words the birthday
  probability is ~2.5e-11 per pair, which is not zero and is not checked.

Dependencies
------------
``numpy`` is required. ``empty`` (the EmptyMicroRobot foundation) must be
importable. Nothing else is used, and nothing reaches the network.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import threading
import time
import zlib
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # the foundation this is built on
    from empty import EmptyRobot
except ImportError as _exc:  # pragma: no cover - environment problem, not logic
    raise ImportError(
        "emptytalker needs the EmptyMicroRobot foundation importable as "
        "`empty` (git clone https://github.com/MaybeRickSanchez/EmptyMicroRobot "
        "and put empty.py on sys.path, or pip install it)."
    ) from _exc


# ---------------------------------------------------------------------- #
# Constants
# ---------------------------------------------------------------------- #

SCHEMA = "emptytalker/1"

_MASK64 = (1 << 64) - 1
_MIX_A = 0xBF58476D1CE4E5B9
_MIX_B = 0x94D049BB133111EB
_GOLDEN = 0x9E3779B97F4A7C15

# Namespaces, so a word hash can never be confused with a position hash or
# a memory hash. blake2b keys, all under the 64-byte limit.
_NS_WORD = b"etr.word"
_NS_STATE = b"etr.state"
_NS_LINE = b"etr.line"

# A word is a run of letters in *any* script (optionally glued by an
# apostrophe), or a number. Everything else that is not whitespace is its own
# token, so punctuation is learned as data and can be placed by position.
#
# ``[^\W\d_]`` is "word character that is neither a digit nor an underscore",
# which under Python's Unicode-aware ``\w`` means a letter in whatever
# alphabet the text is written in. A ``[A-Za-z]`` class here would look fine
# on English and silently produce a zero-word vocabulary on anything else --
# the file would report 116 tokens ingested and 0 words, which is the worst
# possible failure mode for a component that is supposed to be reusable.
#
# Scripts that do not separate words with spaces (Chinese, Japanese) will
# arrive as very long "words". That is a real limitation of a
# position-based model with no segmenter, and it is documented as one rather
# than papered over.
_TOKEN_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?|\d+(?:[.,]\d+)?|[^\s\w]|_")

_ASCII_NO_SPACE_BEFORE = frozenset(",.;:!?)]}%'\"")
_ASCII_NO_SPACE_AFTER = frozenset("([{'\"")
_punct_cache: Dict[str, bool] = {}


def _wordish(token: str) -> bool:
    """Is this token a word rather than punctuation? Script-agnostic."""
    if not token:
        return False
    ch = token[0]
    if ch.isascii():
        return ch.isalnum()
    return ch.isalnum()


def _no_space_before(token: str) -> bool:
    """Punctuation that hugs the token in front of it. ASCII is a fast path;
    anything else is classified once and memoised."""
    if token.isascii():
        return token in _ASCII_NO_SPACE_BEFORE
    hit = _punct_cache.get(token)
    if hit is None:
        hit = not token[:1].isalnum()
        _punct_cache[token] = hit
    return hit


def _no_space_after(token: str) -> bool:
    """An opening bracket or quote, which hugs the token in front of it."""
    if token.isascii():
        return token in _ASCII_NO_SPACE_AFTER
    return token in ("(", "[", "{") or token[:1] in ("'", "\u201c", "\u2018")


def _join(tokens: Sequence[str]) -> str:
    """Render a token list, fixing up the spacing around punctuation."""
    out: List[str] = []
    for token in tokens:
        if not out:
            out.append(token)
            continue
        if _no_space_before(token) or _no_space_after(out[-1]):
            out.append(token)
        else:
            out.append(" ")
            out.append(token)
    return "".join(out)


def _tokenize(text: Any) -> List[str]:
    """Split text into words and punctuation, keeping both."""
    if text is None:
        return []
    if not isinstance(text, str):
        text = str(text)
    return _TOKEN_RE.findall(text)


def _words_of(tokens: Sequence[str]) -> List[str]:
    return [t for t in tokens if _wordish(t)]


def _cap(text: str) -> str:
    """Capitalize the first letter and tidy the ends."""
    text = text.strip()
    if not text:
        return text
    for i, ch in enumerate(text):
        if ch.isalpha():
            return text[:i] + ch.upper() + text[i + 1:]
    return text


_STANDALONE_I_RE = re.compile(r"\bi\b")
_CONTRACT_I_RE = re.compile(r"\bi'(m|ve|ll|d)\b", re.IGNORECASE)


def _fix_case(text: str) -> str:
    """
    The corpus stores every word lowercased, so a walk that lands on ``i``
    renders as ``i``. Fixing that at render time rather than storing a
    second capitalised copy of every word is the whole reason the lexicon
    can be one packed blob instead of two.
    """
    if not text:
        return text
    text = _STANDALONE_I_RE.sub("I", text)
    return _CONTRACT_I_RE.sub(lambda m: "I'" + m.group(1), text)


# ---------------------------------------------------------------------- #
# Hashing
# ---------------------------------------------------------------------- #


def _h64(text: str, ns: bytes = _NS_WORD) -> int:
    """A stable 64-bit hash of a string.

    ``hash()`` is unusable here: Python salts ``str`` hashing per process,
    so every save file would only load in the interpreter that wrote it.
    blake2b is C-speed, stable across runs and machines, and accepts a
    namespacing key.
    """
    return int.from_bytes(
        hashlib.blake2b(text.encode("utf-8"), digest_size=8, key=ns).digest(),
        "little",
    )


def _mix(x: int) -> int:
    """splitmix64 finalizer -- avalanche, so position and word bits of a
    combined hash do not correlate with either input."""
    x &= _MASK64
    x ^= x >> 30
    x = (x * _MIX_A) & _MASK64
    x ^= x >> 27
    x = (x * _MIX_B) & _MASK64
    x ^= x >> 31
    return x


def _at(wh: int, pos: int) -> int:
    """Hash of (word, position). Position is folded in with a multiplicative
    odd constant so that the same word at slot 3 and slot 30 are different
    keys -- this is the whole 'word-position data' idea in one line."""
    return _mix(wh ^ ((int(pos) * _GOLDEN) & _MASK64))


def _state_h(text: str) -> int:
    return _h64(text, _NS_STATE)


# ---------------------------------------------------------------------- #
# Packed hashed tables
# ---------------------------------------------------------------------- #


class _CountIndex:
    """``uint64 key -> uint32 count``, held as two sorted numpy columns.

    Lookup is ``np.searchsorted``, which is O(log n) in C with no Python
    objects per entry. This is the shape every "just use a dict" design
    ends up in, minus the dict.
    """

    __slots__ = ("_keys", "_counts", "_buf_k", "_buf_c", "_flush_at")

    def __init__(self, flush_at: int = 65_536) -> None:
        self._keys = np.zeros(0, dtype=np.uint64)
        self._counts = np.zeros(0, dtype=np.uint32)
        self._buf_k: List[int] = []
        self._buf_c: List[int] = []
        self._flush_at = int(flush_at)

    # -- writing ---------------------------------------------------------
    def add(self, key: int, count: int = 1) -> None:
        self._buf_k.append(int(key) & _MASK64)
        self._buf_c.append(int(count))
        if len(self._buf_k) >= self._flush_at:
            self._flush()

    def _flush(self) -> None:
        if not self._buf_k:
            return
        keys = np.fromiter(self._buf_k, dtype=np.uint64, count=len(self._buf_k))
        cnts = np.fromiter(self._buf_c, dtype=np.uint64, count=len(self._buf_c))
        self._buf_k.clear()
        self._buf_c.clear()
        keys, cnts = self._merge(keys, cnts)
        self._keys, self._counts = keys, cnts

    def _merge(self, keys: np.ndarray, cnts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._keys.size == 0:
            # Still has to aggregate: a corpus smaller than ``flush_at`` is
            # flushed exactly once, and returning the buffer verbatim would
            # store one row per *token* instead of one per distinct key.
            order = np.argsort(keys, kind="stable")
            keys, cnts = keys[order], cnts[order]
            starts = np.flatnonzero(np.concatenate(
                ([True], keys[1:] != keys[:-1])))
            return (np.ascontiguousarray(keys[starts]),
                    np.add.reduceat(cnts, starts).astype(np.uint32, copy=False))
        allk = np.concatenate((self._keys.astype(np.uint64), keys))
        allc = np.concatenate((self._counts.astype(np.uint64), cnts))
        order = np.argsort(allk, kind="stable")
        allk, allc = allk[order], allc[order]
        starts = np.flatnonzero(np.concatenate(([True], allk[1:] != allk[:-1])))
        sums = np.add.reduceat(allc, starts)
        return (np.ascontiguousarray(allk[starts]),
                sums.astype(np.uint32, copy=False))

    # -- reading ---------------------------------------------------------
    def get(self, key: int, default: int = 0) -> int:
        key = int(key) & _MASK64
        i = int(np.searchsorted(self._keys, key))
        if i < self._keys.size and int(self._keys[i]) == key:
            return int(self._counts[i])
        return default

    def top(self, k: int) -> List[Tuple[int, int]]:
        """The ``k`` most frequent keys, most frequent first."""
        self._flush()
        if self._keys.size == 0:
            return []
        k = int(min(k, self._keys.size))
        if k >= self._keys.size:
            idx = np.argsort(-self._counts.astype(np.int64), kind="stable")[:k]
        else:
            part = np.argpartition(-self._counts.astype(np.int64), k - 1)[:k]
            idx = part[np.argsort(-self._counts[part].astype(np.int64), kind="stable")]
        return [(int(self._keys[i]), int(self._counts[i])) for i in idx]

    def items(self) -> Iterable[Tuple[int, int]]:
        self._flush()
        return zip(self._keys.tolist(), self._counts.tolist())

    def prune(self, keep: int) -> int:
        """Keep only the ``keep`` most frequent entries. Returns how many
        were dropped. This is what makes the store bounded on a corpus
        larger than the configured caps."""
        self._flush()
        if self._keys.size <= keep:
            return 0
        dropped = self._keys.size - int(keep)
        idx = np.argpartition(-self._counts.astype(np.int64), int(keep) - 1)[:keep]
        idx = idx[np.argsort(-self._counts[idx].astype(np.int64), kind="stable")]
        self._keys = np.ascontiguousarray(self._keys[idx])
        self._counts = np.ascontiguousarray(self._counts[idx])
        return int(dropped)

    # -- plumbing --------------------------------------------------------
    def clear(self) -> None:
        self._keys = np.zeros(0, dtype=np.uint64)
        self._counts = np.zeros(0, dtype=np.uint32)
        self._buf_k.clear()
        self._buf_c.clear()

    @property
    def nbytes(self) -> int:
        self._flush()
        return int(self._keys.nbytes + self._counts.nbytes)

    def __len__(self) -> int:
        self._flush()
        return int(self._keys.size)

    def to_json(self) -> Dict[str, Any]:
        self._flush()
        return {
            "keys": _pack(self._keys),
            "counts": _pack(self._counts),
        }

    @classmethod
    def from_json(cls, blob: Dict[str, Any]) -> "_CountIndex":
        obj = cls()
        obj._keys = _unpack(blob["keys"], np.uint64)
        obj._counts = _unpack(blob["counts"], np.uint32)
        return obj


class _GroupIndex:
    """``(uint64 group, uint64 member) -> uint32 count`` as three sorted
    columns, grouped so that *every* member of one group is a contiguous
    slice. This is the transition table: group = (word, position), member =
    the word that came next."""

    __slots__ = ("_g", "_m", "_c", "_buf_g", "_buf_m", "_buf_c", "_flush_at")

    def __init__(self, flush_at: int = 65_536) -> None:
        self._g = np.zeros(0, dtype=np.uint64)
        self._m = np.zeros(0, dtype=np.uint64)
        self._c = np.zeros(0, dtype=np.uint32)
        self._buf_g: List[int] = []
        self._buf_m: List[int] = []
        self._buf_c: List[int] = []
        self._flush_at = int(flush_at)

    def add(self, group: int, member: int, count: int = 1) -> None:
        self._buf_g.append(int(group) & _MASK64)
        self._buf_m.append(int(member) & _MASK64)
        self._buf_c.append(int(count))
        if len(self._buf_g) >= self._flush_at:
            self._flush()

    def _flush(self) -> None:
        if not self._buf_g:
            return
        g = np.fromiter(self._buf_g, dtype=np.uint64, count=len(self._buf_g))
        m = np.fromiter(self._buf_m, dtype=np.uint64, count=len(self._buf_m))
        c = np.fromiter(self._buf_c, dtype=np.uint64, count=len(self._buf_c))
        self._buf_g.clear()
        self._buf_m.clear()
        self._buf_c.clear()
        if self._g.size:
            g = np.concatenate((self._g, g))
            m = np.concatenate((self._m, m))
            c = np.concatenate((self._c, c))
        order = np.lexsort((m, g))
        g, m, c = g[order], m[order], c[order]
        fresh = np.ones(g.size, dtype=bool)
        if g.size > 1:
            fresh[1:] = (g[1:] != g[:-1]) | (m[1:] != m[:-1])
        starts = np.flatnonzero(fresh)
        self._g = np.ascontiguousarray(g[starts])
        self._m = np.ascontiguousarray(m[starts])
        self._c = np.ascontiguousarray(
            np.add.reduceat(c, starts).astype(np.uint32, copy=False))

    def slice(self, group: int) -> Tuple[np.ndarray, np.ndarray]:
        """Every member of ``group`` with its count. Empty arrays if none."""
        self._flush()
        group = int(group) & _MASK64
        lo = int(np.searchsorted(self._g, group, "left"))
        hi = int(np.searchsorted(self._g, group, "right"))
        if hi <= lo:
            return _EMPTY_U64, _EMPTY_U32
        return self._m[lo:hi], self._c[lo:hi]

    def prune(self, keep: int) -> int:
        """Drop the rarest (group, member) pairs down to ``keep``."""
        self._flush()
        if self._g.size <= keep:
            return 0
        dropped = int(self._g.size - keep)
        idx = np.argpartition(-self._c.astype(np.int64), int(keep) - 1)[:keep]
        idx = idx[np.argsort(-self._c[idx].astype(np.int64), kind="stable")]
        self._g = np.ascontiguousarray(self._g[idx])
        self._m = np.ascontiguousarray(self._m[idx])
        self._c = np.ascontiguousarray(self._c[idx])
        return dropped

    def clear(self) -> None:
        self._g = np.zeros(0, dtype=np.uint64)
        self._m = np.zeros(0, dtype=np.uint64)
        self._c = np.zeros(0, dtype=np.uint32)
        self._buf_g.clear()
        self._buf_m.clear()
        self._buf_c.clear()

    @property
    def nbytes(self) -> int:
        self._flush()
        return int(self._g.nbytes + self._m.nbytes + self._c.nbytes)

    def __len__(self) -> int:
        self._flush()
        return int(self._g.size)

    def to_json(self) -> Dict[str, Any]:
        self._flush()
        return {"g": _pack(self._g), "m": _pack(self._m), "c": _pack(self._c)}

    @classmethod
    def from_json(cls, blob: Dict[str, Any]) -> "_GroupIndex":
        obj = cls()
        obj._g = _unpack(blob["g"], np.uint64)
        obj._m = _unpack(blob["m"], np.uint64)
        obj._c = _unpack(blob["c"], np.uint32)
        return obj


_EMPTY_U64 = np.zeros(0, dtype=np.uint64)
_EMPTY_U32 = np.zeros(0, dtype=np.uint32)


def _pack(arr: np.ndarray) -> str:
    """numpy column -> compact base64 for the JSON state file."""
    raw = zlib.compress(np.ascontiguousarray(arr).tobytes(), 6)
    return base64.b64encode(raw).decode("ascii")


def _unpack(blob: str, dtype: Any) -> np.ndarray:
    raw = zlib.decompress(base64.b64decode(blob.encode("ascii")))
    return np.frombuffer(raw, dtype=dtype).copy()


# ---------------------------------------------------------------------- #
# Word-position index
# ---------------------------------------------------------------------- #


class WordPositionIndex:
    """
    Hashed word-position statistics, held in sorted ``uint64`` columns.

    Ingest reads text, splits it into word tokens *and their positions*, and
    records five things:

    ``_uni``    word hash               -> occurrences
    ``_pos``    (position, word hash)   -> count     "what word goes in slot i"
    ``_wpos``   (word hash, position)   -> count     "where does this word go"
    ``_nxt``    (word@pos hash, next)   -> count     the positional transition table
    ``_bg``     (word hash, next)       -> count     a position-free bigram fallback
    plus ``_end``  line-final token     -> count     so terminators are learned
    and     ``_lens`` line-length bucket -> count     so length can be learned

    Nothing else survives ingestion. There is no per-sentence object, no
    list of tokens, no dict of tuples -- the corpus is *destroyed* on the
    way in and only the counts remain. The one exception is the capped
    lexicon needed to turn a hash back into a word at speak time, and even
    that is not a dict: it is three parallel numpy columns (sorted hashes,
    offsets, and one concatenated blob), which is roughly 14 bytes a word
    instead of the ~130 a ``{str: int}`` costs.

    Caps (all constructor arguments): ``max_vocab`` words, ``max_pairs``
    distinct (position, word) pairs, ``max_transitions`` distinct
    (context, next) pairs. Exceeding one prunes the rarest entries rather
    than the process, so a 500 MB transcript costs about what a 5 MB one
    does.

    The numbers are in the module docstring under *Cost* -- short version:
    1.3 bytes per token on repeated English, 54x smaller than keeping the
    same text as a list of token lists, and O(distinct pairs) rather than
    O(tokens) so the advantage grows with repetition rather than with the
    file.
    """

    def __init__(
        self,
        *,
        max_vocab: int = 30_000,
        max_pairs: int = 250_000,
        max_transitions: int = 1_000_000,
        max_lines: int = 400_000,
        keep_ratio: float = 0.75,
    ) -> None:
        self.max_vocab = int(max_vocab)
        self.max_pairs = int(max_pairs)
        self.max_transitions = int(max_transitions)
        self.max_lines = int(max_lines)
        self.keep_ratio = float(keep_ratio)

        # -- the lexicon, packed ----------------------------------------
        # NOT a dict. A dict of str->int costs ~130 bytes per word (two
        # str headers, an int, two table slots, two pointers). The packed
        # form is 8 bytes of hash, 4 bytes of offset, and the characters:
        # ~14 bytes for a five-letter word. That is the single biggest
        # memory win in this file and it is why there is no ``_lex`` dict.
        self._lex_h = np.zeros(0, dtype=np.uint64)    # sorted word hashes
        self._lex_off = np.zeros(1, dtype=np.uint32)  # N+1 offsets
        self._lex_blob = b""                          # lowercased utf-8
        self._pend: Dict[str, int] = {}               # not yet packed
        self._pend_rev: Dict[int, str] = {}
        self._memo: Dict[str, int] = {}               # bounded read cache
        self._memo_cap = 4096
        # punctuation is not vocabulary, but it *is* emittable (the walk
        # needs to be able to land on a comma). Capped hard, since there
        # are only a handful of marks a person would ever want spoken.
        self._punct: Dict[int, str] = {}
        self._punct_cap = 64
        self._vocab_open = True

        self._uni = _CountIndex()
        self._pos = _GroupIndex()    # group=position, member=word
        self._wpos = _GroupIndex()   # group=word,     member=position (lazy)
        self._wpos_ready = False
        self._nxt = _GroupIndex()    # group=word@pos, member=next word
        self._bg = _GroupIndex()     # group=word,     member=next word
        self._end = _CountIndex()    # line-final token
        self._lens = _CountIndex()   # power-of-two line-length bucket

        self.n_docs = 0
        self.n_lines = 0
        self.n_tokens = 0
        self.n_raw_bytes = 0
        self.sources: List[str] = []
        self.pruned = 0
        self.words_dropped = 0

    # -- lexicon ---------------------------------------------------------
    @property
    def vocab_open(self) -> bool:
        return self._vocab_open

    def _iter_lexicon(self) -> Iterable[Tuple[int, str]]:
        """Every ``(hash, word)`` currently held, packed or pending."""
        n = int(self._lex_h.size)
        if n:
            lengths = (self._lex_off[1:] - self._lex_off[:-1]).astype(np.int64)
            for i in range(n):
                lo = int(self._lex_off[i])
                yield (int(self._lex_h[i]),
                       self._lex_blob[lo:lo + int(lengths[i])].decode("utf-8", "replace"))
        for word, wh in self._pend.items():
            yield (wh, word)

    def _flush_lexicon(self, force: bool = False) -> None:
        """
        Fold pending words into the packed columns: sort, dedupe by hash,
        rebuild the blob and the offsets.

        Amortised on purpose -- one O(n log n) pass every ``_memo_cap`` new
        words rather than an insertion per word, which would make ingesting
        a large transcript quadratic.
        """
        if not self._pend or (not force and len(self._pend) < self._memo_cap):
            return
        entries = sorted(self._iter_lexicon(), key=lambda e: e[0])
        blob = bytearray()
        offs = [0]
        hashes: List[int] = []
        last: Optional[int] = None
        for wh, word in entries:
            if wh == last:
                continue  # hash collision or a re-added word
            last = wh
            hashes.append(wh)
            blob.extend(word.encode("utf-8"))
            offs.append(len(blob))
        self._lex_h = np.fromiter(hashes, dtype=np.uint64, count=len(hashes))
        self._lex_off = np.fromiter(offs, dtype=np.uint32, count=len(offs))
        self._lex_blob = bytes(blob)
        self._pend.clear()
        self._pend_rev.clear()

    def word_hash(self, word: str) -> Optional[int]:
        """
        The hash of a word, or ``None`` when the vocabulary is closed and
        this word was never admitted. ``None`` is not an error path: the
        caller uses it to break the transition chain, so statistics are
        never recorded for a word the robot could not physically say.
        """
        w = word.lower()
        got = self._memo.get(w)
        if got is not None:
            return got
        pending = self._pend.get(w)
        if pending is not None:
            if len(self._memo) >= self._memo_cap:
                self._memo.clear()
            self._memo[w] = pending
            return pending
        wh = _h64(w)
        if self._lex_h.size:
            i = int(np.searchsorted(self._lex_h, wh))
            if i < self._lex_h.size and int(self._lex_h[i]) == wh:
                if len(self._memo) >= self._memo_cap:
                    self._memo.clear()
                self._memo[w] = wh
                return wh
        if not self._vocab_open:
            self.words_dropped += 1
            return None
        if self._lex_h.size + len(self._pend) >= self.max_vocab:
            self._shrink_vocab()
            if not self._vocab_open:
                self.words_dropped += 1
                return None
        self._pend[w] = wh
        self._pend_rev[wh] = w
        if len(self._pend) >= self._memo_cap:
            self._flush_lexicon(force=True)
        return wh

    def word(self, wh: int) -> Optional[str]:
        """Hash back to a word, or ``None``. Three lookups, all O(log n)
        or O(1); no string is ever duplicated to make this work."""
        wh = int(wh) & _MASK64
        token = self._punct.get(wh)
        if token is not None:
            return token
        pending = self._pend_rev.get(wh)
        if pending is not None:
            return pending
        if not self._lex_h.size:
            return None
        i = int(np.searchsorted(self._lex_h, wh))
        if i >= self._lex_h.size or int(self._lex_h[i]) != wh:
            return None
        lo = int(self._lex_off[i])
        hi = int(self._lex_off[i + 1])
        return self._lex_blob[lo:hi].decode("utf-8", "replace")

    def known(self, word: str) -> bool:
        w = word.lower()
        if w in self._memo or w in self._pend:
            return True
        if not self._lex_h.size:
            return False
        wh = _h64(w)
        i = int(np.searchsorted(self._lex_h, wh))
        return i < self._lex_h.size and int(self._lex_h[i]) == wh

    def vocab(self) -> int:
        return int(self._lex_h.size + len(self._pend))

    def add_text(self, text: str, *, source: str = "inline", weight: int = 1) -> "WordPositionIndex":
        """
        Digest one document. Sentences and lines are split on ``.!?`` and
        newlines, so the position counter restarts per line and the
        learned "which word is last" is a real line-final distribution.
        Returns ``self``, so ingestion chains.
        """
        if not text:
            return self
        self.n_docs += 1
        self.n_raw_bytes += len(text.encode("utf-8"))
        if source not in self.sources:
            self.sources.append(source)
        for line in _split_lines(text):
            self.add_line(line, weight=weight)
        return self

    def add_line(self, line: str, *, weight: int = 1) -> None:
        tokens = _tokenize(line)
        if not tokens:
            return
        self.n_lines += 1
        self.n_tokens += len(tokens) * weight

        hashes: List[Optional[int]] = []
        for token in tokens:
            if _wordish(token):
                hashes.append(self.word_hash(token))
            else:
                # punctuation is emittable but is not vocabulary: it gets a
                # namespaced hash and a place in a hard-capped table, so the
                # walk can land on a comma or a full stop without a second
                # copy of the corpus existing anywhere
                ph = _h64(token, _NS_LINE)
                self._punct.setdefault(ph, token)
                hashes.append(ph)

        w = weight
        word_positions: List[Tuple[int, int]] = []
        prev: Optional[int] = None
        for i, wh in enumerate(hashes):
            if wh is None:
                # word not emittable -> break the chain here rather than
                # recording statistics that could never be used to speak
                prev = None
                continue
            if _wordish(tokens[i]):
                word_positions.append((i, wh))
                self._uni.add(wh, w)
                self._pos.add(i, wh, w)
                self._wpos.add(wh, i, w)
                if prev is not None:
                    self._nxt.add(_at(prev, i - 1), wh, w)
                    self._bg.add(prev, wh, w)
                prev = wh

        # line-final *punctuation*, and a power-of-two length bucket. Only
        # punctuation counts: a line that simply runs out of words has not
        # taught the robot anything about how to stop, and recording the last
        # word as if it were a terminator is how "after weather" ends up
        # being a learned place to stop talking.
        for tail in range(len(tokens) - 1, -1, -1):
            if hashes[tail] is not None and not _wordish(tokens[tail]):
                self._end.add(hashes[tail], w)
                break
        nwords = max(1, len(word_positions))
        self._lens.add(1 << (nwords.bit_length() - 1), w)

        if len(word_positions) >= self.max_lines:
            return
        if self._nxt.__len__() > self.max_transitions:
            self._prune_to_fit()
        elif self._pos.__len__() > self.max_pairs:
            self._prune_to_fit()

    def _shrink_vocab(self) -> None:
        """
        Close the vocabulary at ``max_vocab``, keeping only the words that
        are actually frequent. Words past the cap are counted but not
        stored, so they cannot be spoken -- an explicit, documented trade of
        coverage for a hard memory bound. The packed blob is rebuilt from
        the survivors, so the dropped words' bytes are released rather than
        merely hidden.
        """
        target = int(self.max_vocab * self.keep_ratio)
        self._uni.prune(target)
        keep = {wh for wh, _ in self._uni.top(target)}
        self._flush_lexicon(force=True)
        survivors = [(wh, word) for wh, word in self._iter_lexicon() if wh in keep]
        blob = bytearray()
        offs = [0]
        hashes: List[int] = []
        for wh, word in sorted(survivors):
            hashes.append(wh)
            blob.extend(word.encode("utf-8"))
            offs.append(len(blob))
        self._lex_h = np.fromiter(hashes, dtype=np.uint64, count=len(hashes))
        self._lex_off = np.fromiter(offs, dtype=np.uint32, count=len(offs))
        self._lex_blob = bytes(blob)
        self._pend.clear()
        self._pend_rev.clear()
        self._memo.clear()
        self._vocab_open = len(hashes) < self.max_vocab
        self.pruned += 1

    def _prune_to_fit(self) -> None:
        self.pruned += 1
        self._flush_lexicon(force=True)
        self._nxt.prune(int(self.max_transitions * self.keep_ratio))
        self._bg.prune(int(self.max_transitions * self.keep_ratio * 0.5))
        self._pos.prune(int(self.max_pairs * self.keep_ratio))
        if self._wpos_ready:
            self._wpos.prune(int(self.max_pairs * self.keep_ratio))
        self._shrink_vocab()

    def close(self) -> None:
        """Flush every buffer and enforce every cap once, at the end of
        ingest. Calling this is what turns a stream of documents into a
        settled, queryable store."""
        for table in (self._uni, self._end, self._lens):
            table._flush()
        for table in (self._pos, self._wpos, self._nxt, self._bg):
            table._flush()
        self._flush_lexicon(force=True)
        if not self._vocab_open or self.vocab() > self.max_vocab:
            self._shrink_vocab()
        self._nxt.prune(self.max_transitions)
        self._bg.prune(self.max_transitions)
        self._pos.prune(self.max_pairs)
        if self._wpos_ready:
            self._wpos.prune(self.max_pairs)

    # -- reading ---------------------------------------------------------
    def count(self, wh: int) -> int:
        return self._uni.get(wh)

    def words_at(self, pos: int) -> Tuple[np.ndarray, np.ndarray]:
        """Which words this corpus has seen in slot ``pos``, and how often."""
        return self._pos.slice(pos)

    def positions_of(self, wh: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Which slots this corpus has seen ``wh`` in, and how often.

        Backed by a lazily materialised reverse index. Generation only ever
        needs the forward direction, so the reverse table -- 16 bytes per
        distinct (word, slot) pair -- is not paid for until something
        actually asks the question.
        """
        if not self._wpos_ready:
            table = _GroupIndex()
            src = self._pos
            src._flush()
            g = src._g
            if g.size:
                # src is grouped by (position, word); regroup by (word, position)
                for pos in np.unique(g).tolist():
                    members, counts = src.slice(pos)
                    for mh, c in zip(members.tolist(), counts.tolist()):
                        table.add(int(mh), int(pos), int(c))
            self._wpos = table
            self._wpos_ready = True
        return self._wpos.slice(wh)

    def successors(self, wh: int, pos: int) -> Tuple[np.ndarray, np.ndarray]:
        """The positional transition: what followed this word in this slot."""
        return self._nxt.slice(_at(wh, pos))

    def followers(self, wh: int) -> Tuple[np.ndarray, np.ndarray]:
        """The position-free bigram fallback, for when a (word, slot) pair
        was never seen -- a word that only ever appears at slot 2 would
        otherwise be a dead end everywhere else."""
        return self._bg.slice(wh)

    def top_words(self, k: int = 16) -> Tuple[np.ndarray, np.ndarray]:
        out = self._uni.top(k)
        if not out:
            return _EMPTY_U64, _EMPTY_U32
        return (
            np.fromiter((h for h, _ in out), dtype=np.uint64, count=len(out)),
            np.fromiter((c for _, c in out), dtype=np.uint32, count=len(out)),
        )

    def terminators(self, k: int = 8) -> List[Tuple[Optional[str], int]]:
        out = []
        for wh, c in self._end.top(k):
            token = self.word(int(wh))
            if token is not None:
                out.append((token, c))
        return out

    def length_buckets(self) -> List[Tuple[int, int]]:
        """``[(bucket, count)]`` where bucket is a power of two -- the
        empirical line-length distribution, used when a relation sets
        ``max_words: null`` and refuses to impose a fixed cap."""
        return [(int(k), int(c)) for k, c in self._lens.top(64)]

    def sample_length(self, rng: np.random.Generator) -> int:
        """Draw a plausible line length from the learned distribution."""
        buckets = self.length_buckets()
        if not buckets:
            return 8
        keys = np.fromiter((b for b, _ in buckets), dtype=np.int64, count=len(buckets))
        cnts = np.fromiter((c for _, c in buckets), dtype=np.float64, count=len(buckets))
        p = cnts / cnts.sum()
        base = int(keys[int(rng.choice(len(keys), p=p))])
        # uniform inside the bucket, so the draw is not pinned to powers of 2
        return int(max(1, base + rng.integers(0, max(1, base))))

    def novelty(self, tokens: Sequence[str]) -> float:
        """Fraction of this token sequence's (word, slot) transitions that
        the index has never recorded. Feeds the creativity loop: a line
        built mostly from unseen pairs is a *new* construction, and that is
        what earns the reward that raises the temperature."""
        if len(tokens) < 2:
            return 1.0
        unseen = 0
        total = 0
        prev = None
        slice_ = self._nxt.slice
        for i, token in enumerate(tokens):
            wh = self.word_hash(token.lower()) if _wordish(token) else _h64(token, _NS_LINE)
            if wh is None:
                prev = None
                continue
            if prev is not None:
                total += 1
                if wh not in set(slice_(_at(prev, i - 1))[0].tolist()):
                    unseen += 1
            prev = wh
        return (unseen / total) if total else 1.0

    # -- introspection ---------------------------------------------------
    def memory_report(self) -> Dict[str, Any]:
        """
        Measured bytes held, and what the same text would have cost as
        Python objects.

        The ``naive`` column is not a strawman: it is what
        ``[line.split() for line in text.splitlines()]`` actually costs --
        8 B of list slot per token, 8 B in the outer list, and a ~49 B
        ``str`` header plus its characters for each token object. Every
        loader that keeps a corpus as token lists to "train a model" is
        paying that. This index pays 20 B per transition and ~14 B per
        vocabulary word instead.
        """
        index_bytes = (
            self._uni.nbytes + self._end.nbytes + self._lens.nbytes
            + self._pos.nbytes + self._nxt.nbytes + self._bg.nbytes
            + (self._wpos.nbytes if self._wpos_ready else 0)
        )
        self._flush_lexicon(force=True)
        lex_bytes = int(
            self._lex_h.nbytes + self._lex_off.nbytes + len(self._lex_blob)
            + 8 * len(self._punct) * 2
        )
        total = index_bytes + lex_bytes
        tokens = max(1, self.n_tokens)
        naive_per_token = 8 + 8 + 49 + 4.5
        return {
            "index_bytes": index_bytes,
            "lexicon_bytes": lex_bytes,
            "total_bytes": total,
            "source_bytes": self.n_raw_bytes,
            "tokens": self.n_tokens,
            "lines": self.n_lines,
            "docs": self.n_docs,
            "vocab": self.vocab(),
            "vocab_open": self._vocab_open,
            "pos_pairs": len(self._pos),
            "transitions": len(self._nxt),
            "bigrams": len(self._bg),
            "bytes_per_token": round(total / tokens, 3),
            "source_bytes_per_token": round(self.n_raw_bytes / tokens, 3),
            "naive_python_bytes_per_token": naive_per_token,
            "ratio_vs_naive": round(naive_per_token / max(1e-9, total / tokens), 2),
            # the store is O(distinct pairs), not O(tokens), so the win
            # grows with repetition. These two say where on that curve a
            # given corpus sits -- a 500-token sample of unique words has no
            # advantage yet, which is the honest answer, not a bug.
            "unique_pairs": len(self._pos) + len(self._nxt) + len(self._bg),
            "tokens_per_pair": round(tokens / max(1, len(self._pos) + len(self._nxt) + len(self._bg)), 3),
            "prune_passes": self.pruned,
            "words_dropped_past_vocab": self.words_dropped,
        }

    # -- plumbing --------------------------------------------------------
    def to_json(self) -> Dict[str, Any]:
        return {
            "caps": {
                "max_vocab": self.max_vocab,
                "max_pairs": self.max_pairs,
                "max_transitions": self.max_transitions,
                "max_lines": self.max_lines,
                "keep_ratio": self.keep_ratio,
            },
            "stats": {
                "n_docs": self.n_docs,
                "n_lines": self.n_lines,
                "n_tokens": self.n_tokens,
                "n_raw_bytes": self.n_raw_bytes,
                "sources": self.sources,
                "pruned": self.pruned,
                "words_dropped": self.words_dropped,
            },
            # The lexicon ships as parallel columns, not a dict: sorted
            # hashes, offsets, and one concatenated blob. A JSON object of
            # 30k words would be several megabytes of repeated keys.
            "lex_h": _pack(self._lex_h),
            "lex_off": _pack(self._lex_off),
            "lex_blob": base64.b64encode(self._lex_blob).decode("ascii"),
            "punct": {str(h): t for h, t in self._punct.items()},
            "uni": self._uni.to_json(),
            "pos": self._pos.to_json(),
            "nxt": self._nxt.to_json(),
            "bg": self._bg.to_json(),
            "end": self._end.to_json(),
            "lens": self._lens.to_json(),
        }

    @classmethod
    def from_json(cls, blob: Dict[str, Any]) -> "WordPositionIndex":
        caps = dict(blob.get("caps", {}))
        obj = cls(
            max_vocab=int(caps.get("max_vocab", 30_000)),
            max_pairs=int(caps.get("max_pairs", 250_000)),
            max_transitions=int(caps.get("max_transitions", 1_000_000)),
            max_lines=int(caps.get("max_lines", 400_000)),
            keep_ratio=float(caps.get("keep_ratio", 0.75)),
        )
        stats = dict(blob.get("stats", {}))
        obj.n_docs = int(stats.get("n_docs", 0))
        obj.n_lines = int(stats.get("n_lines", 0))
        obj.n_tokens = int(stats.get("n_tokens", 0))
        obj.n_raw_bytes = int(stats.get("n_raw_bytes", 0))
        obj.sources = list(stats.get("sources", []))
        obj.pruned = int(stats.get("pruned", 0))
        obj.words_dropped = int(stats.get("words_dropped", 0))
        obj._lex_h = _unpack(blob["lex_h"], np.uint64)
        obj._lex_off = _unpack(blob["lex_off"], np.uint32)
        obj._lex_blob = base64.b64decode(blob["lex_blob"].encode("ascii"))
        obj._punct = {int(h): t for h, t in (blob.get("punct") or {}).items()}
        obj._uni = _CountIndex.from_json(blob["uni"])
        obj._pos = _GroupIndex.from_json(blob["pos"])
        obj._nxt = _GroupIndex.from_json(blob["nxt"])
        obj._bg = _GroupIndex.from_json(blob["bg"])
        obj._end = _CountIndex.from_json(blob["end"])
        obj._lens = _CountIndex.from_json(blob["lens"])
        obj._vocab_open = int(obj._lex_h.size) < obj.max_vocab
        return obj

    def __len__(self) -> int:
        return self.vocab()

    @property
    def nbytes(self) -> int:
        """Everything this index holds, in bytes. Used to total up a robot
        that owns more than one (a background corpus, a private persona)."""
        return int(self.memory_report()["total_bytes"])

    def __repr__(self) -> str:
        return (
            f"WordPositionIndex(vocab={self.vocab()}, tokens={self.n_tokens}, "
            f"lines={self.n_lines}, transitions={len(self._nxt)})"
        )


def _split_lines(text: str) -> List[str]:
    """
    Break a document into utterance-sized lines on sentence punctuation and
    newlines. This is what gives the position counter a reset per line, which
    is the difference between "word 3 of a sentence" (useful) and "word 3 of
    a file" (noise).

    The closing punctuation stays *attached* to the line. Detaching it (the
    obvious implementation, and the one this had) means no ingested line ever
    ends in punctuation, so the line-final-token table ends up holding
    ordinary words -- and the robot then "learns" that a good place to stop
    talking is after "weather". A newline closes a line without terminating
    it, since a line break is not something anybody says out loud.
    """
    out: List[str] = []
    buf: List[str] = []
    for chunk in re.split(r"([.!?…]+|\n+)", text):
        if not chunk:
            continue
        if chunk[0] == "\n":
            line = "".join(buf).strip()
            if line:
                out.append(line)
            buf = []
            continue
        if re.fullmatch(r"[.!?…]+", chunk):
            line = ("".join(buf) + chunk).strip()
            if line:
                out.append(line)
            buf = []
            continue
        buf.append(chunk)
    line = "".join(buf).strip()
    if line:
        out.append(line)
    return [ln for ln in out if len(ln) > 1]


# ---------------------------------------------------------------------- #
# Personality
# ---------------------------------------------------------------------- #

# The six measured dimensions. These are *computed*, never asked for: point
# the persona at a character's dialogue and the numbers fall out of counts.
TRAITS = ("formality", "aggression", "valence", "verbosity", "certainty", "richness")

# Hand-written marker sets. This is the least clever part of the file and it
# is on purpose: there is no LLM, so the only way to know that a line is
# aggressive is to count aggressive words. They are deliberately small, they
# are ordinary English, and they are overridable per-persona in the JSON.
# What makes a Marvel character read as *that* character is the frequency
# of these markers in their corpus, not any knowledge about the character.
_FUNCTION_WORDS = frozenset("""
a an the of to in on at by for with from as is are was were be been being am
that this these those and or but if then than so because which who whom whose
it its he she they them his her their there here have has had do does did will
would shall should may might must can could ought
""".split())
_INFORMAL_MARKERS = frozenset("""
gonna wanna gotta kinda sorta yeah nope nah dude guys buddy kid okay ok hey
stuff things guys like really kinda alot whatever dunno gimme lemme c'mon
comeon alright yeah right well anyway actually basically literally
""".split())
_FORMAL_MARKERS = frozenset("""
therefore thus hence consequently nevertheless nonetheless accordingly
hereby whereas therein thereof notwithstanding pursuant shall hereby
commence terminate endeavor endeavour perceive ascertain regarding
""".split())
_AGGRESSIVE_MARKERS = frozenset("""
kill killed kills killing destroy destroyed destroys destroying crushing
crushed obliterate annihilate burn burned burning slaughter massacre war
battle fight fighting strike struck conquer conquered inevitable doom doomed
crush weapon weapons armies erase ended ending extinction purge buried
bury argument end ended ending
""".split())
_POSITIVE_MARKERS = frozenset("""
good great love loved lovely joy joyful hope hopeful win winning won
beautiful wonderful friend friends kindness mercy glad happy happiness
success triumph excellent amazing brilliant perfect best better fun funny
nice cool fine clear right correct smart clever strong useful helpful
welcome thanks thank please yes agree agree good interesting impressive
favourite favorite favorite easy safe calm gentle warm bright brave honest
""".split())
_NEGATIVE_MARKERS = frozenset("""
bad hate hated hatred death dead die dying died fear afraid doom doomed
loss lost lose war pain painful suffering suffer cruel misery fail failed
failure threat terrible awful horrible disgusting boring dull stupid
ugly slow weak wrong incorrect confusing useless annoying rude lazy
disappointing disappointing worst worse problem problems wrong risky
danger dangerous scary horrible awful hate dislike complaint complain
""".split())
# Negators flip the sign of a nearby opinion word. "not good" is not good,
# and without this the judge reads it as praise -- which is the single most
# common way a naive sentiment reader gets confidently, systematically wrong.
_NEGATORS = frozenset("""
not never no none nothing cannot cant can't dont don't doesnt doesn't
didnt didn't isnt isn't wasnt wasnt wont won't wouldnt wouldn't
hardly barely hardly without lack lacks lacking
""".split())
_HEDGE_MARKERS = frozenset("""
maybe perhaps might possibly probably seems seemed appear appears
somewhat rather quite fairly roughly guess suppose assume could may would
should if whether almost nearly about approximately around
""".split())
_BOOST_MARKERS = frozenset("""
definitely certainly absolutely always never inevitable unquestionably
undeniably truly completely totally forever nothing everything certain
exactly precisely must cannot no none every all only just simply
""".split())
_HUMOR_MARKERS = frozenset("""
haha hehe lol funny joke kidding kiddingly hilarious amusing amusingly
smile grinning chuckle absurd ridiculous
""".split())


# Marker rates measured on a neutral technical English corpus (186 words of
# engineering prose). A speaker whose rate equals one of these is, by
# definition, unremarkable on that axis. These are the "what does normal
# look like" numbers the trait formulas subtract; change them here if your
# corpora are a different register.
_BASE: Dict[str, float] = {
    "function": 0.4194,    # function words per word
    "informal": 0.0107,
    "formal": 0.0000,
    "aggressive": 0.0054,
    "positive": 0.0107,
    "negative": 0.0000,
    "valence": 0.0107,     # positive minus negative
    "hedge": 0.0269,
    "booster": 0.0107,
    "certainty": -0.0162,  # boosters minus hedges
    "humor": 0.0000,
}

_ANY_MARKER = frozenset().union(
    _FUNCTION_WORDS, _INFORMAL_MARKERS, _FORMAL_MARKERS, _AGGRESSIVE_MARKERS,
    _POSITIVE_MARKERS, _NEGATIVE_MARKERS, _HEDGE_MARKERS, _BOOST_MARKERS,
    _HUMOR_MARKERS,
)


def _rate(tokens: Sequence[str], markers: frozenset) -> float:
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t.lower() in markers)
    return hits / len(tokens)


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


class Persona:
    """
    A speaking personality measured out of a corpus of lines.

    Hand it a character's complete dialogue and it produces, from counts
    alone:

    * **six trait dimensions** in [0, 1] -- ``formality``, ``aggression``,
      ``valence``, ``verbosity``, ``certainty``, ``richness`` -- used to
      bias word choice and target length;
    * **signature phrases** -- the n-grams that are *over*-represented
      against a background corpus, found with a smoothed log-odds ratio, so
      a catchphrase surfaces without anybody writing it down;
    * **expressions** -- individual lines ranked by how much they stand out
      from the background, kept so the robot can quote itself in character;
    * **its own** ``WordPositionIndex``, so the persona also decides *which
      words this character is willing to say*, not just how to shape them.

    The background is optional. Without one, phrases are scored against
    corpus-internal frequency and tend to surface the character's most
    repeated line -- still useful, noticeably worse. With one (any other
    speaker's dialogue, or a plain English file) the log-odds is real.

    Nothing here is character-specific. The Marvel case is the same code
    path as a shopkeeper.
    """

    def __init__(self, name: str = "speaker", *, index: Optional[WordPositionIndex] = None) -> None:
        self.name = name
        # The persona normally *shares* the robot's index rather than owning a
        # second copy of the same corpus -- two copies doubles the resident
        # bytes for no extra information. ``own_index=True`` builds a private
        # one, which is what you want for a speaker with a deliberately small
        # vocabulary (a robot, a child, a character who never curses).
        self.owns_index = index is None
        self.index = index if index is not None else WordPositionIndex()
        self.traits: Dict[str, float] = {t: 0.5 for t in TRAITS}
        self.overrides: Dict[str, float] = {}
        self.signature: List[Tuple[str, float]] = []
        self.expressions: List[Dict[str, Any]] = []
        self.n_lines = 0
        self.mean_words = 0.0
        self.notes: Dict[str, Any] = {}
        self._lines: List[str] = []
        # this speaker's own word counts, for expression mining. A plain
        # Counter, not a second index: it is bounded by the speaker's own
        # vocabulary and it is the only thing expression mining needs.
        self._counts: Dict[str, int] = {}
        self._counts_n = 0
        self._counts_cap = 8000
        # Hashes of the marker words only. Sampling asks "does this word
        # carry a personality marker?" once per candidate per step, and
        # answering that from a 300-entry hash set avoids decoding a word
        # string for the ~95% of candidates that are in no marker set.
        self._marker_hashes = frozenset(_h64(w) for w in _ANY_MARKER)

    # -- fitting ---------------------------------------------------------
    def fit(
        self,
        lines: Sequence[str],
        *,
        background: Optional[WordPositionIndex] = None,
        max_signature: int = 12,
        max_expressions: int = 48,
        overrides: Optional[Dict[str, float]] = None,
    ) -> "Persona":
        """
        Measure ``lines``. Returns ``self``.

        ``background`` is another ``WordPositionIndex`` (e.g. a different
        speaker, or a large plain-text file) used as the null hypothesis
        for "this phrase is characteristic of *this* speaker".
        """
        lines = [ln.strip() for ln in lines if ln and ln.strip()]
        if not lines:
            return self
        self.n_lines = len(lines)
        # kept for signature mining, which runs before the expression
        # ranking exists -- without this the two-step mining had nothing to
        # read on the first pass
        self._lines = list(lines)

        # this speaker's own unigram counts, taken from the lines rather than
        # from a second index: expression mining must contrast the character
        # against the background, and the character half of that contrast is
        # exactly this
        counts: Dict[str, int] = {}
        tokens: List[str] = []
        for line in lines:
            words = [w.lower() for w in _words_of(_tokenize(line))]
            tokens.extend(words)
            for w in words:
                counts[w] = counts.get(w, 0) + 1
        if self.owns_index:
            self.index = WordPositionIndex(
                max_vocab=self.index.max_vocab,
                max_pairs=self.index.max_pairs,
                max_transitions=self.index.max_transitions,
            )
        for i, line in enumerate(lines):
            self.index.add_line(line, weight=1)
            if i % 512 == 0 and self.owns_index:
                self.index._prune_to_fit()
        if self.owns_index:
            self.index.close()
        if len(counts) > self._counts_cap:
            counts = dict(sorted(counts.items(), key=lambda kv: -kv[1])[: self._counts_cap])
        self._counts = counts
        self._counts_n = sum(counts.values())

        n = max(1, len(tokens))

        func = _rate(tokens, _FUNCTION_WORDS)
        informal = _rate(tokens, _INFORMAL_MARKERS)
        formal = _rate(tokens, _FORMAL_MARKERS)
        aggr = _rate(tokens, _AGGRESSIVE_MARKERS)
        pos = _rate(tokens, _POSITIVE_MARKERS)
        neg = _rate(tokens, _NEGATIVE_MARKERS)
        hedge = _rate(tokens, _HEDGE_MARKERS)
        boost = _rate(tokens, _BOOST_MARKERS)
        humor = _rate(tokens, _HUMOR_MARKERS)

        self.mean_words = sum(
            len(_words_of(_tokenize(ln))) for ln in lines
        ) / max(1, len(lines))
        ttr = len(set(tokens)) / n

        # Each trait is centred on the rate measured in ordinary English
        # prose and scaled so that a strong signal lands near 0 or 1. The
        # baselines below were measured on a neutral technical corpus and are
        # module constants rather than magic numbers buried in a formula, so
        # recalibrating for a different corpus or a different marker set is
        # a one-line change instead of an archaeology exercise.
        self.traits = {
            "formality": _clip01(0.5 + 3.5 * (func - _BASE["function"])
                                 - 1.5 * informal + 0.8 * formal),
            "aggression": _clip01(0.5 + 25.0 * (aggr - _BASE["aggressive"])
                                  + 1.2 * neg),
            "valence": _clip01(0.5 + 14.0 * ((pos - neg) - _BASE["valence"])),
            "verbosity": _clip01(self.mean_words / 40.0),
            "certainty": _clip01(0.5 + 14.0 * ((boost - hedge) - _BASE["certainty"])),
            "richness": _clip01(0.05 + 1.3 * ttr + 3.0 * humor),
        }
        for key, value in (overrides or {}).items():
            if key in self.traits:
                self.overrides[key] = _clip01(value)
        self._apply_overrides()

        self.signature = self._find_signature(background, max_signature)
        self.expressions = self._find_expressions(lines, background, max_expressions)
        self.notes = {
            "tokens": n,
            "type_token_ratio": round(ttr, 4),
            "function_rate": round(func, 4),
            "informal_rate": round(informal, 5),
            "aggressive_rate": round(aggr, 5),
            "positive_rate": round(pos, 5),
            "negative_rate": round(neg, 5),
            "hedge_rate": round(hedge, 5),
            "booster_rate": round(boost, 5),
            "humor_rate": round(humor, 5),
        }
        return self

    def _apply_overrides(self) -> None:
        for key, value in self.overrides.items():
            self.traits[key] = _clip01(value)

    def _find_signature(
        self, background: Optional[WordPositionIndex], limit: int
    ) -> List[Tuple[str, float]]:
        """
        Over-represented word bigrams, by smoothed log-odds.

        score = log((f_p + a) / (n_p + 2a)) - log((f_b + a) / (n_b + 2a))

        With no background the second term collapses to 1, so the ranking
        degenerates to raw frequency -- the character's most repeated
        phrase. That is a real, documented degradation, not a silent one.
        """
        bigrams: Dict[Tuple[str, str], int] = {}
        for line in self._all_lines():
            toks = [w.lower() for w in _words_of(_tokenize(line))]
            for i in range(len(toks) - 1):
                pair = (toks[i], toks[i + 1])
                bigrams[pair] = bigrams.get(pair, 0) + 1
        if not bigrams:
            return []
        n_p = max(1, sum(bigrams.values()))
        n_b = n_p
        b_freq: Dict[Tuple[str, str], int] = {}
        if background is not None:
            b_freq = _bigram_counts(background)
            n_b = max(1, sum(b_freq.values()))
        a = 0.5
        scored: List[Tuple[str, float]] = []
        for pair, f in bigrams.items():
            if f < 2:
                continue
            fb = b_freq.get(pair, 0)
            score = math.log((f + a) / (n_p + 2 * a)) - math.log((fb + a) / (n_b + 2 * a))
            # longer phrases are more distinctive but rarer; nudge so a
            # two-word tic can outrank a grammatical filler bigram
            scored.append((pair[0] + " " + pair[1], score + 0.35 * math.log1p(f)))
        scored.sort(key=lambda kv: -kv[1])
        return [(text, round(score, 5)) for text, score in scored[:limit]]

    def _find_expressions(
        self, lines: Sequence[str], background: Optional[WordPositionIndex], limit: int
    ) -> List[Dict[str, Any]]:
        """Lines that say the most about this speaker: mean per-content-word
        log-odds against the background, with a length guard so a one-word
        line cannot win by having no rare words in it."""
        b_freq: Dict[str, int] = {}
        b_n = 1
        if background is not None:
            b_freq, b_n = _unigram_counts(background)
        ranked: List[Tuple[float, str]] = []
        for line in lines:
            words = [w.lower() for w in _words_of(_tokenize(line))]
            content = [w for w in words if w not in _FUNCTION_WORDS]
            if not content:
                continue
            total = 0.0
            for w in set(content):
                fp = self._counts.get(w, 0)
                total += math.log((fp + 0.5) / (self.n_lines + 1.0)) - \
                    math.log((b_freq.get(w, 0) + 0.5) / (b_n + 1.0))
            score = total / len(set(content)) * min(1.0, len(content) / 3.0)
            ranked.append((score, line))
        ranked.sort(key=lambda kv: -kv[0])
        return [
            {"text": text, "score": round(score, 5), "traits": self._line_traits(text)}
            for score, text in ranked[:limit]
        ]

    def _all_lines(self) -> Iterable[str]:
        """The lines this persona was fitted on."""
        if getattr(self, "_lines", None):
            yield from self._lines
        else:
            for expr in self.expressions:
                yield expr["text"]

    def _line_traits(self, text: str) -> Dict[str, float]:
        """A short, per-line read-out on three axes, for tagging stored
        expressions. Centred on the same baselines as ``fit()`` so a line's
        tags and the persona's traits are on the same scale."""
        toks = [w.lower() for w in _words_of(_tokenize(text))]
        return {
            "aggression": _clip01(0.5 + 25.0 * (
                _rate(toks, _AGGRESSIVE_MARKERS) - _BASE["aggressive"])),
            "valence": _clip01(0.5 + 14.0 * (
                (_rate(toks, _POSITIVE_MARKERS) - _rate(toks, _NEGATIVE_MARKERS))
                - _BASE["valence"])),
            "formality": _clip01(0.5 + 3.5 * (
                _rate(toks, _FUNCTION_WORDS) - _BASE["function"])),
        }

    # -- use -------------------------------------------------------------
    def word_bias(self, word: str) -> float:
        """
        A multiplier in roughly [0.35, 2.6] for picking this word, derived
        from the measured traits. This is where "personality" actually
        enters generation: not a style guide applied afterwards, but the
        sampling distribution being reweighted per candidate word.

        Most words in any corpus are in no marker set at all, so that case
        returns immediately -- this is called once per candidate word per
        step of every line, and the early exit is what makes it affordable.
        """
        w = word.lower()
        if w not in _ANY_MARKER:
            return 1.0
        bias = 1.0
        bias += 0.55 * (self.traits["aggression"] - 0.5) * 2.0 * (1.0 if w in _AGGRESSIVE_MARKERS else 0.0)
        bias += 0.45 * (self.traits["valence"] - 0.5) * 2.0 * (
            (1.0 if w in _POSITIVE_MARKERS else 0.0) - (1.0 if w in _NEGATIVE_MARKERS else 0.0)
        )
        bias += 0.40 * (0.5 - self.traits["formality"]) * 2.0 * (1.0 if w in _INFORMAL_MARKERS else 0.0)
        bias += 0.40 * (self.traits["formality"] - 0.5) * 2.0 * (1.0 if w in _FORMAL_MARKERS else 0.0)
        bias += 0.35 * (self.traits["certainty"] - 0.5) * 2.0 * (
            (1.0 if w in _BOOST_MARKERS else 0.0) - (0.7 if w in _HEDGE_MARKERS else 0.0)
        )
        bias += 0.30 * (self.traits["richness"] - 0.5) * 2.0 * (1.0 if w in _HUMOR_MARKERS else 0.0)
        # a function word in a long, fancy line is usually filler
        if w in _FUNCTION_WORDS and self.traits["richness"] > 0.6:
            bias *= 0.75
        return float(min(2.6, max(0.35, bias)))

    def bias_for_hash(self, wh: int, index: "WordPositionIndex") -> float:
        """
        ``word_bias`` for a word the caller only has a hash for. Returns 1.0
        without touching the lexicon unless the hash is a known marker.
        """
        if int(wh) not in self._marker_hashes:
            return 1.0
        word = index.word(int(wh))
        return self.word_bias(word) if word else 1.0

    def target_length_scale(self) -> float:
        """How much this speaker stretches or clips a requested length."""
        return float(0.65 + 0.9 * self.traits["verbosity"])

    def tic_chance(self) -> float:
        """How likely a signature phrase is to be worked into a line."""
        return float(_clip01(0.05 + 0.55 * self.traits["certainty"] + 0.25 * self.traits["aggression"]))

    def describe(self) -> str:
        rows = " ".join(
            f"{k}={self.traits[k]:.2f}" + ("*" if k in self.overrides else "")
            for k in TRAITS
        )
        tic = self.signature[0][0] if self.signature else "-"
        return (
            f"<Persona {self.name!r} lines={self.n_lines} "
            f"mean_words={self.mean_words:.1f} {rows} tic={tic!r}>"
        )

    # -- plumbing --------------------------------------------------------
    def to_json(self, *, include_index: bool = True) -> Dict[str, Any]:
        blob = {
            "name": self.name,
            "owns_index": self.owns_index,
            "counts": self._counts,
            "traits": {k: round(v, 5) for k, v in self.traits.items()},
            "overrides": self.overrides,
            "signature": self.signature,
            "expressions": self.expressions,
            "n_lines": self.n_lines,
            "mean_words": round(self.mean_words, 4),
            "notes": self.notes,
            # the lines themselves, so a reload can re-fit without the
            # source .txt files. Capped, and not the index.
            "lines": [ln[:200] for ln in self._lines[:512]],
        }
        # only a *private* index is worth saving; a shared one belongs to the
        # robot and is serialised with it
        if include_index and self.owns_index:
            blob["index"] = self.index.to_json()
        return blob

    @classmethod
    def from_json(cls, blob: Dict[str, Any]) -> "Persona":
        obj = cls(
            name=str(blob.get("name", "speaker")),
            index=WordPositionIndex.from_json(blob["index"]) if blob.get("index") else None,
        )
        obj.owns_index = bool(blob.get("owns_index", blob.get("index") is not None))
        obj._counts = {str(k): int(v) for k, v in (blob.get("counts") or {}).items()}
        obj._counts_n = sum(obj._counts.values())
        obj.traits = {k: float(v) for k, v in (blob.get("traits") or {}).items() if k in TRAITS}
        for key in TRAITS:
            obj.traits.setdefault(key, 0.5)
        obj.overrides = {k: float(v) for k, v in (blob.get("overrides") or {}).items()}
        obj.signature = [(str(t), float(s)) for t, s in (blob.get("signature") or [])]
        obj.expressions = list(blob.get("expressions") or [])
        obj.n_lines = int(blob.get("n_lines", 0))
        obj.mean_words = float(blob.get("mean_words", 0.0))
        obj.notes = dict(blob.get("notes") or {})
        obj._lines = [str(x) for x in (blob.get("lines") or [])]
        return obj

    def __repr__(self) -> str:
        return self.describe()


def _unigram_counts(index: WordPositionIndex) -> Tuple[Dict[str, int], int]:
    counts: Dict[str, int] = {}
    total = 0
    for wh, c in index._uni.items():
        word = index.word(int(wh))
        if word is None:
            continue
        counts[word] = counts.get(word, 0) + int(c)
        total += int(c)
    return counts, max(1, total)


def _bigram_counts(index: WordPositionIndex) -> Dict[Tuple[str, str], int]:
    counts: Dict[Tuple[str, str], int] = {}
    for wh, members, cs in _iter_groups(index._bg):
        prev = index.word(int(wh))
        if prev is None:
            continue
        for mh, c in zip(members, cs):
            nxt = index.word(int(mh))
            if nxt is None:
                continue
            pair = (prev.lower(), nxt.lower())
            counts[pair] = counts.get(pair, 0) + int(c)
    return counts


def _iter_groups(table: _GroupIndex, limit: int = 20_000) -> Iterable[Tuple[int, np.ndarray, np.ndarray]]:
    """Walk a ``_GroupIndex``'s distinct groups. Used only for offline
    analysis (signature/expression mining), never on the speak path, so a
    bounded walk is fine and keeps the call out of the hot loop."""
    table._flush()
    g = table._g
    if g.size == 0:
        return
    boundaries = np.flatnonzero(np.concatenate(([True], g[1:] != g[:-1])))
    if boundaries.size > limit:
        boundaries = boundaries[:limit]
    for i in boundaries:
        lo = int(i)
        hi = lo + 1
        while hi < g.size and g[hi] == g[lo]:
            hi += 1
        yield int(g[lo]), table._m[lo:hi], table._c[lo:hi]


# ---------------------------------------------------------------------- #
# Memory -- the same hashed store, used for remembering
# ---------------------------------------------------------------------- #

# The claim being tested here is that recall does not need a second data
# structure. It does not: MemoryLedger indexes the *situation* text in a
# WordPositionIndex and keeps a bounded ring of small records. Retrieval is
# hashed word overlap + recency decay + how well it went last time. Same
# machinery as the language model, same caps, same memory report.


class MemoryRecord:
    __slots__ = ("seq", "state", "said", "relation", "reward", "novelty", "state_h", "words")

    def __init__(
        self,
        seq: int,
        state: str,
        said: str,
        relation: str,
        reward: float,
        novelty: float,
        state_h: int,
        words: Sequence[str],
    ) -> None:
        self.seq = seq
        self.state = state
        self.said = said
        self.relation = relation
        self.reward = float(reward)
        self.novelty = float(novelty)
        self.state_h = state_h
        self.words = list(words)

    def to_json(self) -> Dict[str, Any]:
        return {
            "seq": self.seq, "state": self.state, "said": self.said,
            "relation": self.relation, "reward": round(self.reward, 5),
            "novelty": round(self.novelty, 5), "state_h": str(self.state_h),
            "words": self.words,
        }

    @classmethod
    def from_json(cls, blob: Dict[str, Any]) -> "MemoryRecord":
        return cls(
            int(blob["seq"]), str(blob["state"]), str(blob["said"]),
            str(blob.get("relation", "")), float(blob.get("reward", 0.0)),
            float(blob.get("novelty", 0.0)), int(blob.get("state_h", 0)),
            list(blob.get("words") or []),
        )

    def __repr__(self) -> str:
        return f"<Memory #{self.seq} r={self.reward:+.2f} {self.said[:40]!r}>"


class MemoryLedger:
    """
    Episodic memory built on the same hashed store as the language model.

    ``record`` is called after every judged exchange. ``recall(query)``
    returns what the robot said in the most similar situations before, best
    first, with recency decay (a half-life in events) and the outcome of
    that exchange folded into the ranking -- so a line that was punished is
    findable but ranks below a line that was rewarded.

    Bounded: ``capacity`` records, then the oldest half is dropped.
    """

    def __init__(self, *, capacity: int = 512, half_life: float = 120.0,
                 record_words: int = 24, max_vocab: int = 8_000) -> None:
        self.capacity = int(capacity)
        self.half_life = float(half_life)
        self.record_words = int(record_words)
        self.index = WordPositionIndex(max_vocab=max_vocab, max_pairs=40_000,
                                       max_transitions=40_000)
        self._records: deque = deque(maxlen=self.capacity)
        self._seq = 0
        self.n_recorded = 0
        self.n_evicted = 0

    def record(
        self, state: str, said: str, *, relation: str = "", reward: float = 0.0,
        novelty: float = 0.0,
    ) -> MemoryRecord:
        words = [w.lower() for w in _words_of(_tokenize(state))][: self.record_words]
        rec = MemoryRecord(
            seq=self._seq, state=state[:400], said=said[:400], relation=relation,
            reward=reward, novelty=novelty, state_h=_state_h(state.lower()),
            words=words,
        )
        self._seq += 1
        if len(self._records) == self.capacity:
            self.n_evicted += self.capacity // 2 + 1
            keep = self.capacity // 2
            for _ in range(keep):
                self._records.popleft()
            self.index._prune_to_fit()
        self._records.append(rec)
        for w in words:
            self.index.word_hash(w)
        self.index.add_line(" ".join(words))
        self.n_recorded += 1
        return rec

    def recall(self, query: str, k: int = 5, *, min_overlap: float = 0.12) -> List[Dict[str, Any]]:
        """
        What did I say when this came up before? Returns dicts with
        ``text``, ``state``, ``relation``, ``reward``, ``overlap``,
        ``recency`` and ``score``, best first.
        """
        if not self._records:
            return []
        qwords = {w.lower() for w in _words_of(_tokenize(query))}
        if not qwords:
            qwords = {query.strip().lower()} if query else set()
        newest = max((r.seq for r in self._records), default=0)
        out: List[Dict[str, Any]] = []
        for rec in self._records:
            overlap = 0.0
            if qwords:
                common = qwords.intersection(rec.words)
                union = qwords.union(rec.words) or qwords
                overlap = len(common) / len(union)
            if overlap < min_overlap:
                continue
            age = max(0, newest - rec.seq)
            recency = 0.5 ** (age / max(1.0, self.half_life))
            score = 0.62 * overlap + 0.23 * recency + 0.15 * (0.5 + 0.5 * rec.reward)
            out.append({
                "text": rec.said, "state": rec.state, "relation": rec.relation,
                "reward": round(rec.reward, 4), "novelty": round(rec.novelty, 4),
                "overlap": round(overlap, 4), "recency": round(recency, 4),
                "score": round(score, 5), "seq": rec.seq,
            })
        out.sort(key=lambda d: -d["score"])
        return out[: int(k)]

    def recent(self, k: int = 5) -> List[Dict[str, Any]]:
        return [r.to_json() for r in list(self._records)[-int(k):]]

    def clear(self) -> None:
        self._records.clear()
        self.index = WordPositionIndex(max_vocab=self.index.max_vocab)

    def to_json(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "half_life": self.half_life,
            "record_words": self.record_words,
            "seq": self._seq,
            "n_recorded": self.n_recorded,
            "n_evicted": self.n_evicted,
            "records": [r.to_json() for r in self._records],
        }

    @classmethod
    def from_json(cls, blob: Dict[str, Any]) -> "MemoryLedger":
        obj = cls(
            capacity=int(blob.get("capacity", 512)),
            half_life=float(blob.get("half_life", 120.0)),
            record_words=int(blob.get("record_words", 24)),
        )
        obj._seq = int(blob.get("seq", 0))
        obj.n_recorded = int(blob.get("n_recorded", 0))
        obj.n_evicted = int(blob.get("n_evicted", 0))
        for raw in blob.get("records") or []:
            rec = MemoryRecord.from_json(raw)
            obj._records.append(rec)
            for w in rec.words:
                obj.index.word_hash(w)
        return obj

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"<MemoryLedger {len(self._records)}/{self.capacity} recorded={self.n_recorded}>"


# ---------------------------------------------------------------------- #
# Planning a sentence
# ---------------------------------------------------------------------- #

# Words considered when the transition table has nothing to offer. This is
# the only place a hard number appears in generation, and it is a floor on
# the candidate set, not a cap on the output.
_FLOOR_CANDIDATES = 12
_WALK_HARD_LIMIT = 400
# per-plan hash->word cache. Lives for one plan() call, so it is a few dozen
# entries and holds references to strings the packed blob already owns --
# it costs lookups, not bytes.
_WORD_CACHE_MAX = 512
# a (word, slot) cell seen at least this many times is trusted on its own
_POS_TRUST = 3
# a line at least this novel has to be, before its words become seed material
_PROMOTE_NOVELTY = 0.25


class Plan:
    """What the planner decided, and why. Purely diagnostic -- everything
    here is also on ``Utterance.parts`` -- but it is the object to read when
    a line comes out wrong and you want to know which of the three decisions
    (length, seed, walk) misfired."""

    __slots__ = ("text", "tokens", "target_words", "candidates", "relation",
                 "seed", "steps", "entropy", "novelty")

    def __init__(self) -> None:
        self.text = ""
        self.tokens: List[str] = []
        self.target_words = 0
        self.candidates: List[str] = []
        self.relation = ""
        self.seed = ""
        self.steps: List[str] = []
        self.entropy = 0.0
        self.novelty = 1.0

    def to_json(self) -> Dict[str, Any]:
        return {
            "text": self.text, "target_words": self.target_words,
            "relation": self.relation, "seed": self.seed, "steps": self.steps,
            "entropy": round(self.entropy, 5),
            "novelty": round(self.novelty, 5),
            "candidates": self.candidates,
        }

    def __repr__(self) -> str:
        return f"<Plan {self.target_words}w rel={self.relation!r} {self.text[:48]!r}>"


class SpeechPlanner:
    """
    Decides three things, in this order, and returns finished lines.

    1. **How long.** From the relation's own ``output.length``: a truncated
       normal over ``mean_words`` scaled by ``spread``, clipped into
       ``[min_words, max_words]`` and stretched by the persona's verbosity.
       ``max_words: null`` removes the ceiling entirely and the length comes
       from the *learned* line-length distribution in the index instead --
       so the response space is bounded by the data, never by a constant in
       this file.
    2. **Which words to start from.** A weighted pool assembled from the
       relation's required and preferred words, the last content word of
       the input (so a reply acknowledges what it was asked), any seeds
       promoted by earlier good feedback, the persona's signature openers,
       and finally the corpus's frequent words.
    3. **How to walk.** At each slot the positional transition table is
       sampled with temperature ``0.30 + 0.90 * creativity`` and every
       candidate word reweighted by the persona's ``word_bias``, falling
       back bigram -> slot profile -> global frequent when the table runs
       dry. High creativity means a flatter distribution, occasional
       mid-sentence re-seeding, and a higher chance of working a signature
       phrase in.

    Several candidate lines are built per call (``width``) and handed to the
    foundation, which picks one. The planner proposes; the EmptyMicroRobot
    core disposes.
    """

    def __init__(self, index: WordPositionIndex, persona: Optional[Persona] = None) -> None:
        self.index = index
        self.persona = persona if persona is not None else Persona()
        self.n_planned = 0
        self.n_forced = 0
        self._last_step = "length=?"
        self._last_walk: Tuple[str, List[str]] = ("none", [])
        self._word_cache: Dict[int, Optional[str]] = {}

    # -- decision 1: length ---------------------------------------------
    def plan_length(
        self, spec: Dict[str, Any], rng: np.random.Generator, *,
        energy: float = 1.0,
    ) -> int:
        lo = spec.get("min_words")
        hi = spec.get("max_words")
        mean = spec.get("mean_words")
        spread = float(spec.get("spread", 0.35) or 0.35)

        if hi is None:
            # No ceiling was declared: take the length from the corpus's own
            # line-length distribution. This is the "no fixed limitation on
            # responses" path -- the cap is whatever this data does, not a
            # constant.
            base = self.index.sample_length(rng)
            step = "length=sampled_from_corpus"
        else:
            hi = int(hi)
            lo = int(lo) if lo is not None else 1
            if mean is None:
                mean = (lo + hi) / 2.0
            mean = float(mean)
            sigma = max(0.35, spread * (hi - lo) / 2.0)
            base = int(round(rng.normal(mean, sigma)))
            base = max(lo, min(hi, base))
            step = f"length=trunc_normal({mean:.1f}+-{sigma:.1f})->{base}"

        scale = self.persona.target_length_scale() * max(0.25, float(energy))
        target = int(round(base * scale))
        if lo is not None:
            target = max(int(lo), target)
        if hi is not None:
            target = min(int(hi), target)
        target = max(1, min(_WALK_HARD_LIMIT, target))
        self._last_step = step
        return target

    # -- decision 2: seeds ----------------------------------------------
    def seed_pool(
        self, relation: Dict[str, Any], input_tokens: Sequence[str],
    ) -> List[Tuple[str, float]]:
        """
        ``[(word, weight)]``. Weight 0 entries are dropped. Everything is
        filtered through the lexicon, so a seed that the corpus has never
        seen cannot be spoken and does not quietly waste a slot.
        """
        spec = relation.get("output", {}) or {}
        pool: Dict[str, float] = {}

        def put(word: Any, weight: float) -> None:
            if not isinstance(word, str):
                return
            w = word.strip().lower()
            if not w or not self.index.known(w):
                return
            pool[w] = max(pool.get(w, 0.0), float(weight))

        for w in spec.get("must_include", []) or []:
            put(w, 3.0)
        for w in spec.get("preferred", []) or []:
            put(w, 1.6)
        for w in relation.get("opening", []) or []:
            put(w, 1.4)
        for w, weight in (spec.get("learned") or {}).items():
            put(w, min(4.0, 0.8 + float(weight)))

        # echo the topic: the last content word the partner used
        heard = [t.lower() for t in _words_of(list(input_tokens))
                 if self.index.known(t.lower())]
        for w in heard[-2:]:
            put(w, 1.1)
        for w in heard[:1]:
            put(w, 0.8)

        for phrase, score in self.persona.signature[:3]:
            first = phrase.split(" ", 1)[0]
            put(first, 0.5 + 0.2 * max(0.0, score))

        if not pool:
            hashes, counts = self.index.top_words(_FLOOR_CANDIDATES)
            for wh, c in zip(hashes.tolist(), counts.tolist()):
                word = self.index.word(int(wh))
                if word:
                    pool[word] = float(c)
        return list(pool.items())

    # -- decision 3: the walk -------------------------------------------
    def _pick(
        self, hashes: np.ndarray, counts: np.ndarray, rng: np.random.Generator,
        temperature: float, *, cap: int = _FLOOR_CANDIDATES,
    ) -> Tuple[Optional[int], float]:
        """
        Sample one successor. Counts become log-probabilities divided by the
        temperature (so a *high* temperature flattens the distribution and
        the robot picks the unlikely word), then every candidate is
        reweighted by the persona's bias for that word. Returns the chosen
        word hash and the entropy of the distribution it was drawn from.

        Written as plain Python arithmetic on a capped candidate list rather
        than as numpy: with a cap of 12 the array construction and three
        ``np.fromiter`` calls cost more than the sampling, and this runs once
        per word of every candidate line on every call.
        """
        if hashes.size == 0:
            return None, 0.0
        if hashes.size > cap:
            keep = np.argpartition(-counts.astype(np.int64), cap - 1)[:cap]
            hashes, counts = hashes[keep], counts[keep]
        raw_h = hashes.tolist()
        cache = self._word_cache
        pool: List[Tuple[int, str, float]] = []
        for h, c in zip(raw_h, counts.tolist()):
            word = cache.get(h)
            if word is None:
                word = self.index.word(int(h))
                if len(cache) < _WORD_CACHE_MAX:
                    cache[int(h)] = word
            if word:
                pool.append((int(h), word, max(1.0, float(c))))
        if not pool:
            return None, 0.0

        inv = 1.0 / max(0.05, temperature)
        top = max(math.log(p[2]) for p in pool)
        scores: List[float] = []
        for h, word, c in pool:
            scores.append((math.log(c) - top) * inv
                          + math.log(self.persona.bias_for_hash(h, self.index)))
        apex = max(scores)
        weights = [math.exp(s - apex) for s in scores]
        total = float(sum(weights))
        if total <= 0.0:
            return None, 0.0
        target = rng.random() * total
        acc = 0.0
        chosen = len(weights) - 1
        for i, w in enumerate(weights):
            acc += w
            if target <= acc:
                chosen = i
                break
        inv_total = 1.0 / total
        entropy = 0.0
        for w in weights:
            p = w * inv_total
            entropy -= p * math.log(p if p > 1e-12 else 1e-12)
        return pool[chosen][0], entropy

    def _next_candidates(
        self, cur: int, pos: int,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        The candidate set for slot ``pos + 1``, and where it came from.

        Positional and position-free evidence are *merged* rather than
        tried in order. A first-order positional cell that was seen exactly
        once in a small corpus will happily send the walk somewhere absurd,
        while the plain bigram for the same word may have fifty
        observations behind it; taking whichever table is non-empty throws
        that away. So the positional count is added to a discounted bigram
        count, and the two agree on words seen in both.

        Falls back to the slot profile and then to the global frequent
        words, so a word that appears in only one slot is still emittable
        somewhere.
        """
        pm, pc = self.index.successors(cur, pos)
        # A positional cell with real support behind it is the most specific
        # evidence available, and consulting the bigram as well only adds
        # two binary searches and a union. Below the trust floor the cell is
        # probably a single observation, and then the bigram genuinely helps.
        if pm.size and pc.size and max(pc.tolist()) >= _POS_TRUST:
            return pm, pc, "pos"
        bm, bc = self.index.followers(cur)
        if pm.size or bm.size:
            if pm.size and bm.size:
                both = np.union1d(pm, bm)
                pcount = np.zeros(both.size, dtype=np.int64)
                bcount = np.zeros(both.size, dtype=np.int64)
                pcount[np.searchsorted(both, pm)] = pc.astype(np.int64)
                bcount[np.searchsorted(both, bm)] = bc.astype(np.int64)
                # the positional cell is the more specific evidence, so it
                # gets the larger share of the vote
                total = pcount + np.maximum(1, bcount)
                return both, total.astype(np.uint32), "pos+bigram"
            if pm.size:
                return pm, pc, "pos"
            return bm, bc, "bigram"
        pm, pc = self.index.words_at(pos + 1)
        if pm.size:
            return pm, pc, "slot"
        pm, pc = self.index.top_words(_FLOOR_CANDIDATES)
        return pm, pc, "global"

    def walk(
        self, seed: str, target: int, rng: np.random.Generator, *,
        temperature: float, ricochet: float = 0.0,
        pool: Optional[Sequence[Tuple[str, float]]] = None,
    ) -> Tuple[List[str], float]:
        """
        Walk the positional transition table from ``seed`` for ``target``
        word tokens. Returns the token list and the mean per-step entropy of
        the distribution it sampled from (a direct read-out of how creative
        this line was allowed to be).
        """
        out: List[str] = []
        cur = self.index.word_hash(seed)
        if cur is None:
            hashes, _ = self.index.top_words(1)
            cur = int(hashes[0]) if hashes.size else None
        if cur is None:
            return out, 0.0
        entropies: List[float] = []
        steps: List[str] = []
        src = "none"
        for i in range(min(target, _WALK_HARD_LIMIT)):
            word = self.index.word(cur)
            if word is None:
                break
            out.append(word)
            if not _wordish(word):
                break  # punctuation ends the line
            if ricochet > 0.0 and pool and rng.random() < ricochet:
                options = [w for w, _ in pool if w != word]
                if options:
                    out.pop()
                    out.append(options[int(rng.integers(len(options)))])
                    cur = self.index.word_hash(out[-1]) or cur
                    steps.append(f"ricochet@{i}")
            hashes, counts, src = self._next_candidates(cur, i)
            self._last_walk = (src, steps)
            nxt, entropy = self._pick(hashes, counts, rng, temperature)
            if nxt is None:
                steps.append(f"dead@{i}")
                break
            entropies.append(entropy)
            cur = nxt
        if not any((not _wordish(t)) for t in out):
            out.append(self._terminator(rng))
        return out, (float(np.mean(entropies)) if entropies else 0.0)

    def _terminator(self, rng: np.random.Generator) -> str:
        options = [(t, c) for t, c in self.index.terminators(8) if t]
        if not options:
            return "."
        total = sum(c for _, c in options)
        pick = rng.random() * max(1, total)
        acc = 0
        for token, c in options:
            acc += c
            if pick <= acc:
                return token
        return options[0][0]

    # -- assembly --------------------------------------------------------
    def _enforce(
        self, tokens: List[str], spec: Dict[str, Any], rng: np.random.Generator,
    ) -> List[str]:
        """
        Make the line satisfy the relation's contract, which the walk has no
        way to know about: required words present, no leading punctuation, a
        terminal mark.
        """
        out = [t for t in tokens]
        while out and not _wordish(out[0]):
            out.pop(0)
        if not out:
            hashes, _ = self.index.top_words(1)
            fallback = [self.index.word(int(h)) for h in hashes.tolist()]
            out = [w for w in fallback if w] or ["..."]

        lowered = {t.lower() for t in out}
        for required in spec.get("must_include", []) or []:
            if not isinstance(required, str):
                continue
            want = required.lower()
            if want in lowered:
                continue
            self.n_forced += 1
            # where a character volunteers a keyword depends on the
            # character: aggressive voices front-load, verbose ones append
            if self.persona.traits["aggression"] > 0.6:
                at = 0
            elif self.persona.traits["verbosity"] > 0.55:
                at = max(0, len(out) - 1)
            else:
                at = int(rng.integers(0, max(1, min(len(out), 4))))
            out.insert(min(at, len(out)), want)
            lowered.add(want)

        if not any((not _wordish(t)) for t in out):
            out.append(self._terminator(rng))
        return out

    def plausible(self, text: str) -> bool:
        """
        The cheapest possible quality gate on a walked line.

        A first-order positional Markov walk over real text is allowed to
        produce "good good", "the the", or a three-word fragment, and those
        must never reach the caller. This checks exactly three things: there
        are enough words, enough *distinct* words, and no immediately
        repeated word. It is a filter, not a grammar -- a line that passes
        can still be nonsense, and the feedback judge is what catches that.
        """
        words = [w.lower() for w in _words_of(_tokenize(text))]
        if len(words) < 3:
            return False
        if len(set(words)) < max(3, int(0.55 * len(words))):
            return False
        for a, b in zip(words, words[1:]):
            if a == b:
                return False
        bigrams = list(zip(words, words[1:]))
        if len(bigrams) >= 3 and len(set(bigrams)) < 0.7 * len(bigrams):
            return False
        return True

    def plan(
        self,
        relation: Dict[str, Any],
        input_tokens: Sequence[str],
        rng: np.random.Generator,
        *,
        creativity: float = 0.2,
        energy: float = 1.0,
        width: int = 5,
        tic_chance: Optional[float] = None,
    ) -> Plan:
        """
        Build finished lines and return them all. The caller (or the
        foundation, via ``decide(options=...)``) picks the one it likes.

        Candidates that fail ``plausible()`` are dropped; if that would
        leave fewer than two to choose between, the unfiltered set is kept
        anyway -- reporting "nothing to say" because a very small corpus
        cannot produce grammatical noise would be worse than saying the
        noise.
        """
        self._word_cache = {}
        spec = dict(relation.get("output", {}) or {})
        plan = Plan()
        plan.relation = str(relation.get("id", ""))

        target = self.plan_length(spec, rng, energy=energy)
        plan.target_words = target
        plan.steps.append(getattr(self, "_last_step", "length=?"))

        pool = self.seed_pool(relation, input_tokens)
        if not pool:
            plan.text = ""
            plan.steps.append("no-seed-available")
            return plan
        names = [w for w, _ in pool]
        raw_w = [w for _, w in pool]
        scale = float(sum(raw_w)) or 1.0
        weights = [w / scale for w in raw_w]

        def draw_seed() -> str:
            target = rng.random()
            acc = 0.0
            for i, w in enumerate(weights):
                acc += w
                if target <= acc:
                    return names[i]
            return names[-1]

        temperature = 0.30 + 0.90 * float(np.clip(creativity, 0.0, 1.0))
        ricochet = 0.04 * float(np.clip(creativity, 0.0, 1.0))
        chance = self.persona.tic_chance() if tic_chance is None else float(tic_chance)

        filtered: List[str] = []
        raw: List[str] = []
        tics = 0
        seed = names[0]
        for _ in range(max(1, int(width))):
            seed = draw_seed()
            tokens, entropy = self.walk(
                seed, target, rng, temperature=temperature,
                ricochet=ricochet, pool=pool,
            )
            tic_words: List[str] = []
            if chance > 0.0 and self.persona.signature and rng.random() < chance:
                tokens, tic_words = self._insert_tic(tokens, rng)
                tokens = self._fit_length(tokens, spec, tic_words)
                tics += 1
            tokens = self._fit_length(tokens, spec, tic_words)
            tokens = self._enforce(tokens, spec, rng)
            text = _fix_case(_cap(_join(tokens)))
            if not text or text in raw:
                continue
            raw.append(text)
            if self.plausible(text):
                filtered.append(text)
            plan.entropy = max(plan.entropy, entropy)

        seen = filtered if len(filtered) >= 2 else raw
        if not filtered:
            plan.steps.append("quality-filter-relaxed")
        plan.candidates = seen
        plan.text = seen[0] if seen else ""
        plan.seed = seed
        if plan.text:
            plan.tokens = _tokenize(plan.text)
            plan.novelty = self.index.novelty(plan.tokens)
        if tics:
            plan.steps.append(f"tic x{tics}")
        plan.steps.append(
            f"temp={temperature:.2f} kept={len(seen)} of {len(raw)} walked")
        self.n_planned += 1
        return plan

    def _fit_length(
        self, tokens: List[str], spec: Dict[str, Any], keep: Sequence[str],
    ) -> List[str]:
        """
        Bring a line back inside the relation's length contract.

        Working a signature phrase into a line can push it well past
        ``max_words``, which silently breaks a promise the relation made to
        the caller -- a game that asked for a four-word barks and got nine is
        going to render it off the edge of the dialogue box.

        The words in ``keep`` (the signature phrase) survive; filler is
        dropped from the middle first, because losing the middle of a
        Markov-walked line costs nothing and losing the catchphrase costs the
        character. If it still will not fit, the tic is dropped and the walk
        is left to stand on its own.
        """
        hi = spec.get("max_words")
        if hi is None:
            return tokens
        hi = int(hi)
        keep_set = {w.lower() for w in keep}

        def words_of(seq: Sequence[str]) -> int:
            return sum(1 for t in seq if _wordish(t))

        if words_of(tokens) <= hi:
            return tokens
        keep_tokens = [t for t in tokens if t.lower() in keep_set]
        if not keep_tokens:
            return tokens[:max(1, hi)]
        # drop filler, keeping the first and last word and every kept word
        out: List[str] = []
        for i, t in enumerate(tokens):
            is_edge = i == 0 or i == len(tokens) - 1
            if is_edge or t.lower() in keep_set or not _wordish(t):
                out.append(t)
            if words_of(out) >= hi and words_of(keep_tokens) < hi:
                break
        if words_of(out) > hi or words_of(out) < words_of(keep_tokens):
            return self._fit_length(keep_tokens, spec, keep_tokens[:0])
        return out

    def _insert_tic(self, tokens: List[str], rng: np.random.Generator) -> Tuple[List[str], List[str]]:
        """Work a measured signature phrase into a line, at a point where a
        comma already exists if one does."""
        if not self.persona.signature:
            return tokens, []
        floor = min(sc for _, sc in self.persona.signature)
        scores = [max(0.0, sc - floor) for _, sc in self.persona.signature]
        total = float(sum(scores))
        if total <= 0.0:
            # every signature scores the same (or they are all negative after
            # the shift) -- pick uniformly rather than dividing by zero
            idx = int(rng.integers(len(scores)))
        else:
            target = rng.random() * total
            acc = 0.0
            idx = len(scores) - 1
            for i, sc in enumerate(scores):
                acc += sc
                if target <= acc:
                    idx = i
                    break
        phrase = self.persona.signature[idx][0]
        parts = _tokenize(phrase)
        if not parts:
            return tokens
        anchors = [i for i, t in enumerate(tokens) if not _wordish(t)]
        at = anchors[int(rng.integers(len(anchors)))] if anchors else len(tokens) - 1
        merged = tokens[:at] + parts + tokens[at:]
        return merged, [w.lower() for w in parts]


# ---------------------------------------------------------------------- #
# Judging feedback
# ---------------------------------------------------------------------- #

# Laplace prior weight for the learned judge. Below this many observations
# the learned good/bad signal is shrunk hard toward "no opinion", so an
# untrained judge cannot dominate the intrinsic score.
_JUDGE_PRIOR = 8.0


class Verdict:
    """
    The result of judging one utterance.

    ``intrinsic`` is the hand-written self-check (0..1, always available).
    ``extrinsic`` is the *learned* judge's estimate that this utterance was
    good (0..1), shrunk toward 0.5 until it has evidence. ``reward`` is the
    signed blend actually pushed back into the robot. ``creativity`` is the
    value of the robot's creativity scalar after this exchange, so a caller
    can watch the robot get looser or tighter line by line.
    """

    __slots__ = ("reward", "good", "intrinsic", "extrinsic", "terms",
                 "creativity", "novelty", "source")

    def __init__(self, reward: float, good: bool, intrinsic: float, extrinsic: float,
                 terms: Dict[str, float], creativity: float, novelty: float,
                 source: str) -> None:
        self.reward = float(reward)
        self.good = bool(good)
        self.intrinsic = float(intrinsic)
        self.extrinsic = float(extrinsic)
        self.terms = terms
        self.creativity = float(creativity)
        self.novelty = float(novelty)
        self.source = source

    def as_dict(self) -> Dict[str, Any]:
        return {
            "reward": round(self.reward, 5), "good": self.good,
            "intrinsic": round(self.intrinsic, 5), "extrinsic": round(self.extrinsic, 5),
            "creativity": round(self.creativity, 5), "novelty": round(self.novelty, 5),
            "source": self.source, "terms": {k: round(v, 5) for k, v in self.terms.items()},
        }

    def __repr__(self) -> str:
        return (
            f"<Verdict {'good' if self.good else 'bad '} r={self.reward:+.3f} "
            f"intr={self.intrinsic:.3f} ext={self.extrinsic:.3f} "
            f"nov={self.novelty:.2f} creativity={self.creativity:.3f}>"
        )


class FeedbackJudge:
    """
    Decides whether what the robot just said was good or bad, and moves two
    numbers as a result.

    **Intrinsic.** Five checks that need no training data, computed from the
    line itself:

    ``contract``   are the relation's required words actually in there
    ``length``     how close to the requested/mean length it landed
    ``novelty``    how much of it was not already in the index
    ``richness``   type-token ratio, so "the the the" cannot score well
    ``persona``    mean word-bias, i.e. does it sound like this character

    **Learned.** A *second* ``EmptyRobot`` instance, separate from the one
    that picks the line, is trained on ``good``/``bad`` for the feature
    profile of past utterances (the partner's words, the produced words, a
    length bucket, a contract-hit count, a persona-fit bucket, a novelty
    bucket). ``decide(features, options=["good","bad"])`` then estimates
    ``p(good)`` for a new utterance. It is shrunk toward 0.5 by
    ``_JUDGE_PRIOR`` observations, so it starts as a no-op and takes over
    only as feedback accumulates. This is the part that makes the robot's
    taste *its own* rather than a constant someone typed in.

    **Explicit.** If the caller passes a signal it wins: ``1``/``0``/``-1``,
    or free text scored by the same valence word lists. A human saying
    "that was awful" beats every heuristic in this class.

    **Creativity.** One scalar, moved by ``reward * novelty`` and used as
    the sampler's temperature. A rewarded *new* construction makes the robot
    try more new constructions; a punished one makes it tighten up. That is
    the whole mechanism for "gradually becoming more creative", and it is
    one inspectable number.
    """

    def __init__(
        self,
        judge_core: "EmptyRobot",
        *,
        w_intrinsic: float = 0.55,
        w_extrinsic: float = 0.45,
        creativity: float = 0.15,
        eta_up: float = 0.18,
        eta_down: float = 0.11,
        max_creativity: float = 0.80,
        baseline: float = 0.15,
        decay: float = 0.03,
        advantage_deadband: float = 0.03,
    ) -> None:
        self.core = judge_core
        self.w_intrinsic = float(w_intrinsic)
        self.w_extrinsic = float(w_extrinsic)
        self.creativity = float(np.clip(creativity, 0.0, 1.0))
        self.eta_up = float(eta_up)
        self.eta_down = float(eta_down)
        self.max_creativity = float(max_creativity)
        self.baseline = float(np.clip(baseline, 0.0, 1.0))
        self.decay = float(decay)
        self.advantage_deadband = float(advantage_deadband)
        # running mean reward, binned by whether the line was new
        self._bin_mean: Dict[str, float] = {"novel": 0.0, "familiar": 0.0}
        self._bin_n: Dict[str, int] = {"novel": 0, "familiar": 0}
        self._nov_sum = 0.0
        self._nov_n = 0

        self.n_judged = 0
        self.n_good = 0
        self.n_bad = 0
        self.n_novel_rewarded = 0
        self.n_novel_punished = 0
        self.n_explicit = 0
        self.ema_intrinsic = 0.5
        self.ema_reward = 0.0
        self.n_promoted = 0
        self.history: deque = deque(maxlen=64)
        self._promoted: Dict[str, float] = {}

    # -- intrinsic -------------------------------------------------------
    def intrinsic(
        self, text: str, spec: Dict[str, Any], persona: Persona,
        index: WordPositionIndex, novelty: float,
    ) -> Tuple[float, Dict[str, float]]:
        tokens = _tokenize(text)
        words = [w.lower() for w in _words_of(tokens)]
        terms: Dict[str, float] = {}

        required = [str(r).lower() for r in (spec.get("must_include") or [])]
        present = set(words)
        if required:
            hits = sum(1 for r in required if r in present or
                       any(r in w or w in r for w in words))
            terms["contract"] = hits / len(required)
        else:
            terms["contract"] = 1.0

        mean = spec.get("mean_words")
        if mean is None:
            lo = spec.get("min_words")
            hi = spec.get("max_words")
            mean = ((float(lo) + float(hi)) / 2.0) if (lo is not None and hi is not None) else max(1.0, len(words))
        mean = max(1.0, float(mean))
        spread = max(1.0, float(spec.get("spread", 0.35) or 0.35) * mean)
        terms["length"] = float(math.exp(-0.5 * ((len(words) - mean) / spread) ** 2))

        terms["novelty"] = float(np.clip(novelty, 0.0, 1.0))

        if words:
            ttr = len(set(words)) / len(words)
            bigrams = list(zip(words, words[1:]))
            rep = 1.0 - (len(set(bigrams)) / len(bigrams)) if len(bigrams) > 1 else 0.0
            terms["richness"] = float(np.clip(0.65 * ttr / 0.75 + 0.35 * (1.0 - rep), 0.0, 1.0))
        else:
            terms["richness"] = 0.0

        known = [w for w in words if index.known(w)]
        if words:
            in_corpus = len(known) / len(words)
            terms["persona"] = float(np.clip(
                0.5 * in_corpus
                + 0.5 * (np.mean([persona.word_bias(w) for w in words[:16]]) - 1.0) / 1.6
                + 0.5, 0.0, 1.0,
            ))
        else:
            terms["persona"] = 0.0

        weights = {"contract": 0.30, "length": 0.20, "novelty": 0.15,
                   "richness": 0.20, "persona": 0.15}
        score = sum(weights[k] * terms.get(k, 0.0) for k in weights)
        return float(np.clip(score, 0.0, 1.0)), terms

    # -- learned ---------------------------------------------------------
    def extrinsic(self, features: Dict[str, Any]) -> Tuple[float, str]:
        """
        ``(p_good, source)``. The shrinkage is honest about the foundation's
        confidence being a heuristic composition, not a probability: until
        the judge's own memory has seen enough profiles, its answer is
        pulled almost entirely to "no opinion".
        """
        try:
            msg = self.core.decide(features, options=["good", "bad"], top_k=4)
        except Exception:  # a decision core failure must not kill the loop
            return 0.5, "error"
        # The margin between the two options is a far better read-out than
        # ``msg.confidence``, which is a bounded heuristic composition and is
        # near zero for a two-way problem no matter how clear the margin is.
        # Log-squashing the margin turns the ranker's separation into a
        # usable 0..1 estimate; a tie falls out as exactly 0.5.
        scores = {str(c): float(v) for c, v in (msg.ranking or [])}
        if "good" in scores and "bad" in scores:
            margin = scores["good"] - scores["bad"]
            raw = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, margin))))
        elif msg.content == "good":
            raw = float(msg.confidence)
        elif msg.content == "bad":
            raw = 1.0 - float(msg.confidence)
        else:
            raw = 0.5
        # Shrink toward "no opinion" until there is real evidence behind the
        # margin. Without this the judge is fully confident about a
        # two-option problem from its third training episode, which is not a
        # thing a nine-example judge has earned.
        n = self.n_judged
        p = (raw * n + 0.5 * _JUDGE_PRIOR) / (n + _JUDGE_PRIOR)
        return float(np.clip(p, 0.0, 1.0)), "learned"

    def train_judge(self, features: Dict[str, Any], good: bool, reward: float) -> None:
        label = "good" if good else "bad"
        try:
            self.core.learn_episode(features, label, outcome=label, reward=abs(reward) or 0.1)
        except Exception:
            pass

    # -- explicit --------------------------------------------------------
    @staticmethod
    def signal_from_text(text: str) -> Tuple[float, str]:
        """
        A tiny opinion reader, and deliberately a small one.

        Scans for valence words, applies a one-step negation flip, and
        normalises by length. It returns 0 -- abstain -- when it sees nothing
        either way, because an abstention is more useful to the caller than a
        coin flip: with no human opinion available, ``judge()`` falls back to
        the intrinsic score, which is at least a real signal, whereas a
        fabricated +/-0.01 from three neutral words is noise.

        It is not sentiment analysis. It will misread irony, sarcasm, and
        anything longer than a short comment. What it is good at is the thing
        it is actually pointed at: reading the one-word reactions a game or an
        application collects, where "great" and "terrible" mean what they say.
        """
        words = [w.lower() for w in _words_of(_tokenize(text))]
        if not words:
            return 0.0, "text"
        score = 0.0
        hits = 0
        for i, word in enumerate(words):
            if word in _POSITIVE_MARKERS:
                value = 1.0
            elif word in _NEGATIVE_MARKERS:
                value = -1.0
            elif word in _AGGRESSIVE_MARKERS:
                value = -0.5
            else:
                continue
            # a negator in the two preceding tokens flips the reading
            if any(w in _NEGATORS for w in words[max(0, i - 2):i]):
                value = -value
            score += value
            hits += 1
        if not hits:
            return 0.0, "text"
        return float(np.clip(score / max(1.0, len(words)) * 6.0, -1.0, 1.0)), "text"

    @staticmethod
    def coerce_signal(signal: Any) -> Tuple[float, str]:
        if signal is None:
            return 0.0, "none"
        if isinstance(signal, bool):
            return (1.0 if signal else -1.0), "explicit"
        if isinstance(signal, (int, float)):
            return float(np.clip(float(signal), -1.0, 1.0)), "explicit"
        if isinstance(signal, str):
            return FeedbackJudge.signal_from_text(signal)
        return 0.0, "none"

    # -- the whole loop --------------------------------------------------
    def judge(
        self,
        features: Dict[str, Any],
        text: str,
        spec: Dict[str, Any],
        persona: Persona,
        index: WordPositionIndex,
        novelty: float,
        *,
        signal: Any = None,
    ) -> Tuple[Verdict, List[str]]:
        """
        Judge one utterance. Returns the ``Verdict`` and the content words to
        promote as seeds if the robot earned them.
        """
        intrinsic, terms = self.intrinsic(text, spec, persona, index, novelty)
        explicit, source = self.coerce_signal(signal)
        extrinsic, ext_src = self.extrinsic(features)

        if explicit:
            reward = 0.7 * explicit + 0.3 * (2.0 * intrinsic - 1.0)
            used = "explicit"
        else:
            reward = self.w_intrinsic * intrinsic + self.w_extrinsic * (2.0 * extrinsic - 1.0)
            used = "blend"

        terms["extrinsic"] = extrinsic
        good = reward > 0.0
        self.train_judge(features, good, reward)
        if explicit:
            self.n_explicit += 1
        self.n_judged += 1
        self.n_good += 1 if good else 0
        self.n_bad += 0 if good else 1

        # --- the creativity scalar -------------------------------------
        # The thing that has to be measured is not "was this rewarded" but
        # "do *new* constructions do better than familiar ones". So reward is
        # binned by novelty and the two running means are compared. That
        # distinction is the whole mechanism:
        #
        #   * an environment where everything is rewarded leaves both means
        #     equal, the advantage stays at zero, and the robot does not get
        #     noisier for nothing;
        #   * an environment where new lines are rewarded and repeated ones
        #     are not opens a gap and the robot loosens up;
        #   * an environment that punishes new lines closes the gap the other
        #     way and it tightens up.
        #
        # The bin boundary is the *running mean* novelty, not a fixed
        # threshold. A fixed threshold looks simpler and is wrong: on a small
        # corpus almost every generated line is entirely new, the "familiar"
        # bin never fills, the comparison never gets made, and the creativity
        # scalar sits frozen at its baseline forever looking like a mechanism
        # that does not respond. Asking "was this more novel than what I
        # usually produce" always splits the distribution in two.
        #
        # The gap is then shrunk by how many observations each bin actually
        # has, so the first handful of exchanges move the scalar a little
        # rather than a lot.
        before = self.creativity
        self._nov_sum += float(novelty)
        self._nov_n += 1
        boundary = self._nov_sum / self._nov_n
        is_novel = novelty > boundary
        bin_name = "novel" if is_novel else "familiar"
        seen = self._bin_n[bin_name]
        self._bin_mean[bin_name] = (
            reward if seen == 0
            else 0.88 * self._bin_mean[bin_name] + 0.12 * reward
        )
        self._bin_n[bin_name] = seen + 1
        n_novel = self._bin_n["novel"]
        n_familiar = self._bin_n["familiar"]
        confidence = min(n_novel, n_familiar) / (min(n_novel, n_familiar) + 4.0)
        advantage = (self._bin_mean["novel"] - self._bin_mean["familiar"]) * confidence
        if advantage > self.advantage_deadband:
            self.creativity += self.eta_up * advantage
            if is_novel:
                self.n_novel_rewarded += 1
        elif advantage < -self.advantage_deadband:
            self.creativity -= self.eta_down * (-advantage)
            if is_novel:
                self.n_novel_punished += 1
        self.creativity -= self.decay * (self.creativity - self.baseline)
        self.creativity = float(np.clip(self.creativity, 0.0, self.max_creativity))
        terms["advantage"] = advantage
        terms["novel_bin"] = float(n_novel)
        terms["creativity_delta"] = self.creativity - before

        self.ema_intrinsic = 0.85 * self.ema_intrinsic + 0.15 * intrinsic
        self.ema_reward = 0.85 * self.ema_reward + 0.15 * reward

        verdict = Verdict(
            reward=float(np.clip(reward, -1.0, 1.0)), good=good,
            intrinsic=intrinsic, extrinsic=extrinsic, terms=terms,
            creativity=self.creativity, novelty=novelty,
            source=used if used == "explicit" else f"{used}:{ext_src}",
        )
        self.history.append({"text": text[:120], "reward": round(reward, 4),
                             "novelty": round(novelty, 4),
                             "creativity": round(self.creativity, 4),
                             "source": verdict.source})
        return verdict, self._promote(text, index, verdict, novelty)

    def _promote(
        self, text: str, index: WordPositionIndex, verdict: Verdict, novelty: float,
    ) -> List[str]:
        """
        Turn a rewarded line into seed material for the next one. This is
        the "use some of the newly acquired data" step: nothing is
        re-ingested, the words simply gain standing in the relation's seed
        pool, and the effect is visible in the next ``plan.steps``.
        """
        if verdict.reward <= 0.15 or novelty <= _PROMOTE_NOVELTY:
            return []
        promoted: List[str] = []
        for word in _words_of(_tokenize(text)):
            w = word.lower()
            if len(w) < 3 or w in _FUNCTION_WORDS:
                continue
            if not index.known(w):
                continue
            if w not in self._promoted:
                self._promoted[w] = 0.0
                promoted.append(w)
        if promoted:
            self.n_promoted += 1
        return promoted

    def note_promoted(self, words: Sequence[str], weight: float) -> None:
        for w in words:
            self._promoted[w] = self._promoted.get(w, 0.0) + float(weight)

    def to_json(self) -> Dict[str, Any]:
        return {
            "weights": {"intrinsic": self.w_intrinsic, "extrinsic": self.w_extrinsic},
            "creativity": self.creativity,
            "eta": {"up": self.eta_up, "down": self.eta_down, "max": self.max_creativity,
                    "baseline": self.baseline, "decay": self.decay,
                    "advantage_deadband": self.advantage_deadband},
            "bins": {"mean": dict(self._bin_mean), "n": dict(self._bin_n),
                     "novelty_sum": self._nov_sum, "novelty_n": self._nov_n},
            "n_judged": self.n_judged, "n_good": self.n_good, "n_bad": self.n_bad,
            "n_novel_rewarded": self.n_novel_rewarded,
            "n_novel_punished": self.n_novel_punished,
            "n_explicit": self.n_explicit, "n_promoted": self.n_promoted,
            "ema_intrinsic": round(self.ema_intrinsic, 5),
            "ema_reward": round(self.ema_reward, 5),
            "promoted": {k: round(v, 4) for k, v in self._promoted.items()},
            "history": list(self.history),
        }

    @classmethod
    def from_json(cls, judge_core: "EmptyRobot", blob: Dict[str, Any]) -> "FeedbackJudge":
        weights = dict(blob.get("weights") or {})
        eta = dict(blob.get("eta") or {})
        obj = cls(
            judge_core,
            w_intrinsic=float(weights.get("intrinsic", 0.55)),
            w_extrinsic=float(weights.get("extrinsic", 0.45)),
            creativity=float(blob.get("creativity", 0.15)),
            eta_up=float(eta.get("up", 0.18)),
            eta_down=float(eta.get("down", 0.11)),
            max_creativity=float(eta.get("max", 0.80)),
            baseline=float(eta.get("baseline", 0.15)),
            decay=float(eta.get("decay", 0.03)),
            advantage_deadband=float(eta.get("advantage_deadband", 0.03)),
        )
        bins = dict(blob.get("bins") or {})
        mean = dict(bins.get("mean") or {})
        n = dict(bins.get("n") or {})
        obj._bin_mean = {"novel": float(mean.get("novel", 0.0)),
                         "familiar": float(mean.get("familiar", 0.0))}
        obj._bin_n = {"novel": int(n.get("novel", 0)), "familiar": int(n.get("familiar", 0))}
        obj._nov_sum = float(bins.get("novelty_sum", 0.0))
        obj._nov_n = int(bins.get("novelty_n", 0))
        obj.n_judged = int(blob.get("n_judged", 0))
        obj.n_good = int(blob.get("n_good", 0))
        obj.n_bad = int(blob.get("n_bad", 0))
        obj.n_novel_rewarded = int(blob.get("n_novel_rewarded", 0))
        obj.n_novel_punished = int(blob.get("n_novel_punished", 0))
        obj.n_explicit = int(blob.get("n_explicit", 0))
        obj.n_promoted = int(blob.get("n_promoted", 0))
        obj.ema_intrinsic = float(blob.get("ema_intrinsic", 0.5))
        obj.ema_reward = float(blob.get("ema_reward", 0.0))
        obj._promoted = {k: float(v) for k, v in (blob.get("promoted") or {}).items()}
        for item in blob.get("history") or []:
            obj.history.append(item)
        return obj


# ---------------------------------------------------------------------- #
# Relations: probable input -> probable output
# ---------------------------------------------------------------------- #


class RelationBook:
    """
    "Which outputs correspond to these probable inputs, and what should they
    contain and how long should they be."

    A relation is:

    ``{"id": "greet",
      "input":  {"keywords": ["hello", "hi"], "examples": ["hello there"]},
      "output": {"must_include": ["hello"], "preferred": ["well"],
                 "min_words": 4, "max_words": 14, "mean_words": 8,
                 "spread": 0.35, "learned": {}}}

    ``max_words: null`` means *no ceiling at all* -- length then comes from
    the corpus's learned line-length distribution, so the reply length is a
    property of the data rather than a constant in this file.

    Matching is two independent votes added together: a hashed keyword /
    exemplar overlap computed locally, and the foundation's own
    ``decide()`` over the relation ids it has been taught. When the two
    disagree the stronger one is reported, because they fail differently --
    the local vote is exact but literal, the core's is fuzzy but has
    learned what this particular partner tends to ask for.
    """

    def __init__(self, relations: Sequence[Dict[str, Any]]) -> None:
        self.relations: List[Dict[str, Any]] = []
        self._by_id: Dict[str, Dict[str, Any]] = {}
        for rel in relations or ():
            self.add(rel)

    def add(self, relation: Dict[str, Any]) -> Dict[str, Any]:
        rel = dict(relation or {})
        rel.setdefault("id", f"rel{len(self.relations)}")
        rel.setdefault("input", {})
        rel.setdefault("output", {})
        # ``to_json`` writes the promoted seed pool alongside the spec rather
        # than inside it, so put it back where the planner looks for it
        if rel.get("learned"):
            rel["output"].setdefault("learned", dict(rel["learned"]))
        rel["output"].setdefault("learned", {})
        if rel["id"] in self._by_id:
            self.relations = [r for r in self.relations if r["id"] != rel["id"]]
        self.relations.append(rel)
        self._by_id[rel["id"]] = rel
        return rel

    def get(self, rid: str) -> Optional[Dict[str, Any]]:
        return self._by_id.get(rid)

    def ids(self) -> List[str]:
        return [r["id"] for r in self.relations]

    def teach(self, core: "EmptyRobot") -> None:
        """
        Hand the relation exemplars to the foundation so it can vote on
        which relation an utterance belongs to. This is the foundation
        doing exactly what it was built for; nothing about it is specific
        to speech.
        """
        for rel in self.relations:
            for example in (rel.get("input") or {}).get("examples", []) or []:
                try:
                    core.learn(str(example), rel["id"])
                except Exception:
                    pass

    def match(
        self, text: str, core: Optional["EmptyRobot"] = None, top: int = 3,
    ) -> List[Tuple[Dict[str, Any], float, Dict[str, float]]]:
        """
        Rank relations against an utterance. Returns
        ``[(relation, score, terms)]``, best first.

        Two independent votes:

        * **literal** -- keyword hits, plus a soft substring pass so
          "dangerous" matches the keyword "danger", plus overlap with the
          relation's own example utterances. The soft pass is normalised
          against *three* hits rather than against the whole keyword list,
          because a relation with eight keywords should not need all eight
          to be a confident match, and a relation with one keyword should
          not need eight either. The exemplar overlap is normalised by the
          longer of the two sides, so a three-word example cannot score 0.67
          by accident against a five-word question.
        * **learned** -- the foundation's ``decide()`` over the relation ids
          it was taught, which is what lets the robot notice that this
          particular partner means something specific by a given phrasing.

        They are blended only when the core agrees with the local pick, so
        a core that is cold (confidence 0) simply adds nothing.
        """
        if not self.relations:
            return []
        tokens = [t.lower() for t in _tokenize(text)]
        words = set(_words_of(tokens))
        blob = " ".join(tokens)
        core_pick: Optional[str] = None
        core_conf = 0.0
        if core is not None and len(self.relations) > 1:
            # one decide() for the whole book, not one per relation -- the
            # earlier version called it inside the loop, which was six
            # full candidate-generation passes to answer one question
            try:
                msg = core.decide(text, options=self.ids(), top_k=len(self.relations))
                core_pick = str(msg.content) if msg.content is not None else None
                core_conf = float(msg.confidence)
            except Exception:
                core_pick, core_conf = None, 0.0
        scored: List[Tuple[Dict[str, Any], float, Dict[str, float]]] = []
        for rel in self.relations:
            spec_in = rel.get("input") or {}
            keywords = {str(k).lower() for k in (spec_in.get("keywords") or [])}
            keyword_hits = len(keywords & words) / max(1, len(keywords)) if keywords else 0.0
            soft = min(1.0, sum(1.0 for k in keywords if k and k in blob) / 3.0) if keywords else 0.0
            ex_hits = 0.0
            ex_shared = 0
            for example in (spec_in.get("examples") or []) or []:
                ew = {w.lower() for w in _words_of(_tokenize(str(example)))}
                if not ew:
                    continue
                shared = len(ew & words)
                if shared < 2:      # one shared word is noise, not a match
                    continue
                score = shared / max(len(ew), len(words))
                if score > ex_hits:
                    ex_hits, ex_shared = score, shared
            local = 0.6 * max(keyword_hits, soft) + 0.4 * ex_hits
            terms = {
                "local": round(local, 5), "keyword": round(keyword_hits, 5),
                "soft": round(soft, 5), "exemplar": round(ex_hits, 5),
                "exemplar_shared": ex_shared,
            }
            score = local
            if core_pick == rel["id"] and core_conf > 0.0:
                score = 0.6 * local + 0.4 * core_conf
                terms["core"] = round(core_conf, 5)
            scored.append((rel, float(score), terms))
        scored.sort(key=lambda item: -item[1])
        return scored[: int(top)]

    def to_json(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": r["id"],
                "input": r.get("input", {}),
                "output": {k: v for k, v in r.get("output", {}).items() if k != "learned"},
                "learned": r.get("output", {}).get("learned", {}),
            }
            for r in self.relations
        ]

    @classmethod
    def from_json(cls, blob: Sequence[Dict[str, Any]]) -> "RelationBook":
        return cls(blob or ())

    def __len__(self) -> int:
        return len(self.relations)

    def __repr__(self) -> str:
        return f"<RelationBook {self.ids()}>"


# ---------------------------------------------------------------------- #
# The utterance handed back to the caller
# ---------------------------------------------------------------------- #


class Utterance:
    """
    What ``EmptyTalkerRobot.speak()`` returns.

    The ergonomics deliberately match the foundation's ``Message`` so the
    same feedback pattern works whether the robot spoke a line or chose an
    action::

        line = talker.speak("hello")
        hud.say(line.text)
        line.reward()      # or line.punish(), or line.reinforce(0.3)

    ``reward()`` and ``punish()`` are not bookkeeping: they push the reward
    into the EmptyMicroRobot core that chose the line (so the trace behind
    it gets stronger or weaker), into the world model (so ``(situation,
    line) -> outcome`` is learned), into the feedback judge (so the taste
    being learned is this robot's own), into the memory ledger, and into the
    creativity scalar. One call, five places, all of them the reason the
    second line is better than the first.
    """

    __slots__ = ("text", "confidence", "alternatives", "relation", "relation_score",
                 "target_words", "novelty", "parts", "plan", "state", "elapsed_ms",
                 "warned", "_talker", "_msg", "_line", "_verdict", "_judged")

    def __init__(self, text: str, confidence: float, *, relation: str = "",
                 alternatives: Optional[List[Tuple[str, float]]] = None,
                 target_words: int = 0, novelty: float = 1.0,
                 parts: Optional[Dict[str, Any]] = None, plan: Optional[Plan] = None,
                 state: Any = None, elapsed_ms: float = 0.0,
                 talker: Optional["EmptyTalkerRobot"] = None,
                 msg: Any = None, line: str = "") -> None:
        self.text = text
        self.confidence = float(confidence)
        self.alternatives = alternatives or []
        self.relation = relation
        self.relation_score = 0.0
        self.target_words = int(target_words)
        self.novelty = float(novelty)
        self.parts = parts or {}
        self.plan = plan
        self.state = state
        self.elapsed_ms = float(elapsed_ms)
        self.warned = ""
        self._talker = talker
        self._msg = msg
        self._line = line
        self._verdict: Optional[Verdict] = None
        self._judged = False

    # -- feedback --------------------------------------------------------
    def reward(self, amount: float = 1.0, outcome: Any = None) -> Optional[Verdict]:
        return self.reinforce(abs(float(amount)), outcome=outcome)

    def punish(self, amount: float = 1.0, outcome: Any = None) -> Optional[Verdict]:
        return self.reinforce(-abs(float(amount)), outcome=outcome)

    def reinforce(self, amount: float, outcome: Any = None) -> Optional[Verdict]:
        """
        Signed feedback. ``+1`` "that was right", ``-1`` "that was wrong",
        anything between is a partial signal, and a string is read as an
        opinion by the valence word lists. The resulting ``Verdict`` is
        stored on the utterance and returned, so the caller can see what the
        robot made of it.
        """
        if self._talker is None:
            return None
        signal: Any = amount
        if outcome is not None and isinstance(outcome, str):
            signal = outcome
        return self._talker._apply_feedback(self, signal)

    def feedback(self, signal: Any = None) -> Optional[Verdict]:
        """
        Explicit feedback in one call. ``feedback(1)`` / ``reward()``,
        ``feedback(-1)`` / ``punish()``, ``feedback("that was awful")`` to
        let the valence word lists read the comment. Returns the ``Verdict``.
        """
        if signal is None:
            return self.reward()
        if isinstance(signal, (int, float)) and not isinstance(signal, bool):
            return self.reinforce(float(signal))
        return self.reinforce(1.0, outcome=str(signal))

    @property
    def verdict(self) -> Optional[Verdict]:
        return self._verdict

    def __bool__(self) -> bool:
        return bool(self.text)

    def __str__(self) -> str:
        return self.text

    def as_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 5),
            "relation": self.relation,
            "target_words": self.target_words,
            "novelty": round(self.novelty, 5),
            "alternatives": [[t, round(float(s), 5)] for t, s in self.alternatives],
            "elapsed_ms": round(self.elapsed_ms, 4),
            "verdict": self._verdict.as_dict() if self._verdict else None,
            "plan": self.plan.to_json() if self.plan else None,
        }

    def __repr__(self) -> str:
        return (
            f"Utterance({self.text!r}, confidence={self.confidence:.3f}, "
            f"relation={self.relation!r}, alternatives={len(self.alternatives)})"
        )


# ---------------------------------------------------------------------- #
# A working default, so the robot can speak with zero files
# ---------------------------------------------------------------------- #

# Twenty lines of ordinary declarative English. Not a character, not a
# parody -- just enough for the machinery to have something to walk when a
# user constructs the robot with no data. Point it at real files and this
# is replaced (or kept, as a background corpus, if you pass it as one).
SEED_CORPUS = """
Well now that is interesting, I have not thought about it in a long while.
There is more to this than it seems at first, and I would not rush to judge it.
I have seen this before, and it never ended the way anyone expected.
You are asking the right question, which is rarer than you might imagine.
The answer is simple, but simple things are rarely easy to accept.
I do not know yet, and I would rather say so than pretend otherwise.
Everything changes when you look at it long enough to see the shape of it.
You should be careful what you ask, because I will answer it honestly.
That took a long time to build, and it will take a long time to understand.
Good. Then we are agreed on the first point at least, which is something.
I remember the day clearly, and I remember what it cost everyone involved.
Nothing about this is simple, and anyone who tells you otherwise is selling.
Let us be honest about what we know and what we are only guessing at.
The same thing happened last time, and the same thing will happen again.
I have a plan, but it is not a good one, and I will not pretend otherwise.
Wait. Listen to what you just said, and hear how certain you sound.
There is a reason that nobody talks about the second half of the story.
You are not asking the question you think you are asking.
Fine. Then we do it the hard way, as is apparently the only way left.
Of course. What else would you expect me to do, sit here and wait?
"""

DEFAULT_CONFIG: Dict[str, Any] = {
    "schema": SCHEMA,
    "persona": {
        "name": "speaker",
        "background": [],
        "overrides": {},
    },
    "corpus": {"sources": [], "seed": True},
    "relations": [
        {
            "id": "greet",
            "input": {"keywords": ["hello", "hi", "hey", "greetings", "morning"],
                      "examples": ["hello there", "hey, how are you"]},
            "output": {"preferred": ["well", "hello"],
                       "min_words": 4, "max_words": 14, "mean_words": 8,
                       "spread": 0.4},
        },
        {
            "id": "ask",
            "input": {"keywords": ["what", "why", "how", "who", "where", "when",
                                   "question", "tell"],
                      "examples": ["what do you want", "why is that"]},
            "output": {"min_words": 6, "max_words": 26, "mean_words": 14,
                       "spread": 0.45},
        },
        {
            "id": "agree",
            "input": {"keywords": ["yes", "sure", "okay", "right", "correct",
                                   "exactly", "good"],
                      "examples": ["yes that is right", "exactly"]},
            "output": {"preferred": ["good", "then"],
                       "min_words": 3, "max_words": 16, "mean_words": 8,
                       "spread": 0.4},
        },
        {
            "id": "warn",
            "input": {"keywords": ["danger", "careful", "stop", "wrong", "bad",
                                   "risk", "hurt", "dead"],
                      "examples": ["be careful", "that is dangerous"]},
            "output": {"min_words": 5, "max_words": 22, "mean_words": 12,
                       "spread": 0.45},
        },
        {
            "id": "farewell",
            "input": {"keywords": ["bye", "goodbye", "leave", "see", "later",
                                   "thanks", "thank"],
                      "examples": ["goodbye", "see you later"]},
            "output": {"min_words": 3, "max_words": 12, "mean_words": 6,
                       "spread": 0.4},
        },
        {
            # No ceiling: this one's length comes from the corpus.
            "id": "monologue",
            "input": {},
            "output": {"max_words": None, "spread": 0.5},
        },
    ],
    "limits": {
        "max_vocab": 30_000,
        "max_pairs": 250_000,
        "max_transitions": 1_000_000,
        "memory_capacity": 20_000,
        "memory_records": 512,
        "n_features": 32_768,
        # how many distinct lines ever get the planner's prior written into
        # the foundation as a starting belief. Once spent, ranking runs on
        # real feedback alone. Lower it if speak() latency matters more to
        # you than having a ranked candidate set on the first few hundred
        # utterances.
        "seed_budget": 600,
    },
    "feedback": {"creativity": 0.15, "w_intrinsic": 0.55, "w_extrinsic": 0.45},
    "state": {},
}


# ---------------------------------------------------------------------- #
# The robot
# ---------------------------------------------------------------------- #


class EmptyTalkerRobot:
    """
    A speaking robot. Composed, not inherited: the EmptyMicroRobot core is
    held as ``self.core`` and used for three things (relation voting,
    choosing between generated candidate lines, and world-model learning of
    ``(situation, line) -> outcome``), and a second core instance is held by
    the ``FeedbackJudge`` for the good/bad classification.

        talker = EmptyTalkerRobot()             # zero files, zero setup
        talker.ingest_txt("character.txt")      # one or many .txt files
        talker.build_persona("Character")

        line = talker.speak("hello there")
        print(line.text, line.confidence)
        line.reward()

        talker.save("character.json")
        same = EmptyTalkerRobot.load("character.json")

    The typical closed loop in a game or an application::

        line = talker.speak(player_utterance)          # propose + choose
        result = npc.act(line)                          # your code
        line.reward() if result.ok else line.punish()   # learn
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        seed: int = 0,
        speaker_core: Optional["EmptyRobot"] = None,
        judge_core: Optional["EmptyRobot"] = None,
    ) -> None:
        cfg = _deep_merge(DEFAULT_CONFIG, config or {})
        self.config = cfg
        self.schema = str(cfg.get("schema", SCHEMA))
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._lock = threading.RLock()
        self._started = time.perf_counter()

        limits = dict(cfg.get("limits") or {})
        self.limits = limits

        # --- the foundation, used for choosing lines -------------------
        core_caps = dict(
            n_features=int(limits.get("n_features", 32_768)),
            memory_size=int(limits.get("memory_capacity", 20_000)),
        )
        self.core = speaker_core if speaker_core is not None else EmptyRobot(**core_caps)
        judge_caps = dict(core_caps)
        judge_caps["memory_size"] = max(2_000, int(limits.get("memory_capacity", 20_000)) // 4)
        self.judge_core = judge_core if judge_core is not None else EmptyRobot(**judge_caps)

        # --- the hashed store ------------------------------------------
        self.index = WordPositionIndex(
            max_vocab=int(limits.get("max_vocab", 30_000)),
            max_pairs=int(limits.get("max_pairs", 250_000)),
            max_transitions=int(limits.get("max_transitions", 1_000_000)),
        )
        self.background: Optional[WordPositionIndex] = None
        self.persona = Persona(str((cfg.get("persona") or {}).get("name", "speaker")))
        self.planner = SpeechPlanner(self.index, self.persona)
        self.memory = MemoryLedger(capacity=int(limits.get("memory_records", 512)))
        fb = dict(cfg.get("feedback") or {})
        self.judge = FeedbackJudge(
            self.judge_core,
            w_intrinsic=float(fb.get("w_intrinsic", 0.55)),
            w_extrinsic=float(fb.get("w_extrinsic", 0.45)),
            creativity=float(fb.get("creativity", 0.15)),
        )

        self.relations = RelationBook(cfg.get("relations") or ())
        self.relations.teach(self.core)

        self.n_said = 0
        self.n_empty = 0
        self.n_core_chosen = 0
        self.n_prior_chosen = 0
        self.latency_ms: List[float] = []
        self.warn = ""
        # every distinct line ever handed to the foundation as a candidate;
        # keeps the prior-teaching in _seed_candidates to once per line, and
        # bounded, so seeding cannot become the dominant cost of speaking
        self._seeded: set = set()
        self.seed_budget = int(limits.get("seed_budget", 600))

        if bool((cfg.get("corpus") or {}).get("seed", True)):
            self.ingest_text(SEED_CORPUS, source="<seed>", role="speech")

    # -- configuration ---------------------------------------------------
    def config_blob(self) -> Dict[str, Any]:
        return {
            "schema": SCHEMA,
            "persona": {
                "name": self.persona.name,
                "overrides": self.persona.overrides,
                "background": self.index.sources,
            },
            "corpus": {
                "sources": list(self.index.sources),
                "seed": "<seed>" in self.index.sources,
                "background_sources": list(self.background.sources) if self.background else [],
            },
            "relations": self.relations.to_json(),
            "limits": dict(self.limits),
            "feedback": {
                "creativity": self.judge.creativity,
                "w_intrinsic": self.judge.w_intrinsic,
                "w_extrinsic": self.judge.w_extrinsic,
            },
            "state": {},
        }

    def add_relation(self, rid: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new probable-input -> probable-output mapping at
        runtime. This is the hook a game uses to teach a new event without
        restarting: the robot can speak about it from the next line on."""
        rel = dict(spec or {})
        rel["id"] = rid
        added = self.relations.add(rel)
        for example in (added.get("input") or {}).get("examples", []) or []:
            try:
                self.core.learn(str(example), rid)
            except Exception:
                pass
        return added

    # -- ingestion -------------------------------------------------------
    def ingest_text(
        self, text: str, *, source: str = "inline", role: str = "speech",
    ) -> "EmptyTalkerRobot":
        """
        Digest one document into the hashed word-position store.

        ``role="speech"`` feeds both the language model and the persona.
        ``role="background"`` builds the null hypothesis used for signature
        mining (any other speaker, or a large neutral text file).
        Returns ``self``.
        """
        if not text:
            return self
        if role == "background":
            if self.background is None:
                self.background = WordPositionIndex(
                    max_vocab=self.index.max_vocab,
                    max_pairs=self.index.max_pairs,
                    max_transitions=self.index.max_transitions,
                )
            self.background.add_text(text, source=source)
            self.background.close()
            return self
        lines = _split_lines(text)
        self.index.add_text(text, source=source)
        self.index.close()
        self.persona.name = self.persona.name or str((self.config.get("persona") or {}).get("name", "speaker"))
        self.persona._pending_lines = list(getattr(self.persona, "_pending_lines", [])) + lines
        return self

    def ingest_txt(
        self, *paths: str, role: str = "speech", encoding: str = "utf-8",
    ) -> "EmptyTalkerRobot":
        """
        Digest one or many ``.txt`` files. A directory, a glob, or a list
        all work. Returns ``self``; call ``build_persona()`` afterwards to
        measure what was ingested.
        """
        resolved = _expand_paths(paths)
        if not resolved:
            return self
        for path in resolved:
            with open(path, "r", encoding=encoding, errors="replace") as handle:
                self.ingest_text(handle.read(), source=os.path.basename(path), role=role)
        return self

    def build_persona(
        self,
        name: str = "",
        *,
        dialogue_paths: Optional[Sequence[str]] = None,
        background_paths: Optional[Sequence[str]] = None,
        overrides: Optional[Dict[str, float]] = None,
        max_signature: int = 12,
        max_expressions: int = 48,
        own_index: bool = False,
    ) -> Persona:
        """
        Measure a speaking personality and install it.

        ``dialogue_paths`` are additional ``.txt`` files of this character's
        speech (a full transcript is fine -- a 4 MB one costs about 6 MB of
        hashed index and then nothing). ``background_paths`` is any other
        speaker's dialogue, used as the null hypothesis for signature
        mining.

        ``own_index=True`` gives the persona a private ``WordPositionIndex``
        instead of sharing the robot's, which is what you want when the
        speaker's vocabulary should be *limited* rather than shared -- a
        talking appliance, a child, a character who never swears. The default
        is to share, because a second copy of the same corpus doubles the
        resident bytes and teaches the persona nothing. Returns the
        ``Persona``.

        For a Marvel character::

            talker.ingest_txt("thanos_transcript.txt", role="speech")
            talker.ingest_txt("generic_english.txt",  role="background")
            talker.build_persona("Thanos", overrides={"aggression": 0.9})

        The overrides are optional and only ever nudge; the measured
        numbers come from the data.
        """
        if name:
            self.persona.name = name
        if dialogue_paths:
            for path in _expand_paths(dialogue_paths):
                with open(path, "r", encoding="utf-8", errors="replace") as handle:
                    self.ingest_text(handle.read(), source=os.path.basename(path), role="speech")
        if background_paths:
            for path in _expand_paths(background_paths):
                with open(path, "r", encoding="utf-8", errors="replace") as handle:
                    self.ingest_text(handle.read(), source=os.path.basename(path), role="background")
        if not self.persona.name:
            self.persona.name = str((self.config.get("persona") or {}).get("name", "speaker"))

        lines = list(getattr(self.persona, "_pending_lines", []))
        if not lines:
            lines = [e["text"] for e in self.persona.expressions]
        if own_index:
            self.persona.owns_index = True
            self.persona.index = WordPositionIndex(
                max_vocab=self.index.max_vocab,
                max_pairs=self.index.max_pairs,
                max_transitions=self.index.max_transitions,
            )
        else:
            self.persona.owns_index = False
            self.persona.index = self.index
        self.persona.fit(
            lines,
            background=self.background or _seed_background(),
            max_signature=max_signature,
            max_expressions=max_expressions,
            overrides=overrides or (self.config.get("persona") or {}).get("overrides"),
        )
        self._rebind()
        self._teach_expressions()
        return self.persona

    def _teach_expressions(self) -> None:
        """
        Let the foundation retrieve this character's own lines. A taught pair
        is (expression, expression) under its relation, which is what makes
        ``recall`` able to hand back something the persona actually said.
        """
        for expr in self.persona.expressions[:32]:
            text = str(expr.get("text") or "")
            if not text:
                continue
            try:
                self.core.learn(
                    {"style": self.persona.name, "line": text, "traits": expr.get("traits", {})},
                    text,
                )
            except Exception:
                pass

    def _match_is_decisive(self, said: str, margin: float = 1.6) -> bool:
        """
        True when a single relation's literal score clearly beats the rest.
        Measured without the core: the cheapest possible pass over keyword,
        soft-substring and exemplar evidence.
        """
        tokens = [t.lower() for t in _tokenize(said)]
        words = set(_words_of(tokens))
        blob = " ".join(tokens)
        scores: List[float] = []
        for rel in self.relations.relations:
            spec_in = rel.get("input") or {}
            keywords = {str(k).lower() for k in (spec_in.get("keywords") or [])}
            soft = min(1.0, sum(1.0 for k in keywords if k and k in blob) / 3.0) if keywords else 0.0
            exact = len(keywords & words) / max(1, len(keywords)) if keywords else 0.0
            ex = 0.0
            for example in (spec_in.get("examples") or []) or []:
                ew = {w.lower() for w in _words_of(_tokenize(str(example)))}
                if ew and len(ew & words) >= 2:
                    ex = max(ex, len(ew & words) / max(len(ew), len(words)))
            scores.append(0.6 * max(soft, exact) + 0.4 * ex)
        if len(scores) < 2:
            return True
        scores.sort(reverse=True)
        return scores[0] > 0.0 and scores[0] >= margin * scores[1]

    def _rebind(self) -> None:
        """
        Re-point the planner at the objects currently installed on the robot.

        Necessary because ``from_json`` swaps ``self.index`` and
        ``self.persona`` for freshly deserialised ones, and a planner still
        holding the constructor-time objects would generate from an empty
        store -- which fails silently, by saying nothing.
        """
        self.planner.index = self.index
        self.planner.persona = self.persona

    def _ensure_persona(self) -> None:
        """
        Fit the persona on first use if it was never fitted. Without this the
        robot ships with every trait at exactly 0.5 -- i.e. with no
        personality at all -- until the caller remembers to call
        ``build_persona()``, which is the wrong default for something
        advertised as a speaking robot.
        """
        if self.persona.n_lines:
            return
        pending = list(getattr(self.persona, "_pending_lines", []))
        if not pending:
            return
        self.build_persona()

    def _seed_candidates(
        self, candidates: Sequence[str], state_desc: Dict[str, Any], spec: Dict[str, Any],
    ) -> None:
        """
        Give the foundation a starting belief about each candidate line.

        This is the join between the two halves of the system. The planner
        can score a line with no memory at all, but the foundation's
        ``decide()`` has no way to rank options it has never seen -- it would
        return an arbitrary pick at confidence 0 and the feedback loop would
        have nothing to attach to. So each *newly proposed* line is taught
        once, with the planner's intrinsic prior as its reward. The
        foundation then ranks by that prior, real feedback overwrites it
        through ``Message.reinforce()`` and the world model, and after a few
        exchanges the selection is the foundation's judgement rather than the
        planner's.

        Once per distinct line, not once per call: the point is a prior, and
        re-teaching it every turn would drown out the real signal. And only
        while the budget lasts -- bootstrapping has an end. Past
        ``seed_budget`` lines the prior has done its job and the
        foundation ranks on real feedback alone, which is both cheaper and
        more honest than a prior that never stops being re-taught.
        """
        if len(self._seeded) >= self.seed_budget:
            return
        room = self.seed_budget - len(self._seeded)
        for line in candidates:
            if line in self._seeded or room <= 0:
                continue
            try:
                prior = self.judge.intrinsic(
                    line, spec, self.persona, self.index,
                    self.index.novelty(_tokenize(line)),
                )[0]
            except Exception:
                prior = 0.4
            try:
                self.core.learn_episode(
                    state_desc, line, outcome="proposed", reward=float(prior),
                )
            except Exception:
                continue
            self._seeded.add(line)
            room -= 1

    # -- speaking --------------------------------------------------------
    def features(self, utterance: str, relation: str, spec: Dict[str, Any],
                 persona_fit: float, novelty: float) -> Dict[str, Any]:
        """
        The feature profile handed to both EmptyMicroRobot instances.

        It is a plain dict on purpose: the foundation tokenizes dicts
        natively, so this needs no encoder, no vocabulary and no
        fixed-length vector -- the hashing is the foundation's job.
        """
        words = _words_of(_tokenize(utterance))
        required = [str(r).lower() for r in (spec.get("must_include") or [])]
        present = {w.lower() for w in words}
        hits = sum(1 for r in required if r in present)
        return {
            "said": utterance,
            "length": len(words),
            "len_bucket": min(8, len(words) // 4),
            "relation": relation,
            "persona": self.persona.name,
            "contract_hits": hits,
            "contract_missed": max(0, len(required) - hits),
            "fit_bucket": int(np.clip(persona_fit, 0.0, 1.0) * 5),
            "novelty_bucket": int(np.clip(novelty, 0.0, 1.0) * 5),
            "distinct": len(set(w.lower() for w in words)),
        }

    def _prior_ranking(
        self, candidates: Sequence[str], spec: Dict[str, Any],
    ) -> List[Tuple[str, float]]:
        """
        Rank candidates by the intrinsic prior, tie-broken by how novel each
        one is so the robot does not become deterministic. Used only when the
        foundation has no opinion (see ``speak``).
        """
        scored: List[Tuple[str, float]] = []
        for line in candidates:
            try:
                base = self.judge.intrinsic(
                    line, spec, self.persona, self.index,
                    self.index.novelty(_tokenize(line)),
                )[0]
            except Exception:
                base = 0.0
            scored.append((line, float(base)))
        scored.sort(key=lambda kv: -kv[1])
        return scored

    def _persona_fit(self, text: str) -> float:
        words = [w.lower() for w in _words_of(_tokenize(text))][:24]
        if not words:
            return 0.0
        known = [w for w in words if self.index.known(w)]
        if not known:
            return 0.0
        bias = float(np.mean([self.persona.word_bias(w) for w in known]))
        return float(np.clip((bias - 0.35) / 2.25, 0.0, 1.0))

    def speak(
        self,
        text: Any = None,
        *,
        relation: Optional[str] = None,
        width: int = 5,
        energy: float = 1.0,
        creativity: Optional[float] = None,
        state: Optional[Dict[str, Any]] = None,
    ) -> Utterance:
        """
        Say something.

        ``text`` is what the partner said (or a game event name, or
        ``None`` for an unprompted monologue). With no text, or text that
        matches nothing, the ``monologue`` relation is used and the length
        comes from the corpus.

        The call does four things in order: pick the relation, plan several
        complete candidate sentences, let the foundation choose between
        them, and measure how novel the winner is. Nothing is emitted until
        the foundation has ranked it, so ``alternatives`` is always
        populated and always inspectable.
        """
        started = time.perf_counter()
        with self._lock:
            self._ensure_persona()
            said = "" if text is None else (text if isinstance(text, str) else json.dumps(text, default=str))
            matches = self.relations.match(
                said,
                # The learned vote only earns its cost when the literal one is
                # ambiguous. On an obvious match it is a full candidate
                # generation pass over the whole core for a number that
                # cannot change the answer.
                core=self.core if not self._match_is_decisive(said) else None,
                top=3,
            )
            chosen: Optional[Dict[str, Any]] = None
            if relation is not None:
                chosen = self.relations.get(relation)
            if chosen is None and matches:
                best_rel, best_score, terms = matches[0]
                if best_score > 0.0 or len(self.relations) == 1:
                    chosen = best_rel
            if chosen is None:
                chosen = self.relations.get("monologue") or (
                    self.relations.relations[0] if self.relations.relations else None)
            if chosen is None:
                self.n_empty += 1
                return Utterance("", 0.0, relation="", state=state,
                                 talker=self, elapsed_ms=(time.perf_counter() - started) * 1000.0)

            rel_id = str(chosen.get("id", ""))
            spec = dict(chosen.get("output") or {})
            spec["learned"] = dict(self.judge._promoted)
            temp_creativity = self.judge.creativity if creativity is None else float(creativity)

            plan = self.planner.plan(
                chosen, _tokenize(said), self._rng,
                creativity=temp_creativity, energy=energy, width=width,
            )
            candidates = [c for c in plan.candidates if c]
            if not candidates:
                self.n_empty += 1
                return Utterance("", 0.0, relation=rel_id, state=state,
                                 talker=self, plan=plan,
                                 elapsed_ms=(time.perf_counter() - started) * 1000.0)

            # The *state* is the situation, never the candidate. Folding
            # the proposed line into the query would make every query unique
            # (maximal novelty, poor retrieval) and would put the answer
            # inside the question. Line is the action; situation is the state.
            state_desc = {
                "kind": "utterance", "relation": rel_id,
                "persona": self.persona.name, "input": (said or rel_id)[:400],
            }
            self._seed_candidates(candidates, state_desc, spec)

            best = candidates[0]
            confidence = 0.0
            ranking: List[Tuple[str, float]] = [(c, 0.0) for c in candidates]
            source = "planner"
            msg = None
            if len(candidates) > 1:
                try:
                    msg = self.core.decide(
                        state_desc, options=candidates, top_k=len(candidates),
                    )
                    ranking = [(str(c), float(s)) for c, s in (msg.ranking or [])]
                    confidence = float(msg.confidence)
                except Exception:
                    msg = None
                    ranking = []

            # A foundation with no experience of any of these exact lines has
            # nothing to say about them: every candidate scores identically and
            # confidence is 0. That is correct behaviour for the core, and
            # useless for a speaking robot, so the planner's intrinsic prior
            # breaks the tie. The core leads whenever it has an opinion; this
            # is the floor, and which one decided is reported rather than
            # hidden.
            if not _discriminates(ranking, confidence):
                ranking = self._prior_ranking(candidates, spec)
                if ranking:
                    best = ranking[0][0]
                    confidence = _spread(ranking)
                    source = "planner-prior"
            elif ranking and ranking[0][0] in candidates:
                best = ranking[0][0]
                source = "core"
            self.n_core_chosen += 1 if source == "core" else 0
            self.n_prior_chosen += 1 if source == "planner-prior" else 0
            if not ranking:
                ranking = [(c, 1.0 / (i + 1)) for i, c in enumerate(candidates)]

            fit = self._persona_fit(best)
            novelty = self.index.novelty(_tokenize(best))
            if plan.text != best:
                plan.text = best
                plan.tokens = _tokenize(best)
                plan.novelty = novelty

            self.n_said += 1
            elapsed = (time.perf_counter() - started) * 1000.0
            self.latency_ms.append(elapsed)
            if len(self.latency_ms) > 512:
                del self.latency_ms[:256]

        return Utterance(
            best, confidence, relation=rel_id, alternatives=ranking,
            target_words=plan.target_words, novelty=novelty,
            parts={
                "relation_score": round(float(matches[0][1]), 5) if matches else 0.0,
                "match_terms": {k: round(v, 5) for k, v in (matches[0][2].items() if matches else [])},
                "candidates": len(candidates),
                "seed": plan.seed,
                "entropy": round(plan.entropy, 5),
                "persona_fit": round(fit, 5),
                "creativity": round(temp_creativity, 5),
                "temperature": round(0.30 + 0.90 * temp_creativity, 5),
                "walk": plan.steps,
                "source": source,
            },
            plan=plan, state=state or said, elapsed_ms=elapsed,
            talker=self, msg=msg, line=best,
        )

    def respond(self, text: Any = None, **kwargs: Any) -> Utterance:
        """Alias for ``speak()``; reads better in a dialogue loop."""
        return self.speak(text, **kwargs)

    def voice(self, event: Dict[str, Any], **kwargs: Any) -> Utterance:
        """
        The game-facing shorthand: hand it an event dict, get one line back.

            line = talker.voice({"event": "npc_greeted", "mood": "hostile"})

        ``event`` is used as the relation id when it matches one, otherwise
        its string form is matched like any other utterance.
        """
        event = dict(event or {})
        rid = str(event.get("event") or event.get("relation") or "")
        if rid and self.relations.get(rid):
            return self.speak(event.get("text") or rid, relation=rid, **kwargs)
        return self.speak(json.dumps(event, default=str) if event else None, **kwargs)

    # -- feedback --------------------------------------------------------
    def _apply_feedback(self, utterance: Utterance, signal: Any) -> Verdict:
        """
        The single place feedback lands. Five destinations, in this order:
        the foundation trace that produced the line, the foundation world
        model keyed on (situation, line), the learned good/bad judge, the
        creativity scalar, and the memory ledger.
        """
        with self._lock:
            if utterance._judged:
                return utterance._verdict  # type: ignore[return-value]
            utterance._judged = True

            rel = self.relations.get(utterance.relation) or {}
            spec = dict(rel.get("output") or {})
            said = utterance.text
            heard = "" if utterance.state is None else str(utterance.state)

            feats = self.features(
                said, utterance.relation, spec,
                self._persona_fit(said), utterance.novelty,
            )
            feats["input"] = heard[:400]

            verdict, promoted = self.judge.judge(
                feats, said, spec, self.persona, self.index, utterance.novelty,
                signal=signal,
            )
            utterance._verdict = verdict

            # 1-2: push the reward into the foundation, both the trace vote
            # and the empirical world model
            if utterance._msg is not None:
                try:
                    utterance._msg.reinforce(
                        verdict.reward,
                        outcome="good" if verdict.good else "bad",
                    )
                except Exception:
                    pass
            try:
                self.core.learn_episode(
                    {"input": heard[:400], "relation": utterance.relation,
                     "persona": self.persona.name, "said": said},
                    said,
                    outcome="good" if verdict.good else "bad",
                    reward=verdict.reward,
                )
            except Exception:
                pass

            # 3: the words of a rewarded novel line gain standing as seeds
            if promoted:
                weight = float(np.clip(verdict.reward, 0.0, 1.0))
                self.judge.note_promoted(promoted, weight)
                spec.setdefault("learned", {})
                for word in promoted:
                    spec["learned"][word] = round(
                        float(spec["learned"].get(word, 0.0)) + weight, 5
                    )
                # keep the seed pool from drifting into a loop
                if len(spec["learned"]) > 64:
                    ranked = sorted(spec["learned"].items(), key=lambda kv: -kv[1])[:64]
                    spec["learned"] = dict(ranked)

            # 4-5: memory, and a last-resort reinforcement when there is no
            # message to reinforce (single-candidate calls)
            self.memory.record(
                heard or utterance.relation, said,
                relation=utterance.relation, reward=verdict.reward,
                novelty=utterance.novelty,
            )
            try:
                self.core.reinforce_last(
                    verdict.reward,
                    outcome="good" if verdict.good else "bad",
                ) if utterance._msg is None else None
            except Exception:
                pass
            return verdict

    def feedback(self, utterance: Utterance, signal: Any = None) -> Optional[Verdict]:
        """Explicit alias for ``utterance.reward()`` / ``utterance.punish()``."""
        if signal is None:
            return utterance.reward()
        if isinstance(signal, (int, float)) and not isinstance(signal, bool):
            return utterance.reinforce(float(signal))
        return utterance.reinforce(1.0 if signal else -1.0, outcome=str(signal))

    # -- memory ----------------------------------------------------------
    def recall(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """What the robot said when this came up before. Same hashed store
        as the language model -- see ``MemoryLedger``."""
        return self.memory.recall(query, k)

    def remember(self, event: str, line: str, **kwargs: Any) -> MemoryRecord:
        """
        Write something into memory without speaking. For world facts the
        robot should be able to recall but not paraphrase.
        """
        return self.memory.record(event, line, **kwargs)

    # -- the foundation, still reachable --------------------------------
    def decide(self, situation: Any, options: Optional[List[Any]] = None) -> Any:
        """
        Direct passthrough to the foundation's ``decide()``, for the
        non-speaking half of a robot's brain::

            move = talker.decide({"hp": 0.1, "ammo": 0}, options=["flee", "hide"])

        Returns the foundation's ``Message``, so ``reward()`` / ``punish()``
        work exactly as documented there. It is not wired into the speech
        loop and does not learn from the speech loop's rewards.
        """
        return self.core.decide(situation, options=options)

    def simulate(self, situation: Any, action: Any = None) -> Dict[str, Any]:
        """Passthrough to the foundation's empirical world model."""
        return self.core.simulate(situation, action)

    def recall_expression(self, context: str, k: int = 3) -> List[str]:
        """
        Ask the foundation for this persona's own lines that fit a context.
        Useful for a "catchphrase" button: the answers are real lines from
        the ingested dialogue, not generated ones.
        """
        try:
            msg = self.core.response(
                {"style": self.persona.name, "context": context}, top_k=max(2, k)
            )
        except Exception:
            return []
        return [str(c) for c, _ in (msg.ranking or [])[:k]]

    # -- introspection ---------------------------------------------------
    def memory_report(self) -> Dict[str, Any]:
        """The hashed-data claim, measured. Compare ``index_bytes`` and
        ``total_bytes`` against ``source_bytes`` and against what the same
        text would cost as Python token lists."""
        report = self.index.memory_report()
        report["background_bytes"] = self.background.nbytes if self.background else 0
        report["persona_index_bytes"] = (
            report["index_bytes"] if self.persona.owns_index
            and self.persona.index is not self.index else 0
        )
        report["memory_index_bytes"] = (
            self.memory.index._pos.nbytes + self.memory.index._uni.nbytes
        )
        report["grand_total_bytes"] = (
            report["total_bytes"] + report["persona_index_bytes"]
            + report["background_bytes"] + report["memory_index_bytes"]
        )
        return report

    def stats(self) -> Dict[str, Any]:
        """Everything worth watching, in one dict."""
        report = self.memory_report()
        lat = sorted(self.latency_ms) if self.latency_ms else [0.0]
        return {
            "persona": {
                "name": self.persona.name,
                "lines": self.persona.n_lines,
                "mean_words": round(self.persona.mean_words, 3),
                "traits": {k: round(v, 4) for k, v in self.persona.traits.items()},
                "overrides": self.persona.overrides,
                "signature": self.persona.signature[:5],
                "expressions": len(self.persona.expressions),
            },
            "corpus": {
                "docs": self.index.n_docs,
                "lines": self.index.n_lines,
                "tokens": self.index.n_tokens,
                "vocab": len(self.index),
                "vocab_open": self.index.vocab_open,
                "transitions": len(self.index._nxt),
                "pos_pairs": len(self.index._pos),
                "prune_passes": self.index.pruned,
                "words_dropped": self.index.words_dropped,
                "sources": self.index.sources[:12],
                "background_sources": (
                    list(self.background.sources)[:6] if self.background else []),
                "background_words": self.background.vocab() if self.background else 0,
            },
            "memory_bytes": {
                "index": report["index_bytes"],
                "lexicon": report["lexicon_bytes"],
                "total": report["grand_total_bytes"],
                "source": report["source_bytes"],
                "bytes_per_token": report["bytes_per_token"],
                "naive_python_bytes_per_token": report["naive_python_bytes_per_token"],
            },
            "speech": {
                "said": self.n_said,
                "empty": self.n_empty,
                "chosen_by_core": self.n_core_chosen,
                "chosen_by_prior": self.n_prior_chosen,
                "seed_budget": self.seed_budget,
                "seeded": len(self._seeded),
                "relations": len(self.relations),
                "planned": self.planner.n_planned,
                "forced_words": self.planner.n_forced,
                "latency_p50_ms": round(lat[len(lat) // 2], 4),
                "latency_p95_ms": round(lat[min(len(lat) - 1, int(len(lat) * 0.95))], 4),
            },
            "feedback": {
                "judged": self.judge.n_judged,
                "good": self.judge.n_good,
                "bad": self.judge.n_bad,
                "explicit": self.judge.n_explicit,
                "creativity": round(self.judge.creativity, 5),
                "novel_rewarded": self.judge.n_novel_rewarded,
                "novel_punished": self.judge.n_novel_punished,
                "ema_intrinsic": round(self.judge.ema_intrinsic, 5),
                "ema_reward": round(self.judge.ema_reward, 5),
                "promoted_words": len(self.judge._promoted),
                "judge_memory": len(self.judge_core),
                "reward_novel": round(self.judge._bin_mean["novel"], 5),
                "reward_familiar": round(self.judge._bin_mean["familiar"], 5),
                "novelty_boundary": round(
                    self.judge._nov_sum / self.judge._nov_n, 4) if self.judge._nov_n else 0.0,
            },
            "memory": {
                "records": len(self.memory),
                "capacity": self.memory.capacity,
                "recorded": self.memory.n_recorded,
                "evicted": self.memory.n_evicted,
            },
            "core": {
                "n_memory": len(self.core),
                "n_actions_known": self.core.stats().get("n_actions_known"),
                "n_world_cells": self.core.stats().get("world_cells"),
                "n_rules": self.core.stats().get("n_rules"),
                "latency_ema_ms": self.core.stats().get("latency_ema_ms"),
            },
            "uptime_s": round(time.perf_counter() - self._started, 3),
        }

    def explain(self, utterance: Utterance) -> str:
        """A readable account of one utterance, decision by decision."""
        parts = utterance.parts or {}
        plan = utterance.plan
        verdict = utterance._verdict
        lines = [
            f"relation      {utterance.relation!r} "
            f"(score {parts.get('relation_score', 0.0):.3f}, {parts.get('match_terms')})",
            f"persona       {self.persona.describe()}",
            f"length        {utterance.target_words} words targeted"
            + (f", {len(_words_of(_tokenize(utterance.text)))} produced" if utterance.text else ""),
            f"seed          {parts.get('seed', '-')!r}   "
            f"creativity {parts.get('creativity', 0.0):.3f} -> "
            f"temperature {parts.get('temperature', 0.0):.3f}",
            f"walk          {parts.get('walk', [])}   "
            f"entropy {parts.get('entropy', 0.0):.3f}",
            f"candidates    {parts.get('candidates', 0)} proposed, "
            f"chosen by {parts.get('source', 'planner')}",
            f"novelty       {utterance.novelty:.3f}   "
            f"persona_fit {parts.get('persona_fit', 0.0):.3f}",
            f"confidence    {utterance.confidence:.3f}",
            f"alternatives  " + ", ".join(
                f"{t[:28]!r}={s:.2f}" for t, s in utterance.alternatives[:4]),
        ]
        if plan is not None and plan.candidates:
            lines.append(f"line          {plan.candidates[0][:70]!r}")
        if verdict is not None:
            terms = " ".join(f"{k}={v:.3f}" for k, v in verdict.terms.items())
            lines.append(
                f"verdict       {verdict!r}\n              {terms}"
            )
        if not utterance.text:
            lines.append("verdict       nothing said -- no candidate could be built")
        return "\n".join(lines)

    def report(self, lines: int = 6) -> str:
        """A human-readable version of ``stats()``."""
        s = self.stats()
        c, f, sp, fb = s["corpus"], s["feedback"], s["speech"], s["memory_bytes"]
        return "\n".join([
            f"persona   {s['persona']['name']}  "
            + " ".join(f"{k}={v:.2f}" for k, v in s["persona"]["traits"].items()),
            f"signature " + (", ".join(f"{t!r}" for t, _ in s["persona"]["signature"][:3]) or "-"),
            f"corpus    {c['docs']} docs / {c['lines']} lines / {c['tokens']} tokens, "
            f"vocab {c['vocab']} ({'open' if c['vocab_open'] else 'capped'}), "
            f"{c['transitions']} transitions",
            f"bytes     {fb['total'] / 1024:.1f} KiB resident for "
            f"{fb['source'] / 1024:.1f} KiB of text "
            f"({fb['bytes_per_token']:.1f} B/token vs "
            f"{fb['naive_python_bytes_per_token']:.0f} B/token as Python token lists)",
            f"speech    {sp['said']} said / {sp['empty']} empty, "
            f"chosen by core {sp['chosen_by_core']} / prior {sp['chosen_by_prior']}, "
            f"p50 {sp['latency_p50_ms']:.2f} ms, p95 {sp['latency_p95_ms']:.2f} ms",
            f"feedback  {f['judged']} judged ({f['good']} good / {f['bad']} bad), "
            f"creativity {f['creativity']:.3f}, "
            f"{f['novel_rewarded']} novel rewarded / {f['novel_punished']} punished",
            f"memory    {s['memory']['records']}/{s['memory']['capacity']} records, "
            f"{f['promoted_words']} promoted seed words",
        ])

    # -- persistence -----------------------------------------------------
    def to_json(
        self, *, include_index: bool = True, include_background: bool = False,
        indent: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        The whole robot as plain JSON: config, the packed hashed index, the
        persona measurements, the relation book with its learned seeds, the
        memory ledger, the feedback history and the creativity scalar.

        The index is base64'd zlib-compressed ``uint64`` columns, which is
        most of the point: a 4 MB transcript saves as a JSON file that is a
        small fraction of it, because the text itself is not in there.

        The **background** corpus is not saved by default. It exists only to
        be contrasted against during ``build_persona()``, and the results of
        that contrast -- signature phrases, expression rankings -- are already
        in the file. Shipping it would put a second, redundant copy of a
        corpus the robot never queries at runtime into every save. Pass
        ``include_background=True`` if you would rather not re-supply it.
        """
        blob = self.config_blob()
        blob["state"] = {
            "said": self.n_said,
            "empty": self.n_empty,
            "chosen_by_core": self.n_core_chosen,
            "chosen_by_prior": self.n_prior_chosen,
            "latency_ms": [round(x, 4) for x in self.latency_ms[-64:]],
            "index": self.index.to_json() if include_index else None,
            "background": (
                self.background.to_json()
                if (include_index and include_background and self.background)
                else None
            ),
            "persona": self.persona.to_json(include_index=include_index),
            "memory": self.memory.to_json(),
            "feedback": self.judge.to_json(),
            # the foundation's own memory is not re-serialised (it has its
            # own save/load); this is only the bookkeeping of which lines
            # have already been given a prior
            "seeded": len(self._seeded),
        }
        return blob

    def save(
        self, path: str, *, include_index: bool = True,
        include_background: bool = False, indent: Optional[int] = 1,
    ) -> str:
        """Write the robot to a ``.json`` file. Returns the path."""
        blob = self.to_json(include_index=include_index,
                            include_background=include_background)
        directory = os.path.dirname(os.path.abspath(path))
        if directory and not os.path.isdir(directory):
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(blob, handle, indent=indent, ensure_ascii=False, default=_json_default)
        return path

    @classmethod
    def from_json(
        cls, blob: Dict[str, Any], *, seed: int = 0, **kwargs: Any,
    ) -> "EmptyTalkerRobot":
        """Rebuild a robot from a dict produced by ``to_json()``."""
        blob = dict(blob or {})
        state = dict(blob.get("state") or {})
        obj = cls({k: v for k, v in blob.items() if k != "state"}, seed=seed, **kwargs)
        if blob.get("schema") and blob["schema"] != SCHEMA:
            obj.warn = f"loaded schema {blob['schema']!r}, expected {SCHEMA!r}"
        if state.get("index"):
            obj.index = WordPositionIndex.from_json(state["index"])
        if state.get("background"):
            obj.background = WordPositionIndex.from_json(state["background"])
        if state.get("persona"):
            obj.persona = Persona.from_json(state["persona"])
            if not obj.persona.owns_index:
                obj.persona.index = obj.index
        obj._rebind()
        if state.get("memory"):
            obj.memory = MemoryLedger.from_json(state["memory"])
        if state.get("feedback"):
            obj.judge = FeedbackJudge.from_json(obj.judge_core, state["feedback"])
        obj.relations = RelationBook.from_json(blob.get("relations") or ())
        obj.relations.teach(obj.core)
        obj.n_said = int(state.get("said", 0))
        obj.n_empty = int(state.get("empty", 0))
        obj.n_core_chosen = int(state.get("chosen_by_core", 0))
        obj.n_prior_chosen = int(state.get("chosen_by_prior", 0))
        obj.latency_ms = [float(x) for x in (state.get("latency_ms") or [])]
        return obj

    @classmethod
    def load(cls, path: str, **kwargs: Any) -> "EmptyTalkerRobot":
        """Read a robot back from a ``.json`` file written by ``save()``."""
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_json(json.load(handle), **kwargs)

    def reset(self) -> "EmptyTalkerRobot":
        """
        Forget everything learned but keep the corpus, the persona and the
        relations. Use it to compare a cold robot against a trained one on
        the same data.
        """
        with self._lock:
            self.core = EmptyRobot(
                n_features=int(self.limits.get("n_features", 32_768)),
                memory_size=int(self.limits.get("memory_capacity", 20_000)),
            )
            self.judge_core = EmptyRobot(
                n_features=int(self.limits.get("n_features", 32_768)),
                memory_size=max(2_000, int(self.limits.get("memory_capacity", 20_000)) // 4),
            )
            self.relations.teach(self.core)
            self.judge = FeedbackJudge(
                self.judge_core,
                w_intrinsic=self.judge.w_intrinsic,
                w_extrinsic=self.judge.w_extrinsic,
                creativity=0.05,
            )
            self.memory = MemoryLedger(capacity=self.memory.capacity)
            self.n_said = 0
            self.n_empty = 0
            self.n_core_chosen = 0
            self.n_prior_chosen = 0
            self.latency_ms = []
            self._seeded = set()
        return self

    def __repr__(self) -> str:
        return (
            f"<EmptyTalkerRobot {self.persona.name!r} "
            f"tokens={self.index.n_tokens} vocab={len(self.index)} "
            f"said={self.n_said} creativity={self.judge.creativity:.3f}>"
        )


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


def _discriminates(ranking: Sequence[Tuple[str, float]], confidence: float) -> bool:
    """
    Did the foundation actually express a preference? A ranking where every
    option scored the same is a ranking it could not make, whatever the
    number of alternatives was.
    """
    if not ranking or len(ranking) < 2:
        return False
    if confidence <= 0.0:
        return False
    scores = [float(s) for _, s in ranking]
    return (max(scores) - min(scores)) > 1e-9


def _spread(ranking: Sequence[Tuple[str, float]]) -> float:
    """
    Turn a prior ranking into a 0..1 confidence: how far the best candidate
    sits above the median, squashed. Deliberately modest -- it is a prior, not
    an observation.
    """
    if len(ranking) < 2:
        return 0.0
    scores = sorted((float(s) for _, s in ranking), reverse=True)
    median = scores[len(scores) // 2]
    return float(np.clip(0.5 * math.tanh(4.0 * (scores[0] - median)), 0.0, 1.0))


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in (patch or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _expand_paths(paths: Sequence[Any]) -> List[str]:
    """Accept strings, lists, globs and directories; return sorted file paths."""
    import glob as _glob

    flat: List[str] = []
    for item in paths or ():
        if isinstance(item, (list, tuple, set)):
            flat.extend(str(x) for x in item)
        else:
            flat.append(str(item))
    out: List[str] = []
    for item in flat:
        if os.path.isdir(item):
            for name in sorted(os.listdir(item)):
                if name.lower().endswith((".txt", ".text", ".md", ".dialogue")):
                    out.append(os.path.join(item, name))
        elif any(ch in item for ch in "*?["):
            out.extend(sorted(_glob.glob(item)))
        elif os.path.isfile(item):
            out.append(item)
    seen = set()
    unique = []
    for path in out:
        key = os.path.abspath(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _seed_background() -> WordPositionIndex:
    """The null hypothesis used for signature mining when no background
    file was given. Built once, lazily, from the same neutral seed lines --
    small and generic, so signatures lean on frequency rather than on real
    contrast. Passing a real background corpus makes this much sharper."""
    global _SEED_BG
    if _SEED_BG is None:
        _SEED_BG = WordPositionIndex(max_vocab=8_000, max_pairs=40_000,
                                     max_transitions=40_000)
        _SEED_BG.add_text(SEED_CORPUS, source="<seed-background>")
        _SEED_BG.close()
    return _SEED_BG


_SEED_BG: Optional[WordPositionIndex] = None


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    return str(value)
