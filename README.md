# EmptyTalkerRobot

**A lightweight, offline speech engine for games and robotics — built from hashed statistics, not an LLM.**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/dependency-NumPy-orange.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Offline](https://img.shields.io/badge/network-offline-success.svg)](#offline-by-design)
[![LLM](https://img.shields.io/badge/LLM-none-lightgrey.svg)](#what-it-is-not)

**EmptyTalkerRobot** is a single-file Python speech system built on top of the
[`EmptyMicroRobot`](https://github.com/MaybeRickSanchez/EmptyMicroRobot) decision core.

It constructs utterances from **hashed word-position statistics**, measures a
speaking persona from text, generates multiple candidate lines, lets the
decision core select between them, and uses feedback to change future behavior.

There is:

* no pretrained language model,
* no neural network,
* no model tokenizer,
* no model weights,
* no inference server,
* no cloud API,
* and no network connection required at runtime.

> **If the data is not there, the robot has nothing to say — and it says so.**

---

## Table of Contents

* [Why EmptyTalkerRobot?](#why-emptytalkerrobot)
* [What It Is](#what-it-is)
* [What It Is Not](#what-it-is-not)
* [Architecture](#architecture)
* [How It Works](#how-it-works)

  * [`WordPositionIndex`](#1-wordpositionindex)
  * [`Persona`](#2-persona)
  * [`SpeechPlanner`](#3-speechplanner)
  * [`FeedbackJudge`](#4-feedbackjudge)
* [Installation](#installation)
* [Quick Start](#quick-start)
* [Core API](#core-api)
* [Relations](#relations)
* [Game Integration](#game-integration)
* [Persistence](#persistence)
* [Performance](#performance)
* [Memory](#memory)
* [Offline by Design](#offline-by-design)
* [Limitations](#limitations)
* [When to Use It](#when-to-use-it)
* [When Not to Use It](#when-not-to-use-it)
* [Development](#development)
* [Contributing](#contributing)
* [License](#license)

---

## Why EmptyTalkerRobot?

Modern language models solve a very different problem.

If you need a general-purpose conversational system with broad world knowledge,
semantic reasoning, long-form generation, and instruction following, an LLM is
the appropriate class of technology.

EmptyTalkerRobot deliberately targets a much smaller problem:

> **Give an embodied agent a local, measurable, adaptable voice without shipping a language model.**

This makes it interesting for:

* game NPCs,
* procedural characters,
* small robots,
* simulations,
* offline applications,
* privacy-sensitive environments,
* low-resource deployments,
* and experiments in non-neural language generation.

The goal is not to reproduce an LLM.

The goal is to explore what a useful speaking agent can do with:

**statistics + bounded memory + a decision core + feedback.**

---

# What It Is

EmptyTalkerRobot is a constructive speech system.

Instead of predicting tokens with a neural network, it builds sentences by walking
a compact hashed positional transition structure extracted from text.

The system combines four major components:

| Component           | Purpose                                                         |
| ------------------- | --------------------------------------------------------------- |
| `WordPositionIndex` | Stores hashed word-position transition statistics               |
| `Persona`           | Measures the speaking characteristics of a corpus               |
| `SpeechPlanner`     | Plans length, seeds, transitions, and candidates                |
| `FeedbackJudge`     | Scores generated utterances and feeds reward back into the core |

The final candidate is selected through the `EmptyMicroRobot` decision mechanism.

---

# What It Is Not

EmptyTalkerRobot is intentionally **not**:

* an LLM,
* a chatbot,
* a neural language model,
* a semantic parser,
* a general-purpose language understanding system,
* a replacement for GPT, Claude, Gemini, or similar systems,
* a question-answering engine,
* a knowledge base,
* or a general prose-generation system.

It does not claim to understand what its words mean.

Its linguistic behavior comes from statistical structure present in its source material.

---

# Architecture

```text
                         ┌──────────────────┐
                         │     Text Files    │
                         └────────┬─────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │    WordPositionIndex    │
                    │ hashed positional data  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │        Persona          │
                    │ formality / valence /   │
                    │ aggression / etc.       │
                    └────────────┬────────────┘
                                 │
                                 ▼
User / Game Event ───────► ┌──────────────────┐
                           │  SpeechPlanner   │
                           └────────┬─────────┘
                                    │
                              candidates
                                    │
                                    ▼
                           ┌──────────────────┐
                           │ EmptyMicroRobot  │
                           │ decision core    │
                           └────────┬─────────┘
                                    │
                                    ▼
                               Utterance
                                    │
                                    ▼
                           ┌──────────────────┐
                           │  FeedbackJudge   │
                           └────────┬─────────┘
                                    │
                              reward / punish
                                    │
                                    ▼
                           ┌──────────────────┐
                           │ Learned feedback │
                           └──────────────────┘
```

The important distinction is that the **language layer does not replace the
decision core**.

The decision core knows about features, traces, votes, rewards, and candidate
selection.

EmptyTalkerRobot supplies the language-specific structures needed to turn those
decisions into speech.

---

# How It Works

## 1. `WordPositionIndex`

`WordPositionIndex` converts text into a compact hashed representation.

The pipeline is approximately:

```text
text
  ↓
tokenization
  ↓
word positions
  ↓
64-bit hashing
  ↓
sorted positional columns
  ↓
transition statistics
```

Conceptually, the resulting structure records relationships such as:

```text
word A at position i
        │
        └──► word B
              count = N
```

The store does not need to retain a Python object for every transition.

The original text is not kept as part of the transition representation.

### Why positional information?

A simple unordered word-frequency table loses too much structure.

Position-aware transitions retain information about **where words tend to occur
relative to one another**, which provides the basic material required to construct
short utterances.

---

## 2. `Persona`

`Persona` measures how a character tends to speak.

Given a dialogue corpus, it can estimate dimensions such as:

* formality,
* aggression,
* valence,
* verbosity,
* certainty,
* lexical richness.

It can also identify phrases that are unusually common relative to a background
corpus and retain dialogue lines as quotable expressions.

Example:

```python
talker.build_persona(
    "Thanos",
    dialogue_paths=["thanos.txt"]
)
```

There is no `Thanos`-specific implementation.

The personality comes from the supplied data.

### Important

Persona measurements are intentionally lightweight.

They are statistical heuristics based partly on hand-written lexical lists,
not a psychological model.

---

## 3. `SpeechPlanner`

`SpeechPlanner` decides how an utterance should be constructed.

For each call it can determine:

* target sentence length,
* seed words,
* candidate starting points,
* transition paths,
* generation width,
* and which completed candidates should be passed to the foundation.

Conceptually:

```text
input situation
      │
      ▼
relation selection
      │
      ▼
length planning
      │
      ▼
seed selection
      │
      ▼
transition walking
      │
      ▼
candidate utterances
      │
      ▼
decision core
```

The output is therefore **constructed**, not retrieved from a neural latent space.

---

## 4. `FeedbackJudge`

Feedback is part of the design rather than an external afterthought.

The judge combines two signals.

### Intrinsic score

The intrinsic evaluator can consider:

* required words,
* target length,
* repetition,
* novelty,
* persona fit,
* and other structural properties.

### Learned score

A second `EmptyRobot` instance learns from historical `good` / `bad`
feedback attached to feature profiles of previous utterances.

The signals are combined into a signed reward.

That reward is pushed back into the foundation:

```text
utterance
   │
   ├──► intrinsic evaluation
   │
   └──► learned evaluation
             │
             ▼
          combined reward
             │
             ▼
       EmptyMicroRobot
             │
             ▼
     future behavior changes
```

This feedback also influences the robot's `creativity` state.

---

# Installation

## Requirements

* Python 3.9+
* NumPy
* `EmptyMicroRobot`

Install NumPy:

```bash
pip install numpy
```

Clone the foundation:

```bash
git clone https://github.com/MaybeRickSanchez/EmptyMicroRobot
```

Make sure `empty.py` is importable:

```python
from empty import EmptyRobot
```

Then place `emptytalker.py` somewhere on your Python path.

---

# Quick Start

```python
from emptytalker import EmptyTalkerRobot

talker = EmptyTalkerRobot()

# Load source material
talker.ingest_txt("thanos.txt")

# Build a speaking persona
talker.build_persona(
    "Thanos",
    dialogue_paths=["thanos.txt"]
)

# Generate speech
utterance = talker.speak(
    "hello there, what do you want?"
)

print(utterance.text)
print(utterance.confidence)

# Provide feedback
utterance.reward()
# or:
# utterance.punish()

# Inspect learning state
print(
    talker.stats()["feedback"]["creativity"]
)

# Explain the decision
print(talker.explain(utterance))

# Print a compact report
print(talker.report())
```

---

# Minimal Example

The smallest possible setup is:

```python
from emptytalker import EmptyTalkerRobot

robot = EmptyTalkerRobot()

robot.ingest_txt("dialogue.txt")

response = robot.speak("hello")

print(response.text)
```

A robot can technically start with zero files:

```python
robot = EmptyTalkerRobot()
```

but without source material there is little or no linguistic structure from which
to construct useful speech.

---

# Core API

| Method                       | Description                                         |
| ---------------------------- | --------------------------------------------------- |
| `speak(text)`                | Generate an utterance for an input situation.       |
| `respond(text)`              | Alias for `speak()`.                                |
| `voice(event)`               | Convert a game/event dictionary into an utterance.  |
| `ingest_txt(path)`           | Add one or more `.txt` files to the hashed store.   |
| `build_persona(name, ...)`   | Measure and install a speaking persona.             |
| `add_relation(id, spec)`     | Register an input → output relationship.            |
| `recall(query)`              | Recall previous speech associated with a situation. |
| `decide(situation, options)` | Directly access the foundation's decision system.   |
| `save(path)`                 | Persist the complete robot state.                   |
| `load(path)`                 | Restore a saved robot.                              |
| `stats()`                    | Return detailed runtime statistics.                 |
| `report()`                   | Produce a compact human-readable report.            |
| `explain(utterance)`         | Explain how an utterance was produced.              |

---

# Game Integration

The `voice()` method is intended as a convenient bridge between game logic and
the speech system.

For example:

```python
line = robot.voice({
    "event": "player_entered_room",
    "danger": True,
    "distance": 4
})

print(line.text)
```

A game can therefore provide an event description rather than manually building
a natural-language prompt for every situation.

Relations determine how game situations map onto speech behavior.

---

# Relations

Relations connect probable inputs to probable outputs.

For example:

```python
robot.add_relation(
    "warning",
    {
        "input": ["danger", "enemy", "near"],
        "output": ["warning", "threat"],
    }
)
```

Relations can be registered at runtime and persisted as part of the robot's
configuration/state.

This makes it possible to reuse the same speech engine across multiple:

* characters,
* games,
* robots,
* environments,
* and interaction models.

---

# Persistence

The entire robot can be saved to a JSON file:

```python
robot.save("character.json")
```

and restored later:

```python
robot = EmptyTalkerRobot.load("character.json")
```

The persisted state can include:

* persona measurements,
* hashed word-position data,
* relations,
* experience history,
* feedback history,
* learned weights,
* and other state required to resume operation.

The JSON file therefore acts as both a configuration format and a saved robot
state.

---

# Performance

Measured on a single laptop using:

* NumPy,
* a ~1,400-token corpus,
* a persona fitted from that corpus,
* and a warmed foundation containing approximately 700 traces.

### Latency

| Operation            |     p50 |     p95 |
| -------------------- | ------: | ------: |
| `speak()`            |  6.2 ms |  9.6 ms |
| `speak() + reward()` | 11.3 ms | 15.6 ms |

### Candidate width

| Width |   Time |
| ----: | -----: |
|     2 | 3.8 ms |
|     3 | 4.8 ms |
|     5 | 6.7 ms |
|     8 | 9.5 ms |

### Ingestion

```text
~55,000 tokens/s
```

These numbers are workload-specific and should not be interpreted as universal
benchmarks.

The intended workload is **turn-based or event-driven dialogue**, such as an NPC
speaking after a player action.

It is not designed to generate a fresh utterance every rendered frame.

---

# Memory

The hashed representation is designed to benefit from repeated material.

Measured with a 12 KiB English prose sample:

```text
Single copy:      ~38.7 bytes/token
Same text ×30:     ~1.3 bytes/token
```

The underlying store is approximately:

```text
O(distinct pairs)
```

rather than:

```text
O(total tokens)
```

As a result, repeated transitions can become substantially cheaper to represent.

Actual memory usage depends on corpus vocabulary, transition diversity, hashing,
and implementation details.

---

# Offline by Design

EmptyTalkerRobot does not require:

* an API key,
* a cloud service,
* an inference server,
* a network connection,
* a GPU,
* model weights,
* or remote inference.

After installation and data ingestion, the speech pipeline can run locally.

This can be useful for:

* offline games,
* embedded experiments,
* private datasets,
* disconnected environments,
* local simulations,
* and systems where external inference is undesirable.

---

# Determinism and Data

The system is fundamentally data-driven.

If a concept, phrase, or transition does not exist in the supplied material, the
system has no neural model from which to invent the missing knowledge.

Depending on its thresholds and available statistics, it may decline to generate
a response or construct something from related transitions.

This property is intentional.

It makes the system much more constrained than a general language model, but also
makes its source data easier to inspect.

---

# Limitations

## No Semantic Understanding

EmptyTalkerRobot does not construct a general semantic representation of language.

It manipulates statistical relationships between hashed tokens and positions.

A fluent output should therefore not be interpreted as evidence of comprehension.

---

## Constructed Sentences Can Drift

Sentences are assembled from observed transitions.

Short utterances may remain close to the structure of the source corpus, while
longer generations can become:

* repetitive,
* awkward,
* grammatically inconsistent,
* or semantically incoherent.

---

## Persona Is Heuristic

Persona dimensions are computed from counts and lexical heuristics.

They are not:

* psychological measurements,
* validated personality tests,
* or scientific models of human personality.

They are simply useful control signals for a procedural speech system.

---

## Feedback Quality Matters

The learned good/bad judge depends on the quality of the feedback it receives.

Inconsistent feedback can produce inconsistent learned preferences.

With little feedback, behavior naturally relies more heavily on intrinsic scoring.

---

## Confidence Is Not Probability

The confidence value comes from the foundation's heuristic composition.

It should not automatically be interpreted as:

```text
P(correct)
```

or any other calibrated probability.

If probabilistic interpretation matters for your application, inspect the
foundation's calibration report:

```python
calibration_report()
```

---

## Hash Collisions

The transition store uses 64-bit hashes.

Hash collisions are therefore possible.

A collision can cause distinct words to share a representation.

This is a deliberate trade-off between compactness, speed, and representation
fidelity.

---

# When to Use EmptyTalkerRobot

EmptyTalkerRobot is particularly suited to projects where you want:

* procedural NPC speech,
* character-specific dialogue behavior,
* local/offline generation,
* small CPU footprints,
* fast response times,
* feedback-driven adaptation,
* inspectable generation,
* bounded data structures,
* or experimentation with non-neural language systems.

---

# When Not to Use It

A modern language model is generally the more appropriate technology when the
application requires:

* broad factual knowledge,
* semantic reasoning,
* reliable question answering,
* long-form coherent writing,
* complex instruction following,
* robust multi-turn dialogue,
* translation,
* summarization,
* or general-purpose natural-language understanding.

EmptyTalkerRobot is intentionally not designed for these goals.

---

# Example: Character Pipeline

A complete character setup can look like this:

```python
from emptytalker import EmptyTalkerRobot

robot = EmptyTalkerRobot()

# Source material
robot.ingest_txt("character_dialogue.txt")

# Character voice
robot.build_persona(
    "Character",
    dialogue_paths=["character_dialogue.txt"]
)

# Game situations
robot.add_relation("greeting", greeting_relation)
robot.add_relation("combat", combat_relation)
robot.add_relation("warning", warning_relation)

# Runtime speech
utterance = robot.voice({
    "event": "combat_started"
})

print(utterance.text)

# Feedback
if player_reaction_was_good:
    utterance.reward()
else:
    utterance.punish()

# Persist state
robot.save("character.json")
```

On the next run:

```python
robot = EmptyTalkerRobot.load("character.json")
```

The robot can continue from its saved state.

---

# Design Philosophy

EmptyTalkerRobot intentionally explores a different design space from neural
language models.

The priorities are:

```text
bounded memory
      +
simple statistics
      +
local execution
      +
fast decisions
      +
measurable persona
      +
feedback
      =
procedural speech
```

The resulting system is small enough to inspect and experiment with while still
providing enough structure for an embodied agent to produce characterful speech.

It is not intended to reproduce the capabilities of large neural models.

It is intended to answer a different question:

> **How much useful speech behavior can a small offline agent obtain from
> statistics, memory, planning, and feedback?**

---

# Development

Clone the repository and create a local environment:

```bash
git clone <your-repository-url>
cd EmptyTalkerRobot

python -m venv .venv
```

Activate the environment.

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install numpy
```

Then make sure the `EmptyMicroRobot` foundation is importable:

```python
from empty import EmptyRobot
```

---

# Suggested Project Layout

A typical repository can be organized as:

```text
EmptyTalkerRobot/
├── emptytalker.py
├── README.md
├── LICENSE
├── requirements.txt
├── examples/
│   ├── basic.py
│   └── npc.py
├── data/
│   └── example_dialogue.txt
└── tests/
    └── ...
```

A minimal `requirements.txt`:

```text
numpy
```

---

# Contributing

Contributions are welcome.

Useful areas for experimentation include:

* more compact indexing strategies,
* better candidate planning,
* improved persona measurements,
* alternative feedback signals,
* collision mitigation,
* benchmark tooling,
* memory optimization,
* deterministic generation modes,
* additional game integrations,
* and better diagnostics.

Before submitting a large change, it is useful to open an issue describing the
design and intended behavior.

When contributing, please keep the project's core constraint in mind:

> **EmptyTalkerRobot is intentionally not an LLM.**

Features that require neural model inference should generally remain outside the
core design.

---

# License

MIT License.

See [`LICENSE`](LICENSE) for the complete license text.

---

# Credits

EmptyTalkerRobot is built on the decision core provided by
[`EmptyMicroRobot`](https://github.com/MaybeRickSanchez/EmptyMicroRobot).

---

# Final Summary

**EmptyTalkerRobot is a small, offline, constructive speech engine for games and robotics.**

It:

* learns statistical word-position transitions from text,
* measures a speaking persona,
* plans sentence construction,
* generates multiple candidates,
* lets a decision core choose between them,
* learns from reward and punishment,
* persists its state,
* and runs locally without an LLM.

It does **not** understand language.

It does **not** contain an LLM.

It does **not** attempt to replace general-purpose conversational AI.

It is a deliberately constrained tool for giving small agents a procedural,
measurable, and adaptable voice.
