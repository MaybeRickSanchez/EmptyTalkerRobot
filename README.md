# EmptyTalkerRobot

**A speaking robot built on hashed statistics — not an LLM, not a chatbot.**

EmptyTalkerRobot is a single-file Python module that gives a robot the ability to speak. It is built on top of the `EmptyMicroRobot` decision core, imported as:

```python
from empty import EmptyRobot
```

Every sentence this robot says is **constructed** from hashed statistics derived from text files you provide, scored by the decision core, and re-weighted by the feedback it receives.

There is no pretrained encoder, no model tokenizer, no neural weights, no inference server, and no network connection.

**If the data is not there, the robot has nothing to say — and it says so.**

---

## Read This First

This is **not a chatbot**, and there is **no LLM anywhere in here**.

Most criticism this project receives — including my own first reaction — comes from evaluating it as a conversational agent. That is the wrong benchmark.

EmptyTalkerRobot is a **constructive language system for games and robotics**.

It does not generate prose from a latent space. Instead, it:

1. walks a hashed positional transition table,
2. applies a measured persona,
3. generates multiple candidate utterances,
4. lets the decision core choose between them, and
5. uses feedback to adjust its future behavior.

It is not trying to compete with GPT on open-ended conversation.

It is trying to do something different:

> Give a game NPC or small robot a speaking personality that runs offline, updates in milliseconds, and never sends data anywhere.

---

## What This Project Is NOT

* Not a chatbot
* Not an LLM
* Not a general-purpose language model
* Not a parser
* It does not understand language
* Not a replacement for GPT, Claude, Gemini, or similar conversational systems
* Not designed to write arbitrary prose
* Not designed to answer trivia
* Not designed to hold unrestricted conversations

---

## What This Project IS

* A speaking robot built on a bounded hashed word-position store
* A persona measurement engine
* A sentence-length and candidate planner
* A positional transition-table walker
* A feedback loop that learns from good/bad responses
* A system that can adapt its creativity based on feedback
* Offline and network-free
* CPU-friendly
* Based on NumPy rather than neural inference
* Suitable for games, NPCs, simulations, and small robotics projects

---

# How It Works

`EmptyMicroRobot` provides the underlying decision system.

It is a bounded associative decision memory containing feature-hashed vectors, episodic traces, voting, bounded candidate generation, an empirical world model, hard safety constraints, and `reward()` / `punish()` mechanisms.

The foundation does **not** know what a word is, what a sentence is, or how long an utterance should be.

EmptyTalkerRobot adds the language-specific layer around it.

It provides four main components:

---

## 1. `WordPositionIndex`

The `WordPositionIndex` is a hashed word-position store.

Text files are:

1. read,
2. tokenized into words,
3. associated with their positions,
4. hashed into 64-bit integers, and
5. stored in compact sorted columns.

The resulting structure represents positional transitions such as:

> after word `h` at position `i`, word `j` occurred next `n` times.

The system does not maintain Python string objects for every transition, sentence objects, or tuple-heavy structures.

The original text is not retained by the transition store.

This makes the representation compact and bounded while still preserving enough statistical structure to construct new utterances.

---

## 2. `Persona`

`Persona` measures the speaking characteristics of a corpus.

Point it at a character's dialogue and it can estimate properties such as:

* formality
* aggression
* valence
* verbosity
* certainty
* lexical richness

It also identifies phrases that are over-represented relative to a background corpus and keeps dialogue lines as quotable expressions.

For example, given a character's complete dialogue:

```python
talker.build_persona(
    "Thanos",
    dialogue_paths=["thanos.txt"]
)
```

the robot can derive a character-specific speaking profile without requiring character-specific code.

The persona is **measured from data rather than manually authored**.

---

## 3. `SpeechPlanner`

`SpeechPlanner` decides how an utterance should be constructed.

For each response it can determine:

* approximate sentence length,
* candidate seed words,
* which transitions to follow,
* how to walk the positional index,
* and which completed candidates should be passed to the decision core.

The planner can sample sentence lengths from learned distributions or use the constraints defined by a relation.

Several candidate lines can then be generated before the foundation chooses among them.

---

## 4. `FeedbackJudge`

`FeedbackJudge` evaluates whether an utterance was useful or undesirable.

It combines two independent sources of feedback.

### Intrinsic evaluation

The intrinsic judge checks properties such as:

* required words are present,
* the length is appropriate,
* repetition is controlled,
* the utterance is sufficiently novel,
* the utterance fits the persona.

### Learned evaluation

A second `EmptyRobot` instance learns from previous `good` / `bad` feedback associated with utterance feature profiles.

The two evaluations are combined into a signed reward.

That reward is then pushed back into the foundation.

The feedback loop ultimately affects the robot's `creativity` scalar and therefore influences future generation.

---

# Architecture

At a high level:

```text
                    Text Files
                        │
                        ▼
              ┌──────────────────┐
              │ WordPositionIndex │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │     Persona      │
              └────────┬─────────┘
                       │
User Input ────────────┼─────────────┐
                       ▼             │
              ┌──────────────────┐   │
              │  SpeechPlanner   │   │
              └────────┬─────────┘   │
                       │              │
                       ▼              │
                Candidate Lines      │
                       │              │
                       ▼              │
              ┌──────────────────┐   │
              │  EmptyMicroRobot │◄──┘
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
              │ Learned Feedback │
              └──────────────────┘
```

---

# Installation

EmptyTalkerRobot requires **NumPy** and the `EmptyMicroRobot` foundation.

Install NumPy:

```bash
pip install numpy
```

Clone the foundation:

```bash
git clone https://github.com/MaybeRickSanchez/EmptyMicroRobot
```

Make `empty.py` available on `sys.path`, or install it according to the foundation project's instructions.

Your environment should ultimately be able to run:

```python
from empty import EmptyRobot
```

---

# Quick Start

```python
from emptytalker import EmptyTalkerRobot

talker = EmptyTalkerRobot()

# Add source material
talker.ingest_txt("thanos.txt")

# Build a persona from dialogue
talker.build_persona(
    "Thanos",
    dialogue_paths=["thanos.txt"]
)

# Generate an utterance
u = talker.speak("hello there, what do you want?")

print(u.text)
print(u.confidence)

# Provide feedback
u.reward()
# or:
u.punish()

# Inspect the feedback state
print(talker.stats()["feedback"]["creativity"])

# Explain a specific decision
print(talker.explain(u))

# Print a compact system report
print(talker.report())

# Save the robot
talker.save("thanos.json")

# Restore it later
same = EmptyTalkerRobot.load("thanos.json")
```

The robot can also be created with no data:

```python
talker = EmptyTalkerRobot()
```

In that state it has no corpus from which to construct language.

---

# Core API

| Method                       | Description                                                            |
| ---------------------------- | ---------------------------------------------------------------------- |
| `speak(text)`                | Generate an utterance from the input situation.                        |
| `respond(text)`              | Alias for `speak()`.                                                   |
| `voice(event)`               | Game-facing shorthand for turning an event dictionary into a line.     |
| `ingest_txt(path)`           | Digest one or more `.txt` files into the hashed store.                 |
| `build_persona(name)`        | Measure and install a speaking persona.                                |
| `add_relation(id, spec)`     | Register a probable-input → probable-output relationship at runtime.   |
| `recall(query)`              | Retrieve what the robot said when a similar situation occurred before. |
| `decide(situation, options)` | Directly access the foundation's decision mechanism.                   |
| `save(path)`                 | Persist the complete robot state.                                      |
| `load(path)`                 | Restore a previously saved robot.                                      |
| `stats()`                    | Return detailed runtime statistics.                                    |
| `report()`                   | Return a compact human-readable report.                                |
| `explain(utterance)`         | Explain the decisions behind a generated utterance.                    |

---

# Using `voice()`

For game and simulation code, `voice()` provides a convenient event-oriented interface.

For example:

```python
line = talker.voice({
    "event": "player_entered_room",
    "danger": True,
    "distance": 4
})

print(line.text)
```

The exact event schema depends on the relations configured for the robot.

This allows game code to describe **situations** instead of manually constructing dialogue prompts.

---

# Relations

Relations connect probable situations to probable responses.

They can be registered at runtime:

```python
talker.add_relation(
    "warning",
    {
        "input": ["danger", "enemy", "near"],
        "output": ["warning", "threat"],
    }
)
```

The exact relation specification is defined by the implementation and can be persisted as part of the robot's JSON state.

This mechanism allows the same language system to be reused across different characters, games, and environments.

---

# The JSON Contract

A single `.json` file acts as both an input configuration and a complete persisted robot state.

On input, it can describe:

* the persona,
* relations,
* text files to digest,
* and other configuration.

After saving, it contains the learned state required to resume the robot, including:

* persona measurements,
* the packed hashed word-position index,
* experience history,
* feedback history,
* learned weights,
* and other persisted state.

Example:

```python
talker.save("thanos.json")

same = EmptyTalkerRobot.load("thanos.json")
```

The goal is for the loaded robot to continue from the saved state rather than starting from scratch.

---

# Performance

Measured on one laptop using:

* NumPy only,
* a 1,400-token corpus,
* a persona fitted from that corpus,
* and a warmed foundation containing approximately 700 traces.

Observed timings:

```text
speak()             p50  6.2 ms   p95  9.6 ms
speak() + reward()  p50 11.3 ms   p95 15.6 ms
```

Generation time by candidate width:

```text
width 2    3.8 ms
width 3    4.8 ms
width 5    6.7 ms
width 8    9.5 ms
```

Ingestion:

```text
~55,000 tokens/s
```

The ingestion pass does not require a complete index rebuild.

These measurements represent a **turn-based dialogue workload**, not a 60 Hz rendering workload.

That makes the design appropriate for things such as:

* one NPC line per player action,
* game events,
* interactive simulations,
* small robots,
* and other low-frequency conversational events.

It is not intended to generate a new sentence every rendered frame.

---

# Memory Characteristics

Memory usage was measured rather than inferred.

For one 12 KiB chunk of English prose:

```text
38.7 bytes/token
```

when stored once.

The same text repeated 30 times:

```text
1.3 bytes/token
```

because the store is based on distinct transitions rather than retaining every copy of the original text.

Conceptually:

```text
Memory ≈ O(distinct pairs)
```

rather than:

```text
Memory ≈ O(total tokens)
```

Therefore, repeated material can become substantially cheaper to represent.

---

# Limitations

EmptyTalkerRobot is intentionally constrained.

Understanding these limitations is important before using it.

## It Does Not Understand Language

The system does not build semantic representations of language.

It walks a hashed positional transition structure.

If the words required by a question never appeared in the training material, the system cannot magically acquire the missing knowledge.

Depending on the similarity and confidence thresholds, it may:

* decline to answer, or
* produce an utterance assembled from transitions that it has seen.

A fluent-looking sentence is therefore **not evidence of understanding**.

---

## Sentences Are Constructed

Sentences are assembled from transitions rather than retrieved wholesale from a language model.

Grammaticality therefore comes from the source corpus.

Short outputs may work well, while longer outputs can drift, repeat, or become structurally awkward.

---

## Persona Is Measured, Not Authored

The persona dimensions are computed from counts and hand-written lexical lists.

The measurements are intentionally simple.

They are:

* crude,
* inspectable,
* documented,
* and overridable through configuration.

They should not be interpreted as a scientifically validated psychological model.

---

## Feedback Quality Matters

The learned good/bad judge can only learn from the feedback it receives.

If you provide poor or inconsistent feedback, the learned judge can learn poor or inconsistent preferences.

With little or no feedback, the system falls back toward its intrinsic scoring behavior.

---

## Confidence Is Not Probability

The confidence value is based on the foundation's documented heuristic composition.

It is **not a calibrated probability**.

If you need probabilistic interpretation, inspect the foundation's:

```python
calibration_report()
```

before treating confidence values as probabilities.

---

## Hashing Is Lossy

The word-position store uses 64-bit hashes.

Hash collisions are therefore theoretically possible.

A collision can silently merge two distinct words into the same hashed representation.

This is an intentional trade-off for compactness and bounded storage.

---

# Design Philosophy

EmptyTalkerRobot deliberately chooses a different point in the design space from neural language models.

It prioritizes:

* deterministic and inspectable data structures,
* bounded memory,
* local execution,
* low latency,
* no network access,
* no pretrained model,
* incremental feedback,
* simple deployment,
* and suitability for constrained environments.

The result is not a general conversational intelligence.

It is a **small constructive speech system** that can give an embodied agent a measurable and adaptable voice.

---

# When To Use It

EmptyTalkerRobot can be useful when you need:

* NPC dialogue generated from a specific corpus
* offline game characters
* lightweight robotic speech
* procedural character personalities
* small CPU-only deployments
* privacy-sensitive local processing
* fast feedback-driven adaptation
* reproducible and inspectable generation
* a speech system without model weights or an inference server

---

# When Not To Use It

Use a modern language model instead if you need:

* broad world knowledge,
* reliable question answering,
* semantic reasoning,
* long-form coherent writing,
* robust multi-turn conversation,
* translation,
* summarization,
* complex instruction following,
* or general-purpose natural-language understanding.

EmptyTalkerRobot is intentionally not designed for those tasks.

---

# Example Workflow

A typical NPC pipeline might look like this:

```python
from emptytalker import EmptyTalkerRobot

robot = EmptyTalkerRobot()

# 1. Load the character's source material
robot.ingest_txt("character_dialogue.txt")

# 2. Measure the character's speaking style
robot.build_persona(
    "Character",
    dialogue_paths=["character_dialogue.txt"]
)

# 3. Configure game situations
robot.add_relation(
    "greeting",
    greeting_relation
)

robot.add_relation(
    "combat",
    combat_relation
)

# 4. Generate speech during gameplay
utterance = robot.voice({
    "event": "combat_started"
})

print(utterance.text)

# 5. Give the robot feedback
if player_reaction_was_good:
    utterance.reward()
else:
    utterance.punish()

# 6. Persist the learned state
robot.save("character.json")
```

On the next run:

```python
robot = EmptyTalkerRobot.load("character.json")
```

The robot can continue from its persisted state.

---

# Offline by Design

EmptyTalkerRobot does not require:

* an API key,
* an inference server,
* a cloud account,
* a network connection,
* GPU inference,
* or a remote model.

Once its dependencies and source data are available, the complete generation loop can run locally.

This makes it suitable for environments where network access is unavailable, undesirable, or restricted.

---

# License

MIT [`LICENSE`](LICENSE).
