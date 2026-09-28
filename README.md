# EmptyMicroRobot

**A real alternative to LLMs — for robots, not chatbots.**

EmptyMicroRobot is a tiny, self-contained decision core for robots and embedded
controllers. It is *not* a language model, *not* a conversational agent, and
*not* trying to compete with GPT, Claude, or Gemini on chat. It is trying to do
something those systems cannot: help a small robot decide what to do next, in
real time, on a microcontroller, with no GPU and no internet connection.

---

## Read this first

Most criticism this project receives — including my own first reaction — comes
from evaluating it as a **chatbot**. That is the wrong benchmark.

This project is a genuine alternative to LLMs *in robotics*, not for
conversation. In that space, it may actually be valuable. The confusion is
understandable, because the field has spent the last three years treating
"LLM" and "intelligence" as synonyms. They are not. A robot that needs to
decide when to dock, when to stop, or when to reroute does not need a
trillion-parameter text model. It needs a small, fast, bounded memory that
improves with experience and never phones home.

That is what this is.

### What this project is NOT

- Not a chatbot
- Not a language model
- Not a replacement for GPT / Claude / Gemini in conversation
- Not trying to write prose, answer trivia, or hold a dialogue
- Not claiming to be as smart as a large model

### What this project IS

- A bounded, real-time decision memory for a robot
- An LLM-free adaptive controller for edge devices
- A system that accepts *any* sensor input and learns from outcomes
- A single-file module that runs without a data center
- Something that forgets intelligently, explains its decisions, and fails safe

If you are looking for a chatbot, close this tab. If you are building a robot
and your only two options today seem to be "hard-code every rule" or "stream
sensor data to an LLM over WiFi", read on.

---

## The gap this fills

Today, if you want a robot to make non-trivial decisions, you have three
options.

1. **Hard-coded state machines.** Brittle. Every new situation is a new patch.
2. **Classical planners and RL.** Powerful, but they need a model, a simulator,
   and usually a GPU to train. Often overkill for a small robot.
3. **Call an LLM.** Flexible, but it needs network, latency, cost, and a
   privacy story you probably cannot tell.

EmptyMicroRobot is a fourth option: a tiny adaptive memory that lives on the
device, updates in milliseconds, and never sends sensor data anywhere.

It is not as smart as an LLM. It is not as principled as a planner. But it
occupies a slot neither of them can reach: **on-device, real-time, no-GPU,
no-network decision-making that improves with experience.**

---

## How it works

Every input — text, numbers, dicts, NumPy arrays, pandas DataFrames, bytes,
arbitrary Python objects — is tokenized and feature-hashed into a bounded
sketch. That sketch is stored alongside outcomes in a compact episodic memory.
Temporal decay with surprise-weighted retention makes stale convictions fade
while novel events stick around longer.

When you call `decide(state)`, the module proposes actions from memory, hand-
written rules, and an exploration budget, then applies hard safety constraints
as a veto before returning a result.

No vocabulary fitting. No offline training. No GPU. Memory stays capped no
matter how much the robot sees.

---

## Quick start

    from empty import EmptyRobot

    robot = EmptyRobot()

    # Teach it a fact (supervised)
    robot.learn("battery low", "return_to_dock")

    # Teach it an episode (state -> action -> outcome)
    robot.learn_episode(
        state="battery low, far from dock",
        action="return_to_dock",
        outcome="docked_safely",
    )

    # Let it decide
    action = robot.decide("battery low, far from dock")
    print(action)          # "return_to_dock"

    # See why
    print(robot.explain(action))

That is the whole idea in ten lines.

---

## Core API

| Method | What it does |
|---|---|
| `learn(input, label)` | Supervised teaching: bind an input to a ground-truth label. |
| `learn_episode(state, action, outcome)` | Episodic learning: record a state -> action -> outcome triple. |
| `decide(state)` | Propose an action from memory, rules, and exploration. |
| `constrain(callback)` | Register a safety veto. Fails closed on exception. |
| `explain(action)` | Human-readable and numeric breakdown of a decision. |
| `save(path)` / `load(path)` | Persist and restore memory. |

---

## Design principles

- **Bounded by construction.** Feature hashing means memory never grows with
  input diversity. A robot that runs for a year has the same footprint as one
  that runs for a day.
- **Forgets on purpose.** Recency and surprise both matter. Old certainties
  fade; rare events are kept longer.
- **Fails safe.** The `constrain()` callback is a veto, not a suggestion. If
  it raises, the action is rejected.
- **Explainable by default.** Every decision can be traced back to the
  memories, rules, and exploration that produced it.
- **Calibrated.** A reliability table maps confidence buckets to observed
  success rates, so you can measure whether the robot "knows" what it claims
  to know.

---

## Requirements

- **Required:** NumPy.
- **Optional:** scikit-learn (MiniBatchKMeans, FeatureHasher), joblib
  (compressed saves), pandas (native DataFrame tokenization). All optional
  paths have NumPy-only fallbacks.

> Save backends are not bit-compatible. A file saved with joblib and loaded
> without it will warn, because the hashers differ.

---

## When to use this

- A small robot that needs to adapt on-device, offline.
- A long-running Python process that sees a stream of heterogeneous events
  and needs a bounded, explainable memory.
- An edge device where sending data to an LLM is not an option.

## When NOT to use this

- You want a chatbot.
- You want to generate text.
- You want state-of-the-art reasoning on open-ended problems.
- You already have a GPU and a training pipeline. Use them.

---

## Contributing

Issues and pull requests are welcome. If you are about to file a "this is
worse than GPT-4" issue, please re-read the top of this README first. That is
not the comparison this project is making.

## License

MIT. See `LICENSE`.
