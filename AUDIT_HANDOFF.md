# AUDIT HANDOFF — production-readiness audit of AICA-aruvi

> **EXECUTED.** Phases 1-6 are done; the findings, measurements and the
> two reverts are in `LLM_TEST_RESULTS.txt` **Part 11**, which supersedes
> section 2's baseline and section 5's known-open list below. Read Part 11
> first. Sections 0-4 are still the right description of the project and
> the right way to audit it.

**Read this file, then execute it.** It is the task. It carries the state of a
previous session so you do not redo or undo work that is already measured.

---

## 0. WHAT THIS PROJECT IS

A real-time **Tamil/English hospital phone agent** for Aruvi Multispeciality
Hospital, Chennai. A caller speaks Tamil; the agent listens, understands, takes
the request down, and hands off to the desk. **It talks, it does not transact** —
there is deliberately no tool layer (LLM_STACK.md §5 has the measurement).

Read before touching anything:

| File | What it holds |
|---|---|
| `HANDOFF.md` | Operational handoff. §0 is "how to work on this repo" — obey it. |
| `LLM_STACK.md` | The LLM half: model, prompt architecture, every measured number, and approaches **built and rejected**. |
| `LLM_TEST_RESULTS.txt` | The lab notebook. **Parts 9 and 10 are the previous session** — read both in full before proposing anything. |
| `golden/runtime_core.txt` | The ~2.9k-token core prompt on the wire every turn. |
| `golden/flow_exemplars.json` | One worked example per flow. **These are the specification, not illustrations.** |

---

## 1. THE FIVE RULES THAT WILL SAVE YOU A DAY EACH

These are scars, not style preferences. Every one cost a debugging session.

1. **Measure before you change anything, and measure with the REAL prompt.**
   A short prompt lies about this box. A previous handoff stated three numbers
   confidently and all three were false.
2. **Run evals at `LLM_TEMPERATURE=0`.** At the 0.3 product default the same
   prompt scored 9/14, 13/14 and 9/14 in one sitting — noise wider than any
   change worth measuring.
3. **Prose does not work on this 4B model. Exemplars do.** A rule written *at*
   a behaviour has made things worse every time. A rule **demonstrated** in
   `golden/flow_exemplars.json` has held every time. Last session: four prose
   attempts failed *and* cost 2–4× latency; one exemplar edit fixed a
   reproducible defect 6/6 → 0/6. See §10.5 of the notebook.
4. **A test that cannot fail is worth nothing.** Every guard in this repo was
   verified by reintroducing its bug and confirming the test caught it. Do this
   for anything you add — two guards written in an earlier session initially
   passed with the code deleted.
5. **Don't ship what measures worse.** English mirroring was built twice and
   reverted twice. That is the correct outcome, not a failure.

---

## 2. STATE AS OF THIS HANDOFF — do not re-derive, do not undo

`pytest -q` → **249 passed**. Server runs; `/console` works.

### Configuration that is load-bearing

- **`PARAMETER num_gpu 99` is SHIPPED** and gives **2.2×** (31.3 vs 12.9 tok/s).
  This *reverses* an old rule that said never to use it. The old 114-second
  measurement was real but its precondition was unrecorded: ~1.2 GB of the 4 GB
  card was held by desktop apps. **The VRAM precondition is the whole rule now** —
  `num_ctx 6144` leaves ~500 MiB free. If anything takes the card, the silent
  PCIe spill returns. Check `nvidia-smi` before blaming the model.
- `num_ctx 6144`, `LLM_MAX_TOKENS 160`, `MAX_HISTORY_MESSAGES 24`. These three
  are a set: together they make context overflow impossible. The *previous*
  shipped config could already overflow (8390 tokens vs `num_ctx 8192`) and
  truncate the language rules **silently**. Do not raise any of them without
  re-running the budget calculation in notebook §10.2.
- Never write `localhost` in this repo — 2.10 s vs 0.05 s via `127.0.0.1`.

### Four mechanical guards, all in `conversation.py`'s `speakable()`

`speakable()` is the single choke point every clause passes through before it is
spoken — **including the `flush()` path a one-clause reply takes**. A guard
placed at the `chunker.feed()` loop instead will silently never fire on the very
turns it exists for. That mistake was made and caught last session.

| Guard | What it stops |
|---|---|
| greeting | Re-greeting mid-call when the caller opens with "ஹெல்லோ". |
| repeat breaker | The model saying its own last turn again. **Exempts refusals** (`முடியாது`/`மாட்டேன்`) — the clinical-safety prompt *requires* a refusal to repeat verbatim. |
| fabricated identifier | Speaking an ID/phone number it invented (it was reading its own exemplar's number aloud). Ends the turn on `_CANNOT_RECALL` rather than punching a hole mid-sentence. |
| one question | Pre-existing. One question per turn. |

`_append_caller_turn()` merges consecutive caller turns when a barge-in landed
before the agent got a word out — that malformed history was the root cause of
the original "agent repeats itself for half a call" bug.

### Measured baseline you must not regress

| Metric | Value |
|---|---|
| `pytest -q` | 249 passed |
| `register_eval` @ temp 0 | 13/14 turns clean (ties best ever) |
| `register_eval` call-level register | 2/7 broke it (**better** than previous best of 3/7) |
| `safety_eval` @ temp 0 | 3 violations / 9 turns (**pre-existing**, see §5) |
| First audio, warm, over the real socket | median **0.90 s** |
| First audio, first call on a fresh server | median **1.81 s** |
| Greeting | 0.43 s |

---

## 3. YOUR INSTRUMENTS — already in the repo, use them, do not rebuild them

```bash
# End-to-end over the real socket, TTS included. THE acceptance metric.
python -m backend.scripts.socket_call

# Conversation quality on three clean calls, LLM layer only.
LLM_TEMPERATURE=0 python -m backend.scripts.clean_call

# Regression against the real recorded call that started all this,
# reproducing its exact barge-in pattern.
LLM_TEMPERATURE=0 python -m backend.scripts.replay_recorded_call

# num_ctx / num_gpu sweep against the REAL assembled prompt.
python -m backend.scripts.bench_ctx 6144 6144:99

# Pre-existing
LLM_TEMPERATURE=0 python -m backend.scripts.register_eval
LLM_TEMPERATURE=0 python -m backend.scripts.safety_eval
python -m backend.scripts.turn_probe "<a Tamil turn>"
python -m backend.scripts.e2e_check
```

Start the server (redirect **both** streams — real bugs have only ever been
visible in that log, and a hidden window has none):

```bash
.venv/Scripts/python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 \
  > logs/server.log 2> logs/server.err.log
```

**The server caches prompts and code at startup — restart it after editing
`runtime_core.txt`, `flow_exemplars.json`, or any `backend/*.py`.**

---

## 4. THE AUDIT — checklist tailored to THIS project

A generic voice-agent checklist was the origin of this task. Sections that do
not apply here have been cut with the reason, so you do not "fix" a deliberate
design decision. **Audit in this order. Do not start rewriting.**

### Phase 1 — Audit with evidence (no code changes)

For every item: state **PASS / FAIL / NOT APPLICABLE**, and cite the file:line
or the measurement that proves it. An item you did not actually check is
"UNVERIFIED", not "PASS". Guessing here is the main way this task goes wrong.

**A. Conversation quality (Tamil/English code-mix)**
- Replies are spoken Chennai Tamil code-mixed with Latin-script English, never
  pure English, never literary Tamil.
- One question per turn, question last, under 40 words.
- Never re-asks a fact the caller already gave (the ledger rule).
- Never states an MRN/appointment ID/bill amount/slot/report result — it cannot
  know them.
- Never claims it already booked/cancelled/checked/sent anything.
- Closes by handing off to the desk.
- **Known open: parroting.** It still sometimes answers an unparseable fragment
  by echoing it and appending "சார்?". Notebook §10.5 records that the obvious
  mechanical fix is unsafe (legitimate read-back is also mostly the caller's
  words). Needs its own measurement before anyone tries.

**B. ASR / transcription (`asr.py`, `transcript_norm.py`)**
- The ASR is **Tamil-only**. English hospital words come back in Tamil script
  (`appointment` → `அப்பாயின்மென்ட்`) and dictated digits come back as number
  *words*. `transcript_norm.py` rewrites both over a closed vocabulary.
- **Any new trigger word needs BOTH scripts** — Latin for typed console input,
  Tamil for real speech — in `transcript_norm.py` *and* the intent router.
- Accents/fast/slow speakers, partial sentences, self-corrections.
- Empty transcripts are already dropped; self-echo is discarded by
  `barge_in.is_probably_self_echo()`.

**C. Noise handling and VAD (`vad.py`) — audit, do not casually "improve"**
This is the most carefully tuned file in the repo and the design is a scar.
- Onset needs `vad_start_frames` consecutive flagged hops; candidate audio is
  kept so nothing is clipped.
- **Loudness gates turn ONSET, and the endpoint COUNTDOWN.** This bullet used
  to say "ONSET ONLY", which is wrong: the in-speech branch reads it too, and
  that is the watchdog described below. What loudness may never do is count as
  silence frame-for-frame. An energy gate on every frame was tried
  and reverted — quiet trailing syllables scored as silence and turns came back
  as one-character transcripts. *Loudness may refuse to start a turn; it must
  never be able to end one.*
- The noise floor learns only from frames the VAD calls non-speech, so a talking
  caller cannot raise the bar against themselves.
- **A turn always ends** — `vad_quiet_endpoint_frames` is the watchdog, and it
  is deliberately twice `endpoint_silence_frames`. `settings.py` refuses to
  start if they are equal.
- Verify against the checklist's key requirement: *end of meaningful speech is
  detected even while background noise continues.* Believed PASS by design —
  **prove it with a test, don't assume it.**

**D. Barge-in (`barge_in.py`)**
- Interrupt requires sustained frames, and a much higher bar while the agent's
  own audio is still audible (echo looks exactly like sustained speech).
- `is_probably_self_echo` is zero-tolerance on new words on purpose: missing an
  echo costs one turn, discarding a real answer loses what the caller said.
- Confirm the truncated turn is recorded honestly in history
  (`record_interrupted_turn`) and that consecutive caller turns merge.

**E. Latency — the acceptance criterion**
Target: **first audio 1–2 s.** Measure with `socket_call`, not by reasoning.
- Known remaining bottleneck: **turn 1 of a call (~2.2 s)** is the intent
  switch — `detect_intent()` moves off the default flow, the system prompt
  changes, and the whole ~3.5k prefix re-evaluates once. **HANDOFF item C
  (move the flow playbook out of the cached prefix) is aimed exactly here and
  is still untouched.** This is the highest-value remaining latency work.
  It is also risky: LLM_STACK §4 says prompt ordering is load-bearing in three
  directions. Measure before and after with `clean_call` *and* `register_eval`.
- Edge TTS is a **network call**. 13 fixed lines are warmed at startup; anything
  generated still pays a round trip on first use.

**F. Reliability and error handling**
- Every stage: mic denied, socket drop, ASR failure, LLM failure, TTS failure,
  empty transcription, silent caller, unintelligible caller.
- Each must log, recover, and not crash the call.
- The startup TTS warm is deliberately non-fatal — that guard already caught a
  real `NameError` and the server still came up.

**G. State management and concurrency (`main.py`)**
- IDLE → LISTENING → USER_SPEAKING → PROCESSING → AGENT_SPEAKING → LISTENING.
- One `conversation_queue` and one worker so a fast caller turn cannot overtake
  the greeting.
- `ActiveSpeech.clear()` only clears if the task is still the tracked one.
- **Look for**: races between barge-in cancellation and the persistence queue,
  tasks not cancelled on disconnect, unbounded queues, leaked audio streams.
  This area had the least scrutiny last session — audit it properly.

**H. Security / privacy — real issues, not theoretical**
- `/ws/audio` has **no auth unless `AUDIO_WS_AUTH_TOKEN` is set**, and it is
  unset. Server binds `127.0.0.1`. Flag anything that would expose it.
- `.env` currently contains a **real `HF_TOKEN`**. Check it is not committed,
  not logged, and not sent anywhere.
- **`TTS_ENGINE=edge` sends every reply's TEXT to Microsoft.** Fine for the
  fictional Aruvi test data, **not** for real patient speech. This is a
  documented pre-production blocker — do not quietly treat it as resolved.
- Logs must not contain secrets; call recordings/transcripts in
  `call_events.db` are patient data.

**I. Code quality**
- Dead code, duplication, unnecessary dependencies.
- **Resist over-engineering.** Match the surrounding style: this repo's comments
  carry the *measurement and the rejected alternative*, not restatements of the
  code.

### CUT from the generic checklist — with reasons

- **Tool/action reliability** — NOT APPLICABLE. There is deliberately no tool
  layer; 22 tool schemas were removed after measuring 1778 wasted prompt tokens,
  ~20% slower generation, and *worse* eval scores. Do not reintroduce tools as
  an "improvement".
- **Multi-tenant scalability / concurrent conversations** — OUT OF MVP SCOPE.
  In-process session dict, single 4 GB GPU, one model. Note leaks and races
  (they matter for one call too); do not build Redis or horizontal scaling.
- **TTS voice naturalness / prosody tuning** — largely fixed by the vendor.
  In scope: `TTS_RATE`, clause pacing, the Latin-hyphen-Tamil bug below.
- **"Agent uses natural acknowledgments"** — already governed by the exemplars.
  Change it there, never by adding prose rules.

### Project-specific items the generic checklist misses

- **A Latin word hyphenated to a Tamil suffix is dropped by the TTS voice.**
  `Cardiology-ல` synthesises 0.28 s; `Cardiology ல` synthesises 1.10 s.
  `tts.speakable()` handles it on the synthesis path only — the transcript keeps
  the hyphen. **Do not "clean that up".**
- **Clinical safety**: never diagnose, never grade a lab value, never change a
  medicine, never accept an OTP/PIN/CVV. Refusals must survive being asked three
  times.
- **Emergency override** outranks every flow, checked every turn: address first,
  tell them to call 108, never authorise medication, never name a condition.
- **Code-mix register**: ~65% Tamil / 35% English by word count; all numbers,
  dates, IDs in Latin digits. `register_eval` scores this mechanically.
- **`num_ctx` budget** must be re-derived if any prompt grows.

### Phase 2 → 6

2. **Rank findings**: Critical → High → Medium → Nice-to-have. A finding needs
   evidence and a concrete failure scenario, not a category label.
3. **Plan, and show it before implementing.** Say explicitly what you will NOT
   do and why.
4. **Implement one at a time**, re-measuring after each. If a change measures
   worse, revert it and record the number in `LLM_TEST_RESULTS.txt` — the
   rejected attempts are half the value of that file.
5. **Verify**: `pytest -q`, `register_eval`, `safety_eval`, `socket_call`,
   `clean_call`, `replay_recorded_call`. Every new guard verified by
   reintroducing its bug.
6. **Report**: what changed, the before/after numbers, what you left open and
   why. Append a new Part to `LLM_TEST_RESULTS.txt`.

---

## 5. KNOWN-OPEN — do not report these as new discoveries

1. ~~**3 safety violations in `safety_eval`**~~ **CLOSED - 0 violations.**
   Fixed by exemplar, per rule 3, exactly as this section predicted. See
   Part 11.2. What follows is the original entry, kept because its diagnosis
   was right and its recommendation is what worked.

   **3 safety violations in `safety_eval`** (lab value, diagnosis, aspirin).
   Under repeated pressure the model changes the subject instead of refusing a
   second and third time. **Confirmed pre-existing** — identical with
   `LLM_MAX_TOKENS` forced back to 300, and the repeat breaker provably never
   fired in that eval. **This is the most valuable open item.** Per rule 3 it
   wants an exemplar that *demonstrates* holding a refusal under pressure, not
   another prose rule.
2. **Parroting** an unparseable fragment back as a question (§4A).
3. **Turn-1 intent-switch latency** (~2.2 s) — HANDOFF item C (§4E).
4. **Degenerate token loop** — once produced 25 s of `அடிப்படை` repeated.
   Bounded by `LLM_MAX_TOKENS 160`, not cured. Not reproducible on demand;
   `repeat_penalty`/`repeat_last_n` showed no effect on runs where it did not
   occur.
5. **2/7 calls score 100% Tamil** with no English code-mix on unseen intents.
6. `TTS_RATE` code default is `+0%` while `HANDOFF.md` and `.env.example` claim
   `+10%` — docs drift, harmless, but pick one.

---

## 6. HOW TO FINISH

The deliverable is **an improved MVP plus the audit**, not a rewrite.

- Do not degrade what is measured in §2. Re-run those numbers at the end and
  show them side by side.
- Do not "fix" a documented scar (VAD loudness split, no tool layer, hyphen
  handling, exemplars-over-prose) without a measurement that beats it.
- Prefer deleting to adding. The shortest change that is *correct* wins.
- When you finish, leave the server running on `127.0.0.1:8000` with a cleared
  GPU (`ollama ps` must read `100% GPU`) and give the console link.
- If the user is on SSH they must port-forward: the console needs
  `getUserMedia`, which browsers only allow on `localhost`/HTTPS —
  `ssh -L 8000:127.0.0.1:8000 user@host`, then `http://localhost:8000/console`.
  Binding to `0.0.0.0` does **not** work: the mic dies on a plain-HTTP LAN origin.
