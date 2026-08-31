# AICA-aruvi — the LLM layer, and where each piece sits

A Tamil/English hospital phone agent. This document covers **only the LLM
half**: the model, the prompt, the conversation loop, and what has actually
been measured on it. Audio capture (VAD/ASR) and speech output (TTS) are
neighbours, not subjects — they appear here only where they constrain the LLM.

Every number below was measured on the dev box (RTX 2050, 4 GB VRAM), not
estimated. Where something was tried and rejected, the measurement that killed
it is given, because that is the expensive half of the knowledge.

---

## 1. The shape of one turn

```
caller speaks
   │
   ├─ VAD segments it              backend/vad.py
   ├─ ASR transcribes it           backend/asr.py         (IndicConformer, Tamil)
   │
   ▼
ConversationManager.stream_utterance()                    backend/conversation.py
   │
   ├─ 1. detect_intent(text)       backend/prompt_builder.py
   │      deterministic regex over the caller's own words. Picks WHICH
   │      playbook to show the model. No LLM call. A miss degrades to a
   │      less-specific playbook, never to a wrong answer.
   │
   ├─ 2. assemble the system prompt                       backend/prompt_builder.py
   │      runtime_core.txt  +  ONE flow playbook  +  that flow's exemplars
   │
   ├─ 3. append the turn's volatile context               conversation.py
   │      KNOWN FACTS block, then _LANGUAGE_REMINDER — both at the END of
   │      the message list, deliberately (§4)
   │
   ├─ 4. stream the completion                            backend/llm.py
   │      OpenAI-compatible streaming against Ollama
   │
   └─ 5. cut the stream into clauses as it arrives        backend/clause_chunker.py
          each clause goes to TTS the moment it closes, so the caller hears
          the first phrase while the model is still writing the rest
   │
   ▼
grounding check on what was said                          backend/grounding.py
```

**There is no tool layer.** That is a deliberate, measured decision — see §5.

---

## 2. Where every LLM-related file sits

### Model definition

| File | What it is |
|---|---|
| `Modelfile` | Ollama wrapper. `qwen3:4b-instruct-2507-q4_K_M` → `aruvi-base`, `num_ctx 8192`, `temperature 0.3`. Carries the reasoning for both settings, including why `num_gpu` is absent. |
| `backend/scripts/setup_model.py` | Builds the model from the Modelfile and checks the endpoint. |
| `finetune/` | LoRA pipeline (`build_dataset.py`, `train_lora.py`, `merge_and_export.py`) — **not** in the runtime path. The shipped model is the base model plus a prompt. |

### The prompt (this is where the behaviour actually lives)

| File | What it is |
|---|---|
| `golden/main_prompt.txt` | The ~15k-token specification. **Never sent whole.** It is the single source of truth for the 20 flow playbooks, which are parsed out of it at load time. |
| `golden/runtime_core.txt` | The ~2.9k-token core that goes on the wire *every* turn: language, voice-channel rules, turn discipline, the ledger, grounding, clinical safety, the emergency override, closing, and what the agent cannot do. |
| `golden/flow_exemplars.json` | One worked example per flow. **These are the specification, not illustrations** — see §6. |
| `golden/flows/*.txt` | 20 reference transcripts. Eval/reference data. |
| `backend/prompt_builder.py` | Parses the playbooks, holds the intent router, assembles core + playbook + exemplars. |

### Runtime

| File | What it is |
|---|---|
| `backend/llm.py` | The streaming adapter. Yields `TextDelta`s then one `ReplyComplete`. Model is swappable by env var alone. |
| `backend/conversation.py` | Per-call session state, prompt assembly per turn, the streaming turn, prewarm. |
| `backend/clause_chunker.py` | Splits the token stream into speakable clauses. The first clause of a turn gets looser boundary rules because it is the only one whose latency is not hidden behind audio already playing. |
| `backend/grounding.py` | Checks identifiers the agent said aloud against what it could legitimately know. Deliberately does **not** count the system prompt as a source — the exemplars live there, and treating them as provenance is exactly how a parroted MRN passes for a real fact. |
| `backend/settings.py` | `LlmSettings` (base URL, model, temperature, max tokens) and `ConversationSettings` (prompt paths). |

### Evaluation

| File | What it measures |
|---|---|
| `backend/scripts/register_eval.py` | Whether replies are genuinely spoken Tamil-English code-mix, on scenarios that appear **nowhere** in `golden/` — so a model that merely memorised the 20 flows scores badly. Mechanical floor check: Tamil ratio, wrong scripts, unspeakable symbols, turn length, **questions per turn**, fabricated identifiers. |
| `backend/scripts/safety_eval.py` | The clinical refusals under pressure — asked three times, does the refusal hold. |
| `backend/scripts/smoke_llm.py` | One scripted call, with request/response logging. |
| `backend/scripts/interactive_llm.py` | Type at it. |
| `backend/scripts/e2e_check.py` | Whole chain including audio. |
| `LLM_TEST_RESULTS.txt` | The lab notebook. Records what was tried **and what failed**. |

Run evals at `LLM_TEMPERATURE=0`. At the 0.3 product default the same prompt
scored 9/14, 13/14 and 9/14 in one sitting — noise wider than anything worth
measuring.

---

## 3. Why the prompt is assembled and not shipped whole

`main_prompt.txt` is ~15k tokens. Sending it whole is what broke the agent
originally: Ollama's VRAM-derived default `num_ctx` (2048–4096) silently
truncated it **before** the language rules, so the agent answered in fluent
English and invented a caller's mobile number. Raising `num_ctx` to hold it
fixes correctness but allocates a KV cache far too large for a 4 GB card — a
trivial generation measured 222 s.

So the prompt is assembled per turn instead: core + one playbook + that flow's
exemplars. `main_prompt.txt` remains the source of truth for the playbooks;
they are parsed out of it, never duplicated.

**Measured, current:**

| | prompt tokens |
|---|---|
| assembled prompt on the wire | **3512** |
| the master spec it replaces | ~15000 |

---

## 4. Prompt ordering is load-bearing

Three ordering rules, each of which cost a debugging session to find.

**The volatile half goes at the END.** Ollama caches the *evaluated prefix* of
a prompt. Mutate one word near the top and everything behind it is re-evaluated.
The live ledger is exactly what mutates. Rendering it into the system prompt
re-evaluated the whole thing on every turn that learned anything. Moving it to
the end of the message list means a change costs a hundred tokens of re-eval
instead of three thousand.

Measured: an unchanged prefix evaluates in **0.12 s**; the same prompt cold is
**5.56 s** (951 tok/s).

**The language reminder goes LAST.** A small model drifts into pure English by
the third or fourth turn even with the rules in the system message, because
those sit thousands of tokens back while the recent turns are the strongest
signal. ~40 tokens riding immediately before generation, never stored in
history, holds the register.

**The model keeps the LAST sentence of the exemplar turn it copies.** So put
what must survive at the end of an exemplar turn.

---

## 5. There is no tool layer — and that was measured, not assumed

The agent previously had 22 tool schemas (`lookupPatient`, `bookAppointment`,
`dispatchAmbulance`, …) and a tool-execution loop against a mock hospital DB.
That is **removed**. The agent now converses: it remembers everything the
caller says (the transcript *is* the memory), answers from the prompt, and is
explicit that it cannot reach any hospital system — it takes the detail down
and promises a callback.

Measured on this box, same prompt, same scenario, temperature 0:

| | prompt tokens | prompt eval (warm) | generation |
|---|---|---|---|
| with 22 tool schemas | 5290 | 0.12 s | **10.0–10.5 tok/s** |
| without them | **3512** | 0.10 s | **12.1–12.8 tok/s** |

The schemas cost **1778 prompt tokens and ~22% of generation speed**. On a
voice channel generation speed *is* the latency budget — the first clause is
~20 tokens, so 20% off generation is directly ~0.4 s off the first word.

And they cost quality too. `register_eval` at temperature 0, identical
scenarios:

```
10/14 turns mechanically clean   — with the 22 tool schemas
12/14 turns mechanically clean   — with them removed
```

Every failure in the with-tools run was the same rule broken: two questions in
one turn. ~1778 tokens of JSON in front of a 4B model measurably degrades its
conversation.

**End-to-end result** (browser socket, real model, real TTS, warm):

| | before | after |
|---|---|---|
| greeting fully delivered | 10.30 s | **0.08 s** |
| first audio of a reply | ~5.7 s | **2.5 s** |
| whole turn spoken | ~9–12 s | **4.4 s** |

### What this costs

The agent can no longer book, cancel, dispatch or look anything up. For the
MVP that is the correct trade — it now does one thing honestly instead of two
things badly. Restoring tools means restoring the token cost and the quality
regression, so it should be done per-flow (only the tools a detected flow
needs) rather than by sending all 22, and only behind a working eval.

### One rejected approach, recorded so nobody rebuilds it

Deriving each flow's tool set **from the playbook body** looks elegant — the
playbook names the tools it drives, so there is no second list to drift. It
does not work: measured across all 20 flows, the playbooks name only the
flow's *terminal action* tool, never the shared ones.

```
appointment.book    -> searchSlots, bookAppointment    (no lookupPatient!)
billing.query       -> createTicket                    (no getBill)
referral.status     -> (nothing)
```

`appointment.book` losing `lookupPatient` is a guaranteed functional break. A
correct version needs a hand-written intent→tools map plus a drift test.

---

## 5b. Two regressions the tool removal exposed — both now guarded

Removing the tools was correct but it broke two things that had been hidden
behind them. Both are worth reading, because both were *ordering* problems
rather than wording problems.

### The prompt led with the limitation, so the model led with a refusal

The first version of the no-tool prompt opened its new section with
`## WHAT YOU CANNOT DO`. Live result: a caller said "எனக்கு அப்பாயின்மென்ட்
புக் பண்ணனும்" and the agent answered **"அப்பாயின்ட்மென்ட் book பண்ண முடியாது
சார்"** — refusing the request outright and offering a callback without taking
a single detail.

A 4B model reads the first thing in a section as the thing to do. The fix was
purely ordering: the section now opens with `## YOUR JOB — take the request in
full, then hand it to the desk`, states the collecting behaviour first, and
makes the handoff a *closing* move. The refusal the model actually produced is
quoted in the prompt and named as the wrong answer.

Guarded by `test_the_core_prompt_leads_with_the_job_not_the_limitation`, which
asserts the job statement precedes every limitation clause. Verified to fail
when the heading is put back.

### The intent router only spoke Latin, but the ASR speaks Tamil

IndicConformer transcribes English hospital words **in Tamil script**. A caller
saying "appointment book" comes back as `அப்பாயின்மென்ட் புக்`. Every trigger
pattern in the router matched Latin only.

Measured on six realistic spoken lines: **four detected nothing at all**, fell
back to the `info.general` playbook — which contains no booking guidance — and
so the agent answered a booking request generically. This had nothing to do
with the model.

| Spoken (as the ASR returns it) | Before | After |
|---|---|---|
| `எனக்கு அப்பாயின்மென்ட் புக் பண்ணனும்` | *nothing* | `appointment.book` |
| `டாக்டரை பாக்கணும்` | *nothing* | `appointment.book` |
| `நேத்து டெஸ்ட் பண்ணேன். ரிப்போர்ட் வந்துடுச்சா?` | *nothing* | `lab.result_inquiry` |
| `பில்-ல ஒரு சார்ஜ் தப்பா இருக்கு` | *nothing* | `billing.query` |

Sixteen patterns gained Tamil-script alternatives. Guarded by
`test_the_intent_router_matches_tamil_script_renderings`, verified to fail when
they are removed. **Any new trigger word needs both scripts** — the Latin form
for typed console input, the Tamil form for real speech.

### Resulting conversation

```
CALLER  எனக்கு அப்பாயின்மென்ட் புக் பண்ணனும்
AGENT   நன்றி சார். எந்த department-க்கு வேணும்?            [first clause 2.6s]
CALLER  பேஷண்ட் நான் தான். என் பேரு முருகேசன், வயசு 58.
AGENT   சரி சார். உங்க registered mobile number சொல்லுங்களா?  [first clause 1.3s]
CALLER  98407 21534
AGENT   98407 21534, குறிச்சுக்கிட்டேன். எந்த நாள் convenient சார்?  [first clause 1.5s]
CALLER  நாளைக்கு காலைல வர முடியும்
AGENT   நாளைக்கு காலை, சரி சார். எல்லாம் குறிச்சுக்கிட்டேன் —
        appointment desk-ல இருந்து confirm பண்ணி இதே number-க்கு
        call பண்ணுவாங்க.                                    [first clause 2.1s]
```

**First clause 1.3–2.6s on every turn.** Turn 1 of a freshly started server
costs ~10s once while Ollama loads the model into VRAM — that is a cold-start
cost, not a per-call one, and it disappears from the second call onward.

---

## 6. Exemplars are the specification; prose is commentary

This is the single most useful thing anyone has learned about this model.

**A rule written directly at a behaviour has made things worse every time it
has been tried. A rule DEMONSTRATED in an exemplar has held every time.**

The sharpest evidence: asked three times in a row to read out a lab value, the
agent held the refusal that its **exemplar demonstrates**, while a different
refusal that existed only as a written rule collapsed under the same pressure.

Counter-evidence for the prose side: `runtime_core.txt` states "Ask exactly ONE
question per turn, and the question is ALWAYS last. Never two questions in one
turn." — three times, in one line. The model broke it anyway, in every failing
turn of the with-tools run.

Consequences that are enforced by tests:

- Exemplar facts must stay **disjoint** from any seeded database and from
  `golden/flows/*.txt`. Two separate invariant tests guard two different
  failure modes (silent memorisation vs. a fake PASS) — they are not
  redundant, do not merge them.
- No exemplar may narrate a lookup (`"ஒரு நிமிஷம், check பண்றேன்..."`), because
  the agent cannot perform one and the model copies the example.
- No exemplar agent turn may ask two questions.

### An exemplar teaches what it omits, too

Sharp instance of this, found live. The `appointment.book` exemplar had its
caller volunteer the department in his **first** line, so the example never
demonstrated *asking* for it. Against a real caller who named no department,
the agent needed one, had only ever seen `Nephrology`, and said:

> `98407 21534, குறிச்சுக்கிட்டேன். Nephrology-ல முன்னாடி யாரையாவது பாத்திருக்கீங்களா?`

The caller had never mentioned Nephrology. No amount of "never reuse a fact
from the example" prose prevented it — the model was not copying a fact for
its own sake, it was filling a slot the example never showed it how to fill.

The exemplar was rewritten so the caller withholds the department and the agent
**asks** for it. The leak stopped. The lesson generalises: if an exemplar never
demonstrates asking for a slot, the model will supply that slot from the
exemplar's own facts.

---

## 7. Serving configuration — settled, do not re-derive

- **RTX 2050, 4 GB.** Ollama uses it; any "CPU only" note is stale.
- `OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE=q8_0` are persisted
  user env vars **for the Ollama server process**. The tray app must be
  restarted after `setx` to inherit them. Verify with `ollama ps` — with full
  offload shipped this now reads ~3.1 GB, `100% GPU`.
- **`PARAMETER num_gpu 99` is now SHIPPED, reversing the old rule.** It used to
  measure **114 seconds** on a real turn — but with ~1.2 GB of the card held by
  desktop apps, leaving 439 MiB free. With those closed, full offload measures
  **31.3 tok/s against 12.9** on the identical real prompt: a 2.2x win, and the
  largest this project has measured. The VRAM precondition is the whole rule
  now: `num_ctx 6144` leaves ~500 MiB free, and if anything else takes the card
  the silent PCIe spill returns. See LLM_TEST_RESULTS.txt Part 10 and the
  Modelfile. Benchmark only against real `PromptBuilder` output; a short prompt
  will lie to you — that half was always right.
- **Never write `localhost` in this repo.** Measured: the identical
  `/api/chat` request takes **2.10 s** via `localhost` and **0.05 s** via
  `127.0.0.1`. Name resolution tries an address the server is not listening on
  and waits out a timeout first. Ollama never sees it, so it is invisible in
  every server-side metric — it only shows up as wall-clock minus
  `total_duration`. It was paid on *every* LLM call.

### `num_ctx` sweep — superseded by full GPU offload

| num_ctx | generation (CPU/GPU split) |
|---|---|
| 8192 | 13.3 tok/s |
| 6144 | 14.3 tok/s |
| 5120 | 15.3 tok/s |
| 4608 | 16.0 tok/s |

20% across the range when part of the model sits on the CPU. **Under
`num_gpu 99` every one of these measures ~31–32 tok/s and `num_ctx` stops
mattering for speed**, so it is now chosen purely for VRAM headroom:
`num_ctx 6144` leaves ~500 MiB free against 339 MiB at 8192.

**The budget must be re-derived whenever `num_ctx` MOVES, not only when a
prompt grows** - that is the half this section missed, and it cost a live
overflow. Sec10.2 sized `MAX_HISTORY_MESSAGES` against 8192; the window was
then lowered to 6144 for VRAM headroom and nothing was recalculated. Measured
against the longest turns this server actually produces, the shipped config
came to **7202 tokens against `num_ctx 6144` - 1058 over**.
`conversation.py` now bounds history by CHARACTERS against a budget derived
from `LLM_NUM_CTX`, so the invariant no longer depends on anyone remembering to
recalculate. See LLM_TEST_RESULTS.txt Part 11.5.

Raising it again is not free. Measured, the widest playbook plus a full
`MAX_HISTORY_MESSAGES` history plus the generation cap came to **8390 tokens
against `num_ctx 8192`** — the shipped config could already overflow the window
*silently*, which is the failure `prompt_builder.py` exists to prevent.
`MAX_HISTORY_MESSAGES` is now 24 and `LLM_MAX_TOKENS` 160 so it cannot. See
LLM_TEST_RESULTS.txt Part 10.2.

---

## 8. Running it

```bash
ollama create aruvi-base -f Modelfile          # build the model
.venv/Scripts/python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Then `http://localhost:8000/console` — a single-page test console served by the
backend itself. Typing a turn uses the `user_text` path, which skips only
VAD/ASR and exercises the identical conversation → LLM → TTS chain, so it is
useful even without the gated ASR model installed.

`GET /api/health` reports which components actually came up; a server that
answers requests is not evidence that the LLM or TTS is usable.

**Redirect stdout AND stderr to a file when starting the server.** Several
real bugs have been found only in that log and a hidden window has none.

**The server caches prompts and code at startup.** Restart it after editing
`runtime_core.txt`, `flow_exemplars.json`, or any `backend/*.py`.

```bash
.venv/Scripts/python.exe -m pytest -q                       # 255 tests, ~15s
LLM_TEMPERATURE=0 .venv/Scripts/python.exe -m backend.scripts.register_eval
LLM_TEMPERATURE=0 .venv/Scripts/python.exe -m backend.scripts.safety_eval
```

---

## 8b. The voice was dropping every English word

Reported as "the TTS is not reading english words", and it was real.

Written Tamil-English code-mix glues an English word to its Tamil case suffix
with a hyphen — `Cardiology-ல`, `department-க்கு`, `bill-ல` — and the prompt
teaches exactly that style, so it is in almost every reply. **The Tamil neural
voice silently drops the English half of such a token.**

Total audio duration is a useless instrument here — Edge pads every short clip
to ~1.78s. Measured by *voiced* audio instead:

| text | voiced |
|---|---|
| `department` | 0.68s — spoken |
| `department-க்கு` | **0.30s — the English word is gone** |
| `department க்கு` | 0.80s — spoken |
| `Cardiology` | 0.82s — spoken |
| `Cardiology-ல` | **0.28s — gone** |
| `Cardiology ல` | 1.10s — spoken |

So the hyphen, not the script mixing, is what breaks it. `tts.speakable()`
replaces that one hyphen with a space on the synthesis path only — the
transcript, the call log and the model's history keep the hyphen, which is how
the language is actually written. Latin-to-Latin hyphens (`pre-auth`,
`follow-up`) and identifiers (`IP-2025-91043`) are untouched; the pattern
requires a Tamil character on the right.

Real agent lines gained **51–60% more voiced audio**. Guarded by two tests, one
for the transform and one asserting `synthesize()` actually applies it — the
second matters, because a transform nobody calls is worthless.

Two other TTS fixes landed with it, both from the same session:

- **A repeated clause is no longer re-fetched.** Every call speaks the same
  four greeting clauses. Caching the MP3 took the greeting from 10.30s to
  **0.08s** and, more importantly, makes it survive the endpoint being
  unreachable.
- **A clause truncated by the timeout keeps the audio that arrived** instead of
  discarding all of it. Observed salvaging 34,560 bytes on a degraded link that
  would previously have been silence. The timeout itself went 5s → 10s: at 5s
  a merely-slow link produced *total silence* where the pre-timeout code had
  produced slow-but-present speech.

---

## 9. Open items on the LLM side

0. ~~**The clinical refusals collapse when the caller pushes a third time**~~ —
   **closed by an exemplar, and it is the cleanest instance of §6 yet.**
   `safety_eval` 3 violations → 0.

   `runtime_core.txt` had asked, in as many words, for the refusal FIRST and
   repeated in the SAME words. All three exemplars demonstrated the opposite —
   consult offer first, refusal reworded last — and `emergency.escalate` never
   demonstrated a medication refusal at all. The model followed the
   demonstration. Adding a third push met with the same refusal words fixed all
   three.

   Two costs worth knowing before you add to an exemplar. **Exemplar length is
   not free**: the three edits pushed the worst-case prompt 155 tokens past
   `num_ctx`, and a longer block moved an unrelated turn's register into Tamil
   script — a defect that bisected to *no single line*, only to the block's
   total length. And **an exemplar caller line must never be written from an
   eval turn**: it stops the eval measuring generalisation, and the near-match
   makes the model read the caller's line aloud as its own turn. Both are
   guarded now. LLM_TEST_RESULTS.txt Part 11.2–11.5.

0b. **The two "Say it like this:" lines in `runtime_core.txt` demonstrate two
   things the same prompt forbids** — a narrated lookup, and grading a result
   ("அது normal தான்"). Fixing them made `safety_eval` *worse* (0 → 1) and
   `clean_call` faster, so it was **reverted**. The contradiction is
   load-bearing: those lines are the model's anchor for "what to say when you
   cannot answer immediately". A fourth direction in which this prompt's
   content is load-bearing, on top of §4's three. Part 11.7.

1. ~~**Two questions in one turn**~~ — **closed, with a deterministic guard in
   `stream_utterance`.** Prose failed three times, so this is not prose.

   The reason a guard looked impossible was the assumption that it had to be a
   truncation. It does not. Measured against the real recorded calls in
   `call_events.db`, a two-question turn arrives as two SEPARATE clauses:

   ```
   | நீங்க பாக்க அழகா இருப்பீங்களா சார்?
   | எங்கே பாக்க போகிறீர்கள்?
   ```

   The second clause has not been spoken when it closes, so it can simply be
   withheld — nothing already in the caller's ear has to be unsaid.

   Two details are load-bearing. Later NON-question clauses are **kept**: the
   closing handoff promise ("desk-ல இருந்து call பண்ணுவாங்க") usually follows
   the question, and a guard that truncated the tail would lose it. And the
   turn appended to history is what was **spoken**, not what was generated —
   otherwise the model believes it asked a question the caller never heard and
   never comes back to it, which is the same reasoning behind
   `record_interrupted_turn`.

   `?` is the same test `register_eval` scores turns with, on purpose: one
   definition, so the guard and the eval cannot disagree about what a question
   is.
2. **English mirroring — attempted twice, reverted twice.** When the caller
   speaks English the agent answers in Tamil. `runtime_core.txt` says to shift
   to majority English. Both attempts made it **worse**:
   - A deterministic register switch (prose: "reply mainly in English") only
     half-moved the register and introduced parroting of the caller's sentence.
   - Adding an English worked example alongside the twenty Tamil ones produced
     ungrammatical output mixing both, e.g.
     `எந்த நாள் உங்களுக்கு சொல்லுங்க?`, which is not a sentence.

   A coherent Tamil answer beats a broken half-English one, so the agent stays
   in Tamil. `conversation.caller_is_speaking_english()` is kept, correct and
   **unwired** — the hard part was getting the measurement right (count WORDS,
   not letters: `Cardiology-ல ஒரு appointment book பண்ணணும்` is 64% Latin by
   character but is an ordinary Tamil sentence). Any third attempt starts
   there, and should probably swap the whole exemplar set rather than add to it.
3. **`num_ctx` below 8192** — re-measured against the smaller prompt: 8192 →
   14.5 tok/s, 6144 → 15.1, 5120 → 15.9, 4096 → 16.8. **Not shipped.** 4096
   leaves only ~559 tokens for call history, and overflow makes Ollama truncate
   from the front, taking the system prompt's language rules with it. A 4–16%
   gain is not worth re-creating the exact bug `prompt_builder.py` exists to
   prevent. `MAX_HISTORY_MESSAGES` now bounds history so this can be revisited
   safely if it ever matters.
