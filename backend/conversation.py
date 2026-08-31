"""Conversation Manager: routes a caller transcript through the assembled
prompt and the LLM, producing the agent's turn.

There is deliberately NO tool layer. The agent converses: it remembers what
the caller told it (the transcript is the memory) and answers from the prompt.
Measured on this box, sending the 22 tool schemas cost 1778 of 5290 prompt
tokens and dropped generation from 12.5 to 10.2 tok/s - about a fifth of the
time-to-first-word budget on a voice channel - while register_eval scored the
same scenarios 12/14 mechanically clean without them against 10/14 with them.

Per BACKEND_COMPLETION.md Sec3.1: the prompt is assembled per turn by
prompt_builder.py (condensed core + one flow playbook + that flow's exemplars),
and the ledger is real server-side state per connection_id - an in-process dict
for v1, since Redis only buys reconnect and multi-process, neither of which
exists yet.

stream_utterance() is the interface the live transports use: it yields each
clause as soon as it closes, so TTS can start speaking while the model is still
generating, then a final AgentTurn carrying the full text and the grounding
verdict for the turn. handle_utterance() is the same thing collapsed to a
string, for the eval scripts and tests that have no use for partial output.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import logging
import re

from .clause_chunker import ClauseChunker
from .grounding import grounding_sources, unbacked_action_claims, ungrounded_identifiers
from .llm import LlmClient, LlmReply, ReplyComplete
from .prompt_builder import PromptBuilder, detect_intent
from .settings import ConversationSettings

logger = logging.getLogger("aica.conversation")

# Placeholders golden/main_prompt.txt is known to use (its Sec1/Sec5B/Sec6D
# references: {{agent_name}}, {{caller_mobile}}, {{mrn}}, {{campaign}},
# {{patient_name}}, {{caller_name}}, {{last_visit}}). Anything else in the
# template is almost certainly a typo or a new placeholder nobody wired up -
# substituting it with "" would silently leak a blank into what the agent
# says on a live call, so unknown placeholders are left as literal text and
# logged instead of guessed at.
KNOWN_PLACEHOLDERS = frozenset(
    {"agent_name", "caller_name", "caller_mobile", "mrn", "campaign", "last_visit", "patient_name"}
)

_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")

# Tool results carry two kinds of key. Some are facts the agent must not
# re-ask for and must read back verbatim (mrn, appointment_id, eta_minutes).
# The rest are per-call control flow - whether a lookup hit, whether a tool
# errored, the nested payloads (slot lists, bill line items) that are already
# in the tool message verbatim a few lines up in the history. Only the first
# kind belongs in the standing facts block: restating "found: True" every turn
# teaches the model nothing and spends tokens, and re-flattening a slot list
# into prose invites it to quote a slot that was never offered.
_LEDGER_CONTROL_KEYS = frozenset({"found", "error", "status", "verified", "reason"})

# Human-readable labels for the ledger keys the tools in tools.py actually
# return. A key with no entry here is still shown (falling back to the raw
# key) rather than dropped - a new tool returning a new fact should surface
# to the model immediately, not go silently missing until someone updates
# this table.
_LEDGER_LABELS: dict[str, str] = {
    "appointment_id": "appointment ID",
    "bill_number": "bill number",
    "cancellation_reference": "cancellation reference",
    "confirmation_status": "appointment confirmation",
    "dispatch_id": "ambulance dispatch ID",
    "escalation_id": "escalation ID",
    "eta_minutes": "ambulance ETA (minutes)",
    "order_id": "lab order ID",
    "policy_number": "policy number",
    "preauth_reference": "pre-authorisation reference",
    "refill_reference": "refill reference",
    "request_id": "records request ID",
    "sent_channel": "report sent via",
    "ticket_id": "ticket ID",
    "transferred_to": "call transferred to",
}

# golden/main_prompt.txt Sec5A: the opening line is said verbatim on
# [CALL_CONNECTED], before any LLM call - "Your first action is always to
# SPEAK. Never call a tool or hang up on the first turn."
OPENING_LINE = "வணக்கம், அருவி ஹாஸ்பிட்டல். நான் {{agent_name}} பேசுறேன். உங்களுக்கு எப்படி help பண்ணலாம்?"


def render_template(template: str, metadata: dict[str, str]) -> str:
    def _substitute(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in KNOWN_PLACEHOLDERS:
            logger.warning("unknown template placeholder {{%s}} left unsubstituted", key)
            return match.group(0)
        return metadata.get(key, "")

    return _PLACEHOLDER_RE.sub(_substitute, template)


@dataclass(frozen=True)
class AgentClause:
    """One speakable clause of the agent's reply, released as soon as it closes."""

    text: str


@dataclass(frozen=True)
class AgentTurn:
    """The completed turn: everything said, plus any call-control action."""

    text: str
    # IDs/phone numbers the agent stated that no tool, no caller turn and no
    # standing fact accounts for - see backend/grounding.py. Empty is the
    # expected case; anything here is a fabrication the caller was just told.
    ungrounded: tuple[str, ...] = ()
    # Actions the agent claimed to have COMPLETED with no tool call behind
    # them - "Ambulance அனுப்பிட்டேன்" having dispatched nothing. Separate from
    # `ungrounded` because there is no invented identifier to point at: the
    # sentence is a lie about what the server did, not about what it knows.
    unbacked_claims: tuple[str, ...] = ()


@dataclass
class CallSession:
    connection_id: str
    metadata: dict[str, str]
    ledger: dict[str, object] = field(default_factory=dict)
    messages: list[dict] = field(default_factory=list)
    # Sticky across turns: a caller states their reason once, then answers
    # follow-up questions ("ஆமாம்", a phone number) that match no trigger at
    # all. Re-detecting per turn would drop the playbook mid-flow, so a new
    # detection replaces this and silence leaves it alone (Sec6E).
    intent: str | None = None
    # The previous turn's first clause, and how many turns running the model
    # has opened with it. Drives the repeat breaker - see _is_repeat_opening.
    last_opening: str = ""
    repeat_count: int = 0

    def known_facts(self) -> dict[str, str]:
        """Placeholder substitutions for this turn: opening metadata, overlaid
        by anything the ledger has since learned.

        Ledger wins on conflict: a tool result is a fresher, better-grounded
        source than whatever the call opened with (a CRM guess from caller ID,
        say). A blank/None ledger value never overwrites a real metadata one,
        so a tool returning `{"mrn": None}` cannot erase a known MRN.
        """
        facts = dict(self.metadata)
        for key in KNOWN_PLACEHOLDERS:
            value = self.ledger.get(key)
            if value not in (None, ""):
                facts[key] = str(value)
        return facts


def _format_established_facts(session: CallSession) -> str:
    """Render the non-placeholder ledger facts as a standing block.

    KNOWN FACTS in the core prompt only has slots for the seven caller-identity
    placeholders. Everything else a call establishes - the appointment ID just
    booked, the ticket number just raised, the ambulance ETA - has no slot, and
    lives only in a tool message that scrolls further back with every turn. On
    a long call that is exactly how an agent ends up inventing a reference ID
    at closing time, which the prompt's GROUNDING section forbids outright. So
    the facts ride at the front of every turn instead, where they cannot scroll
    away.
    """
    lines = []
    for key, value in session.ledger.items():
        if key in KNOWN_PLACEHOLDERS or key in _LEDGER_CONTROL_KEYS:
            continue
        # Scalars only. Nested payloads (slot lists, bill line items) are
        # already verbatim in their tool message; flattening them to prose here
        # would both duplicate tokens and blur which values a tool actually
        # returned - the thing GROUNDING most needs kept sharp.
        if not isinstance(value, (str, int, float, bool)) or value in (None, ""):
            continue
        lines.append(f"{_LEDGER_LABELS.get(key, key)}: {value}")

    if not lines:
        return ""
    return (
        "\n## ESTABLISHED THIS CALL — say these back exactly, never re-ask, never re-invent\n"
        + "\n".join(lines)
    )


class ConversationManager:
    """Owns the prompt builder, per-call sessions, and the shared mock hospital DB."""

    def __init__(self, settings: ConversationSettings) -> None:
        self.settings = settings
        self.prompts = PromptBuilder(
            settings.runtime_core_path, settings.prompt_path, settings.exemplars_path
        )
        self._sessions: dict[str, CallSession] = {}

    @property
    def ready(self) -> bool:
        return self.prompts.ready

    def load(self) -> None:
        """Read the prompts once during startup, keeping call-time latency low."""
        self.prompts.load()

    def start_call(self, connection_id: str, **metadata: str) -> str:
        """Open a new call session and return the scripted greeting (Sec5A - no LLM call)."""
        if not self.prompts.ready:
            raise RuntimeError("ConversationManager prompt is not loaded")

        greeting = render_template(OPENING_LINE, metadata)
        session = CallSession(
            connection_id=connection_id,
            metadata=dict(metadata),
            ledger=dict(metadata),
            # The system message is a placeholder here and rewritten every turn
            # by _refresh_system_prompt() once the flow is known - index 0 is
            # reserved for it so history stays append-only.
            messages=[
                {"role": "system", "content": ""},
                {"role": "assistant", "content": greeting},
            ],
        )
        session.messages[0]["content"] = self._system_prompt_for(session)
        self._sessions[connection_id] = session
        return greeting

    async def prewarm(self, connection_id: str, llm: LlmClient) -> bool:
        """Evaluate this call's prompt while the caller is hearing the greeting.

        The greeting is a fixed line spoken with no LLM involvement at all, and
        it takes roughly three seconds of audio to say. For those three seconds
        the model is idle while the caller is occupied - which is exactly long
        enough to pay the one cost that cannot be cached away.

        That cost is the first prompt evaluation. Ollama caches the evaluated
        prefix of a prompt, so the SECOND turn onwards is nearly free (measured
        294 ms), but the first turn of a call has to evaluate ~2.7k tokens cold
        and that measured 6-8 seconds - the single largest contributor to
        first-turn latency. Doing it here moves it off the critical path and
        underneath audio the caller is already listening to.

        Generates one token, which is the smallest amount of work that still
        forces a full prompt evaluation. The result is thrown away; the point
        is the server-side cache it leaves behind.

        Best effort by design. A prewarm that fails, times out or is cancelled
        must never affect the call - the turn that follows simply pays the cold
        cost it would have paid anyway, so every failure mode here degrades to
        "no faster than before".
        """
        session = self._sessions.get(connection_id)
        if session is None:
            return False

        try:
            messages = _with_language_reminder(
                session.messages, self._turn_facts_message(session)
            )
            async for event in llm.stream(messages, max_tokens=1):
                if isinstance(event, ReplyComplete):
                    break
        except asyncio.CancelledError:
            # The caller spoke before the warm finished. Their turn owns the
            # model now; drop this quietly rather than racing it.
            raise
        except Exception as error:
            logger.info("prompt prewarm for %s did not complete (%s)", connection_id, error)
            return False

        logger.info("prompt prewarmed for %s", connection_id)
        return True

    def _system_prompt_for(self, session: CallSession) -> str:
        """The STATIC half of the prompt: rules, playbook, exemplars.

        Deliberately rendered against the call's OPENING metadata, which is
        fixed at start_call, rather than against the live ledger - so this text
        changes only when the detected flow changes, and is byte-identical
        across the turns in between.

        That matters for one measured reason. Ollama caches the evaluated
        prefix of a prompt, and the cache is a prefix cache: mutate one word
        near the top and everything behind it is evaluated again. On this box,
        with this ~2.7k-token prompt:

            identical prefix          36 ms
            prefix mutated by a word  28,895 ms

        The live ledger is exactly what mutates - a tool returns an MRN and the
        KNOWN FACTS block changes - so rendering it in HERE re-evaluated the
        whole prompt on every turn that learned anything, and twice on any turn
        with a tool call, since the prompt is refreshed inside the tool loop.

        The ledger still reaches the model, and still reaches it on the
        iteration right after the lookup (the PART 6 fix this must not undo) -
        it is just carried by _turn_facts_message() at the END of the message
        list, where changing it costs a hundred tokens instead of three
        thousand. Later is also strictly better for recency, which is the same
        reasoning that put _LANGUAGE_REMINDER last.
        """
        return render_template(self.prompts.build(session.intent), dict(session.metadata))

    def _turn_facts_message(self, session: CallSession) -> str:
        """The VOLATILE half: what this call has actually established so far.

        Rendered fresh every turn and appended near the end of the message
        list. See _system_prompt_for for why it is not in the system prompt.

        Only facts that are actually KNOWN are rendered. The block used to list
        all five labels every turn with blanks after them and a paragraph
        explaining what a blank meant - about 70 tokens of empty scaffolding on
        every single turn of every call, since a browser call opens knowing
        nothing but the agent's own name. Worse, it put "mrn:" in front of a
        model the reminder immediately tells never to say an MRN.

        What the caller has said is not lost by dropping it: the transcript is
        in the message list directly above, which is where a conversational
        agent's memory actually lives. This block is only for facts the SERVER
        knows independently - a telephony leg's caller ID, say.
        """
        facts = session.known_facts()
        lines = [
            f"{label}: {facts[key]}"
            for key, label in (
                ("caller_name", "caller_name"),
                ("caller_mobile", "caller_mobile"),
                ("mrn", "mrn"),
                ("patient_name", "patient_name"),
                ("last_visit", "last_visit"),
            )
            if facts.get(key)
        ]
        established = _format_established_facts(session)
        if not lines and not established:
            return ""

        prompt = "\n".join(
            ["## KNOWN FACTS — already verified, never ask for these again", *lines]
        )
        if established:
            prompt = f"{prompt}\n{established}\n"
        return prompt

    def end_call(self, connection_id: str) -> None:
        self._sessions.pop(connection_id, None)

    async def handle_utterance(self, connection_id: str, llm: LlmClient, text: str) -> str:
        """Run one caller turn and return the agent's full reply text.

        Kept for callers with no use for partial output - the eval scripts and
        the unit tests. Live transports should use stream_utterance() instead,
        so the first clause reaches TTS without waiting for the last token.
        """
        spoken: list[str] = []
        async for event in self.stream_utterance(connection_id, llm, text):
            if isinstance(event, AgentTurn):
                return event.text
            if isinstance(event, AgentClause):
                spoken.append(event.text)
        # stream_utterance always ends with an AgentTurn; this is unreachable
        # short of a generator being closed early by its consumer.
        return " ".join(spoken)

    def _check_grounding(self, session: CallSession, reply: str) -> tuple[str, ...]:
        """Flag identifiers in `reply` that nothing in this call accounts for."""
        # Sources are tool results and caller turns (grounding_sources), plus
        # the facts this call actually holds. NOT the system prompt: it carries
        # the few-shot exemplars, whose worked example includes an MRN, and
        # treating that as provenance is exactly how a parroted exemplar passes
        # for a lookup. See backend/grounding.py.
        sources = grounding_sources(session.messages)
        sources += [str(value) for value in session.ledger.values() if value not in (None, "")]
        sources += [str(value) for value in session.metadata.values() if value not in (None, "")]
        invented = ungrounded_identifiers(reply, sources)
        if invented:
            logger.error(
                "GROUNDING: %s stated identifier(s) no tool returned: %s",
                session.connection_id,
                ", ".join(invented),
            )
        return tuple(invented)

    def _check_action_claims(self, session: CallSession, reply: str) -> tuple[str, ...]:
        """Flag actions `reply` says are done that no tool in this call did."""
        called = {
            call["function"]["name"]
            for message in session.messages
            for call in (message.get("tool_calls") or [])
        }
        claims = unbacked_action_claims(reply, called)
        if claims:
            logger.error(
                "UNBACKED CLAIM: %s %s - no tool call behind it",
                session.connection_id,
                "; ".join(claims),
            )
        return tuple(claims)

    def record_interrupted_turn(self, connection_id: str, spoken: str) -> None:
        """Append what the agent actually got out before the caller cut in.

        Barge-in cancels the task consuming stream_utterance(), which can land
        on a yield - leaving the turn's assistant message never appended, so
        the model's next turn sees the caller's line answered by nothing at
        all. Recording the truncated text keeps the history honest: the model
        should believe it said exactly what the caller heard, no more, so it
        can pick up mid-thought rather than start the same sentence again.
        """
        session = self._sessions.get(connection_id)
        if session is None:
            return
        text = spoken.strip()
        if not text:
            return
        if session.messages and session.messages[-1].get("role") == "assistant":
            return
        session.messages.append({"role": "assistant", "content": text})

    async def stream_utterance(self, connection_id: str, llm: LlmClient, text: str):
        """Run one caller turn, yielding each clause as soon as it closes.

        Yields zero or more AgentClause, then exactly one AgentTurn.

        There is no tool loop. This agent talks: it remembers what the caller
        told it (the transcript IS the memory) and answers from the prompt.
        Measured on this box, sending the 22 tool schemas cost 1778 of the
        5290 prompt tokens AND dropped generation from 12.5 to 10.2 tok/s,
        which is ~20% of the time-to-first-word budget on a voice channel -
        and register_eval scored the same scenarios 12/14 clean without them
        against 10/14 with them. Removing them is faster AND better spoken.
        """
        session = self._sessions.get(connection_id)
        if session is None:
            raise RuntimeError(f"no active call session for {connection_id}")

        detected = detect_intent(text)
        if detected is not None and detected != session.intent:
            logger.info("flow detected for %s: %s", connection_id, detected)
            session.intent = detected
        session.messages[0]["content"] = self._system_prompt_for(session)

        _append_caller_turn(session, text)

        chunker = ClauseChunker()
        spoken: list[str] = []
        asked_question = False

        def speakable(clause: str) -> bool:
            """Whether this clause may be spoken, given what already has been.

            Enforces the one turn-discipline rule the model still breaks: ONE
            question per turn. Prose has failed three times (LLM_STACK.md Sec9)
            - runtime_core.txt states the rule three ways in one line and the
            model asks two anyway.

            This is not a truncation of the reply. Measured against the real
            recorded calls in call_events.db, a two-question turn arrives as
            two SEPARATE clauses:

                | நீங்க ... இருப்பீங்களா சார்?
                | எங்கே ... போகிறீர்கள்?

            so the second is still unspoken when it closes and can simply be
            withheld. Later NON-question clauses are kept - the closing line
            ("desk-ல இருந்து call பண்ணுவாங்க") often follows the question,
            and dropping the tail wholesale would lose it.

            '?' is the same test backend/scripts/register_eval.py scores turns
            with, deliberately: one definition, so the guard and the eval
            cannot disagree about what a question is.
            """
            nonlocal asked_question, stuck
            # The repeat check has to live in here rather than at the feed()
            # loop, because a one-clause reply - which is the exact shape a
            # stuck model produces - is never closed by feed() at all. The
            # chunker releases it from flush() once the stream is done, so a
            # guard on the loop alone silently never fires on the only turns it
            # exists for. This function is the one place BOTH paths go through.
            if not spoken and _is_repeat_opening(clause, session.last_opening):
                logger.info(
                    "repeat breaker: %s was about to open with %r again", connection_id, clause
                )
                stuck = True
                return False
            if "?" not in clause:
                return True
            if asked_question:
                logger.info(
                    "turn discipline: withheld a second question from %s: %s",
                    connection_id,
                    clause,
                )
                return False
            asked_question = True
            return True

        reply: LlmReply | None = None
        # Set by speakable() when this turn opens with the clause the last one
        # opened with. Declared before the stream so both the feed() loop and
        # the flush() tail can end the turn on it.
        stuck = False
        stream = llm.stream(
            _with_language_reminder(session.messages, self._turn_facts_message(session))
        )
        async for event in stream:
            if isinstance(event, ReplyComplete):
                reply = event.reply
                break
            for clause in chunker.feed(event.text):
                if not speakable(clause):
                    continue
                spoken.append(clause)
                yield AgentClause(clause)
            if stuck:
                # Nothing of the repeat was spoken and nothing more is worth
                # generating, so drop the rest of the completion on the floor.
                await stream.aclose()
                break

        if reply is None and not stuck:
            raise RuntimeError("LLM stream ended without a ReplyComplete event")

        # The chunker never closes a clause on buffer-end (see
        # clause_chunker.py), so the reply's last clause only exists once the
        # stream is done and we ask for it.
        if not stuck:
            remainder = chunker.flush()
            if remainder and speakable(remainder):
                spoken.append(remainder)
                yield AgentClause(remainder)

        if stuck:
            session.repeat_count = min(session.repeat_count + 1, len(_STUCK_REPLIES))
            recovery = _STUCK_REPLIES[session.repeat_count - 1]
            # Cleared, not set to the recovery line: the recovery lines differ
            # from each other, so comparing against one would never match, and
            # the escalation is what the counter is for.
            session.last_opening = ""
            session.messages.append({"role": "assistant", "content": recovery})
            _trim_history(session)
            for clause in split_reply_into_clauses(recovery):
                yield AgentClause(clause)
            yield AgentTurn(text=recovery)
            return

        spoken_text = " ".join(spoken)
        # What was SPOKEN, not what was generated - the two differ whenever a
        # second question was withheld above. Same reasoning as
        # record_interrupted_turn: the model must believe it said exactly what
        # the caller heard, or it will treat a question nobody was asked as
        # already asked and never come back to it.
        session.messages.append({"role": "assistant", "content": spoken_text})
        # Remember what this turn opened with so the next one can be checked
        # against it, and forgive the earlier repeats: the counter escalates
        # only while the model is stuck turn after turn, so one recovered turn
        # puts a later bad patch back at the gentlest wording.
        session.last_opening = spoken[0] if spoken else ""
        session.repeat_count = 0
        _trim_history(session)
        yield AgentTurn(
            text=spoken_text,
            ungrounded=self._check_grounding(session, spoken_text),
            unbacked_claims=self._check_action_claims(session, spoken_text),
        )


# The assembled system prompt is ~3.5k tokens against the Modelfile's
# num_ctx 8192, so a call has roughly 4.6k tokens of room for its history.
# Nothing used to bound that. A long enough call overflows the window, and
# Ollama truncates from the FRONT - taking the system prompt's language rules
# with it. That is not a hypothetical failure: it is the exact bug that
# backend/prompt_builder.py exists to fix (the agent switches to English and
# starts inventing identifiers), and it is silent.
#
# 40 messages is ~20 exchanges, far longer than any real hospital call and
# still ~2.4k tokens inside the window. messages[0] is the system prompt and
# is never dropped - it is the one message that must survive.
#
# ponytail: a flat cap, not summarisation. If calls ever genuinely run past 20
# exchanges, summarise the dropped span into the facts block instead.
MAX_HISTORY_MESSAGES = 40


def split_reply_into_clauses(text: str) -> list[str]:
    """One-shot helper: feed a whole reply through ClauseChunker at once.

    The chunker never closes a clause on buffer-end (see clause_chunker.py), so
    flush() is what releases the final clause of any complete text - feed()
    alone drops it. Lives here rather than in main.py because the repeat
    breaker below speaks a canned line and needs the same splitting; main.py
    imports it for the scripted greeting.
    """
    chunker = ClauseChunker()
    clauses = chunker.feed(text)
    remainder = chunker.flush()
    if remainder:
        clauses.append(remainder)
    return clauses


# What the agent says instead of repeating itself, in order. The first two ask
# again in fresh words; the third stops asking and hands the call to the desk,
# so a caller on a line that is not working gets an exit instead of the same
# sentence until they hang up.
_STUCK_REPLIES = (
    "மன்னிச்சுடுங்க சார், clear-ஆ கேட்கல. இன்னொரு தரம் சொல்லுங்களா?",
    "Line-ல கொஞ்சம் disturbance சார். கொஞ்சம் மெதுவா சொல்லுங்க?",
    "இன்னும் சரியா கேட்கல சார். Desk-ல இருந்து உங்களுக்கு call பண்ண சொல்றேன். நன்றி சார்.",
)

# A two-word opening ("சரி சார்.", "நானே சார்,") is an acknowledgement, and two
# turns running may legitimately start with one. A longer opening repeated
# verbatim is the model stuck, not the model agreeing.
_MIN_REPEAT_WORDS = 3


def _is_repeat_opening(clause: str, previous: str) -> bool:
    """Whether this turn is opening with the exact clause the last one did.

    Prose cannot hold this. runtime_core.txt has said "Never say a turn you
    have already said" since before the tool removal, and on the real call in
    call_events.db (97dd5ac7) the agent said "98407 21534 என்ன சார்?" on five
    consecutive turns anyway, whatever the caller said in between. Two further
    attempts to teach it - a demonstrated bad-line exchange in the core prompt,
    then a shorter version using bracketed placeholders - both left the repeat
    in place AND pushed the median first clause from 2.7s to 6.8s, because the
    extra section lengthened every OTHER turn too. Measurements in
    LLM_TEST_RESULTS.txt.

    So it is enforced here, for the same reason and in the same shape as
    speakable()'s one-question rule a few lines up: what the model will not do
    on instruction, the server does for it.

    Checked on the FIRST clause, before it is spoken, so none of the repeat
    reaches the caller and the rest of the generation can be abandoned - which
    makes this a latency win on exactly the turns that were slowest.
    """
    if not previous or clause != previous:
        return False
    return len(clause.split()) >= _MIN_REPEAT_WORDS


def _append_caller_turn(session: CallSession, text: str) -> None:
    """Add the caller's line, merging it into the previous one if the agent
    never got a word out in between.

    A barge-in that lands before the FIRST clause is spoken leaves
    record_interrupted_turn() with an empty string and therefore nothing to
    append, so the next caller turn lands directly behind the previous one.
    Measured on the real call in call_events.db (97dd5ac7), a noisy stretch put
    TWELVE consecutive user messages into a 21-message history, and against
    that history the model stopped answering at all: it reproduced its own
    previous turn verbatim on five turns running - "98407 21534 என்ன சார்?" -
    whatever the caller said next. Replaying the identical message list against
    the real model reproduces that reply character for character, and merging
    the runs is what stops it.

    Merging, not dropping. The caller really did say both things before anyone
    answered, and the first half is where the CONTENT usually is - the real
    call lost "ஆஸ்டோ department க்கு வேணும்" behind a barge-in and only the
    noise that followed it would have survived a drop.
    """
    previous = session.messages[-1] if session.messages else None
    if previous is not None and previous.get("role") == "user":
        previous["content"] = f"{previous['content']} {text}"
        return
    session.messages.append({"role": "user", "content": text})


def _trim_history(session: CallSession) -> None:
    """Drop the oldest exchanges, never the system prompt."""
    overflow = len(session.messages) - 1 - MAX_HISTORY_MESSAGES
    if overflow <= 0:
        return
    logger.info("call %s: trimming %d oldest messages", session.connection_id, overflow)
    del session.messages[1 : 1 + overflow]


# Recency beats distance: a small model reliably drifts into pure English by
# the third or fourth turn even with the language rules in the system message,
# because those sit thousands of tokens back while the recent turns are the
# strongest signal. This rides immediately before generation, costs ~40 tokens,
# and is not stored in history - so it never accumulates across a long call.
#
# It used to have to open by naming "call a tool", because a speech-only
# instruction here read to the model as "produce speech now" and suppressed
# tool calling entirely. There are no tools any more, so that constraint is
# gone and the whole message is speaking rules - ordered by how often each is
# actually broken, measured with backend/scripts/register_eval.py on unseen
# scenarios, most-violated first rather than most important-sounding first.
_LANGUAGE_REMINDER = (
    "[Reply now, out loud. ONE question per turn - never two - and put it "
    "last. Under 40 words. Never repeat the caller's own sentence back at "
    "them; acknowledge in two or three words and move on. "
    "Reply in spoken Chennai Tamil (Tamil script) code-mixed with English "
    "hospital words in Latin script - never pure English. "
    "If the caller speaks English, mirror them but keep சார்/மேடம். "
    "Take the request down: ask for the next detail you still need. Never "
    "refuse the request itself and never open with what you cannot do. "
    "Never state an MRN, appointment ID, bill amount, slot time or report "
    "result, and never claim you already booked, cancelled or checked "
    "anything - the desk does that after the call.]"
)


# Counted in WORDS, not letters. Letters over-count English badly: Tamil packs
# a syllable into one glyph where English spells it out, so
# "Cardiology-ல ஒரு appointment book பண்ணணும்" - an ordinary code-mixed
# TAMIL line - is 64% Latin by character and would wrongly flip the agent into
# English. By word it is 2 English of 5, which is what it actually is.
#
# A word counts as Tamil if it contains ANY Tamil character, so "Cardiology-ல"
# is Tamil: the case suffix is what makes the sentence Tamil. Bare digits are
# ignored - a phone number is not evidence of either language.
#
# The bar is deliberately high. Tamil is the safe default (it is the register
# every exemplar demonstrates), so switching needs most of the turn to be
# English, not merely some of it.
_WORD_RE = re.compile(r"[\w஀-௿]+")
_TAMIL_LETTER_RE = re.compile(r"[஀-௿]")
_ENGLISH_SHARE_TO_MIRROR = 0.7


def caller_is_speaking_english(text: str) -> bool:
    """Whether this caller turn is English rather than code-mixed Tamil.

    NOT currently wired into the prompt, deliberately. Switching the register
    on this was built and measured twice and made things WORSE both times:
    prose alone ("reply mainly in English") only half-moved the register and
    introduced parroting, and adding an English worked example alongside the
    twenty Tamil ones produced ungrammatical output mixing both
    ("எந்த நாள் உங்களுக்கு சொல்லுங்க?"). Answering a
    English-speaking caller in coherent Tamil beats answering them in broken
    half-English, so the agent stays in Tamil.

    Kept because the measurement is the correct one and any future attempt
    needs it: count WORDS, not letters (see the comment above).
    """
    words = [w for w in _WORD_RE.findall(text) if not w.isdigit()]
    if not words:
        return False
    english = sum(1 for w in words if not _TAMIL_LETTER_RE.search(w))
    return english / len(words) >= _ENGLISH_SHARE_TO_MIRROR


def _with_language_reminder(messages: list[dict], facts: str = "") -> list[dict]:
    """Append this turn's volatile context, then the reminder.

    Order is load-bearing in two directions. The facts go AFTER the history so
    that changing them re-evaluates a hundred tokens instead of the three
    thousand sitting in front of them (see _system_prompt_for). The reminder
    stays LAST, because that is the message the model reads immediately before
    it decides whether to call a tool or speak - see the comment on
    _LANGUAGE_REMINDER, and the regression test that guards it.
    """
    tail: list[dict] = []
    if facts:
        tail.append({"role": "system", "content": facts})
    tail.append({"role": "system", "content": _LANGUAGE_REMINDER})
    return [*messages, *tail]
