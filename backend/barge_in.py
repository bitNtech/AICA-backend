"""Tracks the in-flight agent-speech task so a fresh caller turn can cancel it.

BACKEND_COMPLETION.md Sec3.2: the moment VAD emits a fresh speech_started while
the agent is speaking, the orchestrator must cancel the in-flight TTS
generation and stop sending further audio immediately. ActiveSpeech is the
"is the agent currently talking" state that didn't exist before - deliberately
kept free of asyncio.Task's caller (main.py's speak()/handle_conversation_turns)
so the cancellation bookkeeping itself is unit-testable without a websocket,
VAD, or TTS model.
"""

from __future__ import annotations

import asyncio
import re
import time

# How long after the last scheduled audio the agent may still be heard in the
# room. The server knows exactly how much audio it has SENT, and the browser
# schedules each clause after the previous one, so summing durations tracks
# playback. This is the extra allowance for speaker-to-microphone travel and
# for the client's own buffering.
AUDIBLE_TAIL_SECONDS = 0.35


def is_probably_self_echo(transcript: str, agent_text: str, threshold: float = 0.75) -> bool:
    """Whether `transcript` is the agent hearing itself through the speakers.

    Echo cancellation in the browser is enabled and mostly works, but what
    leaks through is the agent's own sentence - and unlike background noise it
    transcribes to real words, so "is the transcript empty" does not catch it.
    What does catch it is that the words are the ones the agent just said.

    THE HARD PART IS NOT DETECTING ECHO, IT IS NOT DETECTING A REPLY. A caller
    answering a question reuses the question's words - that is what answering
    is. Measured live: the agent asked "எந்த department-க்கு வேணும்?" and the
    caller said "Cardiology department வேணும் சார்", which is 3 of 4 words the
    agent had just said. Word overlap alone threw the caller's answer away.

    What separates them is NEW CONTENT. An answer carries the thing being
    answered - "Cardiology" - while echo is the agent's own sentence coming
    back with nothing added. So a transcript is echo only when it is BOTH
    mostly the agent's words AND brings essentially nothing of its own.

    Deliberately a containment test rather than a similarity ratio: the ASR
    mangles its own re-heard audio, dropping and merging words, so echo is
    usually a SUBSET of what was said rather than a close match of the whole.
    That is also why one unmatched word is tolerated - a garbled echo often
    produces one - but two mean the caller is saying something.

    A short transcript is never judged an echo. "ஆமாம்" ("yes") is one word,
    it will appear in something the agent said sooner or later, and refusing
    to hear a caller say yes is far worse than occasionally acting on an echo.
    """
    words = _WORDS_RE.findall(transcript.lower())
    if len(words) < 3:
        return False
    said = set(_WORDS_RE.findall(agent_text.lower()))
    if not said:
        return False
    if any(w not in said for w in words):
        # The caller brought a word of their own. Whatever else this is, it is
        # not the agent's sentence coming back.
        #
        # Zero tolerance on purpose, and the asymmetry is the point: missing an
        # echo costs one wasted turn, while discarding a real answer loses what
        # the caller actually said and they have to repeat themselves. Measured
        # live, a one-word allowance was already enough to throw away
        # "Cardiology department வேணும் சார்" - the answer to the agent's own
        # question - because only "Cardiology" was new.
        return False
    return sum(1 for w in words if w in said) / len(words) >= threshold


_WORDS_RE = re.compile(r"[\w\u0b80-\u0bff]+")


class ActiveSpeech:
    """Holds at most one in-flight agent-speech task per call."""

    def __init__(
        self,
        sustained_frames: int = 1,
        sustained_frames_while_audible: int | None = None,
        clock=time.monotonic,
    ) -> None:
        self._task: asyncio.Task | None = None
        self._sustained_frames = max(1, sustained_frames)
        # A much higher bar while the agent's own voice is still in the room.
        # Residual echo of the agent talking looks exactly like sustained
        # speech to a VAD - it IS sustained speech, just not the caller's - so
        # a consecutive-frame gate alone cannot separate them. Demanding
        # noticeably more evidence can: room noise and echo leak in bursts,
        # while a caller who genuinely wants the floor keeps talking.
        self._sustained_frames_while_audible = max(
            self._sustained_frames, sustained_frames_while_audible or self._sustained_frames
        )
        self._speech_frames = 0
        self._clock = clock
        # When the audio already sent will have finished playing. Mirrors the
        # browser's own scheduling clock (playAt += clause duration).
        self._audible_until = 0.0

    def set(self, task: asyncio.Task) -> None:
        self._task = task

    @property
    def audible(self) -> bool:
        """Whether audio this server sent could still be coming out of a speaker."""
        return self._clock() < self._audible_until + AUDIBLE_TAIL_SECONDS

    def note_audio_sent(self, seconds: float) -> None:
        """Record that `seconds` of agent audio has been handed to the client.

        `agent_speaking_end` fires when the server finishes SENDING, which is
        not when the caller stops hearing the agent: the client schedules each
        clause after the previous one, so playback runs on for as long as the
        buffered audio lasts. Tracking it the same way the client does is what
        lets the server know its own voice is still in the room.
        """
        if seconds <= 0:
            return
        self._audible_until = max(self._audible_until, self._clock()) + seconds

    def silence(self) -> None:
        """The client has stopped playback, so nothing more will be heard."""
        self._audible_until = 0.0

    def clear(self, task: asyncio.Task) -> None:
        """Clear only if `task` is still the tracked one (a superseded task's
        own cleanup must not clobber whatever replaced it)."""
        if self._task is task:
            self._task = None

    def note_speech(self, speech_frame: bool, speech_started: bool = False) -> bool:
        """Interrupt only once the caller has been speaking for long enough.

        VAD flags speech per 16 ms hop, and cancelling the agent's turn on the
        FIRST flagged hop is what made it stoppable by a cough, a keystroke or
        a breath - the caller then hears the agent give up mid-sentence for no
        reason. Requiring `sustained_frames` consecutive flagged hops keeps a
        real interjection working (a spoken word is far longer than the gate)
        while noise blips, which are one or two hops, pass under it.

        Returns True on the single frame that actually cancelled something, so
        the caller can log the barge-in once rather than every frame after.
        """
        if speech_started:
            self._speech_frames = 0
        if not speech_frame:
            # THE RESET IS THE WHOLE GATE. Without it these frames only ever
            # accumulate, so scattered background blips - a fan, a door, a
            # keyboard - sum across seconds of silence and eventually cancel
            # the agent although nobody spoke. Measured on real calls: 34 of
            # 87 captured turns transcribed to the empty string, and each was
            # free to reach this counter. The caller then hears the agent stop
            # mid-sentence and stay stopped, because an empty transcript
            # (correctly) starts no new turn - there is nothing to resume it.
            #
            # With the reset the gate means what it says: `sustained_frames`
            # CONSECUTIVE flagged hops, 240 ms of unbroken speech. A real
            # interjection clears that easily; noise essentially never does.
            self._speech_frames = 0
            return False
        self._speech_frames += 1
        required = (
            self._sustained_frames_while_audible if self.audible else self._sustained_frames
        )
        # >=, not ==. The bar FALLS mid-utterance: it is 40 frames while the
        # agent's own audio is still in the room and 15 once that audio has
        # drained, and a turn's clauses are synthesized over a network call
        # measured at 0.9-10.8s, so the room routinely goes quiet while the
        # agent is still mid-turn. With == the counter has already passed 15 by
        # then and can never equal it again, so a caller talking straight
        # through that moment could not interrupt for the rest of the
        # utterance - the one shape of barge-in this gate exists for.
        # interrupt() is idempotent (it returns False once the task is done),
        # so testing >= costs one extra call per frame and nothing else.
        if self._speech_frames < required:
            return False
        return self.interrupt()

    def interrupt(self) -> bool:
        """Cancel the in-flight speech task, if any. Returns whether one was cancelled."""
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            # Cancelling stops the client's playback too, so the room goes
            # quiet immediately - the remaining buffered audio is dropped.
            self.silence()
            return True
        return False
