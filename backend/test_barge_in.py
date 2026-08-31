"""Self-check for ActiveSpeech's cancellation bookkeeping.

Uses plain asyncio.Task fakes (asyncio.sleep) - no websocket, VAD, or TTS
model needed, matching the fakes-over-real-models style of test_asr.py /
test_llm.py / test_tts.py.
"""

from __future__ import annotations

import asyncio
import contextlib

from .barge_in import AUDIBLE_TAIL_SECONDS, ActiveSpeech, is_probably_self_echo


async def test_interrupt_with_no_active_task_is_a_no_op() -> None:
    active_speech = ActiveSpeech()

    assert active_speech.interrupt() is False


async def test_interrupt_cancels_the_active_task() -> None:
    active_speech = ActiveSpeech()
    task = asyncio.create_task(asyncio.sleep(10))
    active_speech.set(task)

    cancelled = active_speech.interrupt()

    assert cancelled is True
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert task.cancelled()


async def test_interrupt_on_an_already_done_task_is_a_no_op() -> None:
    active_speech = ActiveSpeech()
    task = asyncio.create_task(_immediate())
    await task
    active_speech.set(task)

    assert active_speech.interrupt() is False


async def test_double_interrupt_is_safe() -> None:
    active_speech = ActiveSpeech()
    task = asyncio.create_task(asyncio.sleep(10))
    active_speech.set(task)

    first = active_speech.interrupt()
    # A task only becomes done() once the loop actually delivers the
    # cancellation, so let that happen before checking that a second
    # interrupt() (e.g. from a rapid double barge-in) is a no-op rather than
    # an error.
    with contextlib.suppress(asyncio.CancelledError):
        await task
    second = active_speech.interrupt()

    assert first is True
    assert second is False


async def test_clear_only_clears_the_matching_task() -> None:
    active_speech = ActiveSpeech()
    stale_task = asyncio.create_task(asyncio.sleep(10))
    current_task = asyncio.create_task(asyncio.sleep(10))
    active_speech.set(current_task)

    active_speech.clear(stale_task)  # a superseded task's own cleanup

    assert active_speech.interrupt() is True  # current_task is still tracked
    stale_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await stale_task
    with contextlib.suppress(asyncio.CancelledError):
        await current_task


async def test_clear_removes_the_tracked_task() -> None:
    active_speech = ActiveSpeech()
    task = asyncio.create_task(asyncio.sleep(10))
    active_speech.set(task)

    active_speech.clear(task)

    assert active_speech.interrupt() is False
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def _immediate() -> None:
    return None


if __name__ == "__main__":

    async def _main() -> None:
        await test_interrupt_with_no_active_task_is_a_no_op()
        await test_interrupt_cancels_the_active_task()
        await test_interrupt_on_an_already_done_task_is_a_no_op()
        await test_double_interrupt_is_safe()
        await test_clear_only_clears_the_matching_task()
        await test_clear_removes_the_tracked_task()
        print("ok")

    asyncio.run(_main())


def test_a_noise_blip_does_not_interrupt_but_a_spoken_word_does() -> None:
    """The reported failure: any small sound near the mic stopped the agent."""
    loop = asyncio.new_event_loop()
    try:
        speech = ActiveSpeech(sustained_frames=15)
        task = loop.create_task(asyncio.sleep(60))
        speech.set(task)

        # A cough: two flagged 16 ms hops, then silence. Must not cancel.
        assert speech.note_speech(True, speech_started=True) is False
        assert speech.note_speech(True) is False
        for _ in range(30):
            assert speech.note_speech(False) is False
        assert not task.cancelled() and not task.done()

        # A real interjection: 15 continuous hops (240 ms) of speech.
        results = [speech.note_speech(True, speech_started=i == 0) for i in range(15)]
        assert results[-1] is True, "sustained speech must barge in"
        assert results[:-1] == [False] * 14, "should fire exactly once"
        loop.run_until_complete(asyncio.sleep(0))
        assert task.cancelled()
    finally:
        loop.close()


def test_scattered_noise_never_accumulates_into_a_false_barge_in() -> None:
    """The agent-goes-silent bug, at its root.

    note_speech() used to count flagged frames without ever resetting on a
    silent one, so isolated background blips summed across seconds of silence
    and eventually cancelled the agent although nobody had spoken. The caller
    then heard it stop mid-sentence and stay stopped - an empty transcript
    (correctly) starts no new turn, so nothing was left to resume it.

    Measured context: 34 of the 87 real captured turns in call_events.db
    transcribed to the empty string, and every one of those was free to feed
    this counter.
    """

    async def scenario() -> None:
        speech = ActiveSpeech(sustained_frames=15)
        task = asyncio.create_task(asyncio.sleep(10))
        speech.set(task)

        # Ten times as much noise as the gate requires - but never two
        # flagged frames in a row.
        for _ in range(150):
            assert speech.note_speech(True) is False
            assert speech.note_speech(False) is False

        assert not task.cancelled() and not task.done(), "noise cancelled the agent"

        # A real interjection is continuous, and must still land promptly.
        interrupted = any(speech.note_speech(True) for _ in range(15))
        assert interrupted, "a real 240ms interruption was not honoured"
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


# --- the agent hearing its own voice (mic left open, speakers on) ---


def test_the_agent_knows_its_own_audio_is_still_playing() -> None:
    """agent_speaking_end fires when SENDING finishes, not when the caller
    stops hearing the agent - the client schedules clauses back to back."""
    now = [1000.0]
    speech = ActiveSpeech(15, 40, clock=lambda: now[0])

    assert not speech.audible
    speech.note_audio_sent(2.0)
    speech.note_audio_sent(3.0)          # queued behind the first, not overlapping
    assert speech.audible

    now[0] += 4.9
    assert speech.audible, "5s of audio was sent; it cannot be finished at 4.9s"
    now[0] += 0.2 + AUDIBLE_TAIL_SECONDS
    assert not speech.audible


def test_interrupting_silences_the_room_immediately() -> None:
    async def scenario() -> None:
        now = [1000.0]
        speech = ActiveSpeech(1, 1, clock=lambda: now[0])
        task = asyncio.create_task(asyncio.sleep(10))
        speech.set(task)
        speech.note_audio_sent(10.0)
        assert speech.audible

        assert speech.interrupt() is True
        # Cancelling drops the client's buffered playback, so nothing more is
        # heard - leaving `audible` set would gate the NEXT turn wrongly.
        assert not speech.audible
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


def test_interrupting_is_harder_while_the_agent_is_audible() -> None:
    """Residual echo IS sustained speech - just not the caller's - so the
    consecutive gate alone cannot separate it. More evidence can."""

    async def scenario() -> None:
        now = [1000.0]
        speech = ActiveSpeech(15, 40, clock=lambda: now[0])
        task = asyncio.create_task(asyncio.sleep(10))
        speech.set(task)
        speech.note_audio_sent(5.0)

        # 39 unbroken frames of "speech" while the agent is talking: echo.
        for _ in range(39):
            assert speech.note_speech(True) is False
        assert not task.cancelled(), "echo cut the agent off"
        # A caller who genuinely wants the floor keeps going.
        assert speech.note_speech(True) is True
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


def test_the_normal_gate_applies_once_the_room_is_quiet() -> None:
    async def scenario() -> None:
        now = [1000.0]
        speech = ActiveSpeech(15, 40, clock=lambda: now[0])
        task = asyncio.create_task(asyncio.sleep(10))
        speech.set(task)
        assert not speech.audible

        for _ in range(14):
            assert speech.note_speech(True) is False
        assert speech.note_speech(True) is True, "15 frames must interrupt a silent agent"
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


def test_a_caller_talking_through_the_moment_the_room_goes_quiet_can_still_cut_in() -> None:
    """The bar FALLS from 40 frames to 15 when the agent's audio drains, and a
    caller who keeps talking across that moment is already past 15.

    Not a contrived ordering. A turn's clauses are synthesized over a network
    call measured at 0.9-10.8s, so the audio sent so far routinely finishes
    playing while the agent is still mid-turn. Under the == test this counter
    had already overshot the lowered bar and could never equal it again, so
    barge-in was dead for the rest of that utterance - on precisely the turns
    where the caller has the most reason to interrupt.
    """

    async def scenario() -> None:
        now = [1000.0]
        speech = ActiveSpeech(15, 40, clock=lambda: now[0])
        task = asyncio.create_task(asyncio.sleep(10))
        speech.set(task)
        speech.note_audio_sent(0.2)
        assert speech.audible

        # 20 unbroken frames while the agent is still audible: over the quiet
        # bar of 15, under the audible bar of 40, so nothing is cancelled yet.
        for _ in range(20):
            assert speech.note_speech(True) is False
        assert not task.cancelled()

        # The clause finishes playing and the next one has not arrived - the
        # engine is a network call, so this gap is routine.
        now[0] += 1.0
        assert not speech.audible

        assert speech.note_speech(True) is True, "the caller could no longer interrupt"
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


def test_self_echo_is_recognised_but_a_real_caller_turn_is_not() -> None:
    agent = "நன்றி சார். எந்த department-க்கு வேணும்?"

    # The ASR mangles re-heard audio, so echo is a SUBSET of what was said.
    assert is_probably_self_echo("எந்த department-க்கு வேணும்", agent)
    assert is_probably_self_echo("நன்றி சார் எந்த department", agent)

    # THE ONE THAT MATTERS. Answering a question reuses the question's words -
    # that is what answering IS. Measured live: this exact turn was discarded
    # as echo, and the caller had to repeat themselves. The answer carries new
    # content ("Cardiology"); the echo above carries none.
    assert not is_probably_self_echo("Cardiology department வேணும் சார்", agent)
    assert not is_probably_self_echo("கார்தாலஜி department வேணும் சார்", agent)

    # A real answer shares a word or two and must survive.
    assert not is_probably_self_echo("எனக்கு Cardiology-ல appointment வேணும்", agent)
    assert not is_probably_self_echo("என் பேரு முருகேசன், வயசு 58", agent)

    # Never judge a short turn: refusing to hear a caller say yes is far worse
    # than occasionally acting on an echo.
    assert not is_probably_self_echo("ஆமாம்", agent)
    assert not is_probably_self_echo("சரி சார்", agent)

    # Nothing said yet, nothing to echo.
    assert not is_probably_self_echo("எந்த department-க்கு வேணும்", "")
