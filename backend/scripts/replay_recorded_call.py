"""Replay call 97dd5ac7 through the real ConversationManager + real Ollama,
reproducing the exact barge-in pattern (how many clauses actually got out
before the caller cut in), so a candidate fix can be measured against the call
it is meant to fix rather than against a guess.

    LLM_TEMPERATURE=0 PYTHONPATH=. .venv/Scripts/python.exe replay.py
"""
from __future__ import annotations

import asyncio
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backend.conversation import AgentClause, ConversationManager
from backend.llm import LlmClient
from backend.settings import ConversationSettings, LlmSettings

# (caller transcript, clauses that got out before barge-in; None = ran to completion)
CALL = [
    ("ஹெல்லோ", 0),
    ("எனக்கு ஒரு appointment book பண்ணனும்", None),
    ("ஆஸ்டோ department க்கு வேணும்", 0),
    ("ஆற்று", None),
    ("என் பேர் நானே", None),
    ("இருபது", None),
    ("98407", None),
    ("21534", 2),
    ("வெண்ணூர் கிழவனை", 0),
    ("என்னாமோ வளர்ந்து", 0),
    ("இந்த போன் number பாக்க தெரியுமா ஆட்டோ ல", 0),
    ("நான் number சொல்ல உதவிகள்", 0),
    ("number லாம் வந்தது நமக்கு தான் மாத்தி நம்ம", None),
    ("நல்லா பேசிருக்க", 0),
    ("எல்லாம்", 0),
    ("ஸ்பீட் டூ டேஸெல்லாம் காரெக்ட் ஆ இருக்கு", None),
    ("எவ்வளவுதான் கொஞ்சம் எவ்வளவு", 0),
    ("தேடியா", None),
    ("அதான் இடம் இல்லாம கொஞ்சம் கொலர் தான் இருக்கு", 0),
    ("நான் பாத்து", 0),
    ("என்னக்கா முடியாது நான் பேசுறது வர", 0),
    ("பேசினா தகுந்தா கம்மியா தான் இருக்கு ஆங்களோட பாத்துக்கிட்டே", None),
]


async def main() -> None:
    convo = ConversationManager(ConversationSettings())
    convo.load()
    llm = LlmClient(LlmSettings())
    llm.load()

    cid = "replay"
    greeting = convo.start_call(cid, agent_name="Gayathri")
    print(f"AGENT   {greeting}\n")

    session = convo._sessions[cid]
    turns: list[str] = []
    first_clause_times: list[float] = []

    for text, cut in CALL:
        print(f"CALLER  {text}")
        spoken: list[str] = []
        started = time.perf_counter()
        first = None
        agen = convo.stream_utterance(cid, llm, text)
        try:
            async for event in agen:
                if not isinstance(event, AgentClause):
                    continue
                # cut == 0 means the caller cut in before ANY audio got out, so
                # the clause is dropped rather than counted: that is exactly the
                # case record_interrupted_turn() has nothing to append for.
                if cut is not None and len(spoken) >= cut:
                    break
                if first is None:
                    first = time.perf_counter() - started
                spoken.append(event.text)
        finally:
            await agen.aclose()

        if cut is not None:
            convo.record_interrupted_turn(cid, " ".join(spoken))
            tail = " [CUT]"
        else:
            tail = ""
        if first is not None:
            first_clause_times.append(first)
        line = " ".join(spoken)
        turns.append(line)
        print(f"AGENT   {line or '(nothing)'}{tail}"
              + (f"   [{first:.1f}s]" if first is not None else ""))
        print()

    runs = sum(
        1
        for a, b in zip(session.messages, session.messages[1:])
        if a.get("role") == b.get("role") == "user"
    )
    said = [t for t in turns if t]
    print("=" * 70)
    print(f"consecutive-user pairs left in history : {runs}")
    print(f"distinct agent turns                   : {len(set(said))} of {len(said)}")
    if first_clause_times:
        mid = sorted(first_clause_times)[len(first_clause_times) // 2]
        print(f"median first clause                    : {mid:.2f}s")


asyncio.run(main())
