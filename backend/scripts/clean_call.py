"""A realistic, clean booking call - no barge-in, no ASR noise - through the
real ConversationManager and the real Ollama. This is the product: what a
caller who is simply talking to the agent actually gets, and how fast.

    LLM_TEMPERATURE=0 PYTHONPATH=. .venv/Scripts/python.exe clean_call.py
"""
from __future__ import annotations

import asyncio
import statistics
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backend.conversation import AgentClause, ConversationManager
from backend.llm import LlmClient
from backend.settings import ConversationSettings, LlmSettings

CALLS = {
    "appointment (caller opens with hello)": [
        "ஹெல்லோ",
        "எனக்கு ஒரு appointment book பண்ணனும்",
        "Orthopaedics department",
        "என் பேரு முருகேசன்",
        "வயசு 58",
        "98407 21534",
        "நாளைக்கு காலைல வர முடியும்",
    ],
    "lab report chase": [
        "நேத்து blood test பண்ணேன், report வந்துடுச்சா?",
        "என் பேரு கமலா, மேடம்",
        "90045 33218",
    ],
    "billing complaint": [
        "பில்-ல ஒரு charge தப்பா இருக்கு",
        "bill number ARV-4471",
        "consultation fee ரெண்டு தடவை போட்டிருக்காங்க",
    ],
}


async def main() -> None:
    convo = ConversationManager(ConversationSettings())
    convo.load()
    llm = LlmClient(LlmSettings())
    llm.load()

    all_first: list[float] = []
    for name, turns in CALLS.items():
        print("=" * 72)
        print(name)
        print("=" * 72)
        cid = name
        print(f"AGENT   {convo.start_call(cid, agent_name='Gayathri')}\n")
        for text in turns:
            print(f"CALLER  {text}")
            started = time.perf_counter()
            first = None
            spoken = []
            async for event in convo.stream_utterance(cid, llm, text):
                if isinstance(event, AgentClause):
                    if first is None:
                        first = time.perf_counter() - started
                    spoken.append(event.text)
            whole = time.perf_counter() - started
            if first is not None:
                all_first.append(first)
            print(f"AGENT   {' '.join(spoken)}")
            print(f"        [first {first:.2f}s · turn {whole:.2f}s]\n")
        convo.end_call(cid)

    print("=" * 72)
    print(f"first clause: median {statistics.median(all_first):.2f}s  "
          f"min {min(all_first):.2f}s  max {max(all_first):.2f}s  (n={len(all_first)})")
    over = [t for t in all_first if t > 2.0]
    print(f"turns over the 2s target: {len(over)} of {len(all_first)}")


asyncio.run(main())
