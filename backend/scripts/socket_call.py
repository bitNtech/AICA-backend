"""A whole call over the real /ws/audio socket: every turn, time to first TEXT
and to first AUDIO. This is what the caller actually experiences - the LLM
number alone hides both TTS and the server path.
"""
from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time

import websockets

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TURNS = [
    "எனக்கு ஒரு appointment book பண்ணனும்",
    "Orthopaedics department",
    "என் பேரு முருகேசன்",
    "வயசு 58",
    "98407 21534",
    "நாளைக்கு காலைல வர முடியும்",
]


async def main() -> None:
    async with websockets.connect("ws://127.0.0.1:8000/ws/audio", max_size=None) as ws:
        await ws.send(json.dumps({
            "type": "call_started", "audio_format": "pcm_s16le",
            "sample_rate": 16000, "channels": 1, "language": "ta",
        }))
        started = time.perf_counter()
        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=120)
            if isinstance(msg, str) and json.loads(msg).get("type") == "agent_speaking_end":
                break
        print(f"greeting delivered in {time.perf_counter() - started:.2f}s\n")
        print(f"{'caller':<34} {'text':>7} {'audio':>7}  reply")
        print("-" * 96)

        first_audio: list[float] = []
        for turn in TURNS:
            await ws.send(json.dumps({"type": "user_text", "text": turn}))
            t0 = time.perf_counter()
            t_text = t_audio = None
            said = []
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=180)
                now = time.perf_counter() - t0
                if isinstance(msg, bytes):
                    if t_audio is None:
                        t_audio = now
                    continue
                event = json.loads(msg)
                if event.get("type") == "agent_clause":
                    if t_text is None:
                        t_text = now
                    said.append(event["text"])
                elif event.get("type") in ("agent_speaking_end", "agent_interrupted"):
                    break
            if t_audio is not None:
                first_audio.append(t_audio)
            print(f"{turn[:32]:<34} {t_text or float('nan'):>7.2f} "
                  f"{t_audio or float('nan'):>7.2f}  {' '.join(said)[:52]}")

        print("-" * 96)
        print(f"first AUDIO: median {statistics.median(first_audio):.2f}s  "
              f"min {min(first_audio):.2f}s  max {max(first_audio):.2f}s")
        print(f"turns over 2s: {sum(t > 2.0 for t in first_audio)} of {len(first_audio)}")


asyncio.run(main())
