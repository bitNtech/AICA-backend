"""num_ctx sweep against the REAL assembled prompt (LLM_STACK.md Sec7: a short
prompt lies about this box). Reports prompt-eval and generation rates plus the
CPU/GPU split Ollama chose, so the effect of a smaller KV cache on the OFFLOAD
- not just on the cache - is visible.

    PYTHONPATH=. .venv/Scripts/python.exe bench_ctx.py 8192 6144 5120 4096
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backend.conversation import _with_language_reminder, render_template
from backend.prompt_builder import PromptBuilder
from backend.settings import ConversationSettings

OLLAMA = "http://127.0.0.1:11434"

cs = ConversationSettings()
pb = PromptBuilder(cs.runtime_core_path, cs.prompt_path, cs.exemplars_path)
pb.load()
SYSTEM = render_template(pb.build("appointment.book"), {"agent_name": "Gayathri"})

# A real mid-call history, so prompt length matches what a live turn sends.
HISTORY = [
    {"role": "system", "content": SYSTEM},
    {"role": "assistant", "content": "வணக்கம், அருவி ஹாஸ்பிட்டல். நான் Gayathri பேசுறேன். உங்களுக்கு எப்படி help பண்ணலாம்?"},
    {"role": "user", "content": "எனக்கு ஒரு appointment book பண்ணனும்"},
    {"role": "assistant", "content": "கண்டிப்பா சார். எந்த department-க்கு வேணும்?"},
    {"role": "user", "content": "Orthopaedics department"},
    {"role": "assistant", "content": "சரி சார். உங்க பேரு சொல்லுங்க?"},
    {"role": "user", "content": "என் பேரு முருகேசன், வயசு 58"},
    {"role": "assistant", "content": "நன்றி முருகேசன் சார். உங்க registered mobile number ஒரு தடவை சொல்லுங்களா?"},
    {"role": "user", "content": "98407 21534"},
]


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"{OLLAMA}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def split() -> str:
    out = subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("aruvi-base"):
            return " ".join(line.split()[3:5])
    return "?"


def run(num_ctx: int, num_gpu: int | None = None) -> None:
    subprocess.run(["ollama", "stop", "aruvi-base"], capture_output=True)
    time.sleep(3)
    messages = _with_language_reminder(HISTORY)
    options = {"num_ctx": num_ctx, "temperature": 0}
    if num_gpu is not None:
        options["num_gpu"] = num_gpu
    body = {
        "model": "aruvi-base",
        "messages": messages,
        "stream": False,
        "options": options,
    }
    post("/api/chat", body)          # load + warm the prefix
    d = post("/api/chat", body)      # measure warm

    pe, ped = d["prompt_eval_count"], d["prompt_eval_duration"] / 1e9
    ec, ed = d["eval_count"], d["eval_duration"] / 1e9
    print(
        f"num_ctx {num_ctx:>5} gpu {str(num_gpu):>4} | split {split():<18} | "
        f"prompt {pe:>5} tok in {ped:>5.2f}s | "
        f"gen {ec:>4} tok at {ec/ed:>5.1f} tok/s | "
        f"total {d['total_duration']/1e9:>5.2f}s"
    )
    print(f"              reply: {d['message']['content'][:100]}")


for arg in sys.argv[1:]:
    ctx, _, gpu = arg.partition(":")
    run(int(ctx), int(gpu) if gpu else None)
