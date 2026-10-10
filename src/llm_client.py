"""Thin client for the local Ollama server — the proposal's ethics section
commits to a local open-source model rather than sending log excerpts to an
external provider, so this is the only LLM path Phase 7 uses.
"""
import json
import re

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"


def generate(prompt: str, temperature: float = 0.1, timeout: int = 120) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def extract_json(text: str) -> dict:
    """LLMs reliably wrap JSON in prose/code fences even when told not to.
    Grab the first {...} block and parse it; raise clearly if nothing
    parses, so a bad generation surfaces as an explicit failure rather
    than a silently wrong gate decision.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object found in LLM response: {text!r}")
    return json.loads(match.group(0))
