import json
import logging
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHUNKS_DIR, DATA_DIR
from utils.helpers import deterministic_uuid

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# --------------------------------------------------------------
# Prompt template (LLM‑side)
# --------------------------------------------------------------
PROMPT_TEMPLATE = """
You are given a technical excerpt from HashiCorp Terraform documentation.

Excerpt:
\"\"\"
{chunk}
\"\"\"

Based **only** on the excerpt, produce up to 5 distinct instruction‑answer pairs.
Each pair must be one of:
  • factual Q&A,
  • "Explain how this works",
  • "How do I perform X",
  • a failure scenario with a resolution,
  • an architectural explanation (if applicable).

If the excerpt does not contain enough information for a type, simply omit it.
Return a JSON array where each element has the keys:
  "instruction", "input" (always empty), "output".
Do NOT hallucinate any facts outside the provided text.
Return ONLY the JSON array, no other text.
"""

# --------------------------------------------------------------
# Provider configurations
# --------------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Model names
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.2")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250514")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gpt-oss:20b")


def _extract_json_array(text: str) -> List[Dict[str, str]]:
    """Extract JSON array from LLM response, handling markdown code blocks."""
    text = text.strip()

    # Remove markdown code blocks if present
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\n?', '', text)
        text = re.sub(r'\n?```$', '', text)
        text = text.strip()

    # Try to find JSON array in the text
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Try parsing the whole text
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    log.warning("Failed to parse JSON from LLM response")
    return []


def _call_openai(prompt: str) -> List[Dict[str, str]]:
    """Call OpenAI API."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_completion_tokens=2000,
        )

        content = response.choices[0].message.content
        return _extract_json_array(content)
    except Exception as e:
        log.warning(f"OpenAI call failed: {e}")
        raise


def _call_anthropic(prompt: str) -> List[Dict[str, str]]:
    """Call Anthropic API."""
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=ANTHROPIC_API_KEY)

        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.content[0].text
        return _extract_json_array(content)
    except Exception as e:
        log.warning(f"Anthropic call failed: {e}")
        raise


def _call_ollama(prompt: str) -> List[Dict[str, str]]:
    """Call local Ollama API."""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()

        content = response.json().get("response", "")
        return _extract_json_array(content)
    except Exception as e:
        log.warning(f"Ollama call failed: {e}")
        raise


def _check_ollama_available() -> bool:
    """Check if Ollama is running and has the required model."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            if OLLAMA_MODEL in model_names:
                return True
            log.info(f"Ollama running but model {OLLAMA_MODEL} not found")
        return False
    except:
        return False


def _get_available_providers() -> List[str]:
    """Return list of available LLM providers in priority order."""
    providers = []

    if OPENAI_API_KEY:
        providers.append("openai")
        log.info(f"OpenAI available (model: {OPENAI_MODEL})")

    if ANTHROPIC_API_KEY:
        providers.append("anthropic")
        log.info(f"Anthropic available (model: {ANTHROPIC_MODEL})")

    if _check_ollama_available():
        providers.append("ollama")
        log.info(f"Ollama available (model: {OLLAMA_MODEL})")

    return providers


def _call_llm(prompt: str, providers: List[str]) -> List[Dict[str, str]]:
    """Try calling LLM providers in order until one succeeds."""
    last_error = None

    for provider in providers:
        try:
            if provider == "openai":
                return _call_openai(prompt)
            elif provider == "anthropic":
                return _call_anthropic(prompt)
            elif provider == "ollama":
                return _call_ollama(prompt)
        except Exception as e:
            last_error = e
            log.warning(f"Provider {provider} failed, trying next...")
            continue

    if last_error:
        raise last_error
    raise RuntimeError("No LLM providers available")


def generate_supervision() -> None:
    """Iterate over chunk files, ask the LLM for supervision, and write JSONL."""
    providers = _get_available_providers()

    if not providers:
        log.error("No LLM providers available. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or run Ollama.")
        raise RuntimeError("No LLM providers configured")

    log.info(f"Using providers (in order): {providers}")

    chunk_files = sorted(CHUNKS_DIR.glob("*.json"))
    total_chunks = len(chunk_files)
    log.info(f"Processing {total_chunks} chunks")

    out_path = DATA_DIR / "supervision.jsonl"
    total_samples = 0
    processed = 0
    failed = 0

    with out_path.open("w", encoding="utf-8") as writer:
        for i, chunk_file in enumerate(chunk_files, 1):
            chunk = json.loads(chunk_file.read_text(encoding="utf-8"))
            prompt = PROMPT_TEMPLATE.format(chunk=chunk["text"])

            try:
                samples = _call_llm(prompt, providers)

                for s in samples:
                    # Validate required keys
                    if not all(k in s for k in ["instruction", "output"]):
                        continue
                    # Ensure input key exists
                    s.setdefault("input", "")
                    # Attach provenance
                    s["_meta"] = {
                        "doc_id": chunk["doc_id"],
                        "chunk_id": chunk["chunk_id"],
                        "source_url": chunk["url"],
                    }
                    writer.write(json.dumps(s, ensure_ascii=False) + "\n")
                    total_samples += 1

                processed += 1

                if i % 10 == 0 or i == total_chunks:
                    log.info(f"[{i}/{total_chunks}] Processed: {processed}, Failed: {failed}, Samples: {total_samples}")

            except Exception as e:
                failed += 1
                log.warning(f"[{i}/{total_chunks}] Failed to process chunk {chunk_file.name}: {e}")

    log.info("=" * 50)
    log.info("SUPERVISION GENERATION COMPLETE")
    log.info(f"  Chunks processed: {processed}/{total_chunks}")
    log.info(f"  Chunks failed: {failed}")
    log.info(f"  Total samples: {total_samples}")
    log.info(f"  Output: {out_path}")
    log.info("=" * 50)


if __name__ == "__main__":
    import logging

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s")
    generate_supervision()
