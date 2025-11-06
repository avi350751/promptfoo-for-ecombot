import os
import re
import json
import math
from dataclasses import dataclass
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()

from bs4 import BeautifulSoup

# OpenAI-compatible client (works with OpenAI, Azure OpenAI w/ base URL, or any proxy)
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # handled at call time


# -----------------------
# Data structures
# -----------------------

@dataclass
class ResultBundle:
    summary_md: Optional[str] = None
    notes_md: Optional[str] = None

    def to_markdown(self) -> str:
        parts: List[str] = []
        if self.summary_md:
            parts.append("# Summary\n\n" + self.summary_md.strip())
        if self.notes_md:
            parts.append("# Notes for You\n\n" + self.notes_md.strip())
        return "\n\n---\n\n".join(parts).strip()


# -----------------------
# Parsers
# -----------------------

def extract_text_from_html(file_bytes: bytes) -> str:
    soup = BeautifulSoup(file_bytes, "html.parser")
    # Remove scripts/styles
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    text = soup.get_text("\n")
    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def extract_text_from_json(file_bytes: bytes) -> str:
    obj = json.loads(file_bytes.decode("utf-8"))
    """
    Generic, robust JSON → text:
    - If this is a Promptfoo/DeepEval-like result, we’ll still produce readable text.
    - We avoid dumping massive arrays verbatim; instead, we produce a summarized text view
      for large lists of tests/assertions while preserving key fields.
    """
    def _flatten(o, path=""):
        lines = []
        if isinstance(o, dict):
            for k, v in o.items():
                key = f"{path}.{k}" if path else k
                if isinstance(v, (dict, list)):
                    lines.append(f"[{key}]")
                    lines.extend(_flatten(v, key))
                else:
                    lines.append(f"{key}: {v}")
        elif isinstance(o, list):
            max_items_preview = 30  # be sane for huge results
            lines.append(f"(list length: {len(o)})")
            for i, item in enumerate(o[:max_items_preview]):
                lines.append(f"[{path}[{i}]]")
                lines.extend(_flatten(item, f"{path}[{i}]"))
            if len(o) > max_items_preview:
                lines.append(f"... ({len(o) - max_items_preview} more items omitted)")
        else:
            lines.append(f"{path}: {o}")
        return lines

    lines = _flatten(obj)
    text = "\n".join(lines)
    return text.strip()


# -----------------------
# Token helpers
# -----------------------

def approx_token_count(s: str) -> int:
    # Very rough: ~4 chars per token heuristic
    return max(1, math.ceil(len(s) / 4))


def chunk_text_by_tokens(s: str, max_tokens: int = 6000) -> List[str]:
    # Lightweight tokenizer by whitespace, pack until ~max_tokens
    words = s.split()
    chunks, cur, cur_tokens = [], [], 0
    for w in words:
        t = max(1, math.ceil(len(w) / 4))
        if cur_tokens + t > max_tokens and cur:
            chunks.append(" ".join(cur))
            cur, cur_tokens = [], 0
        cur.append(w)
        cur_tokens += t
    if cur:
        chunks.append(" ".join(cur))
    return chunks


# -----------------------
# LLM calls
# -----------------------

def _get_openai_client():
    if OpenAI is None:
        raise RuntimeError(
            "openai package not installed. Add `openai` to requirements and `pip install -r requirements.txt`."
        )
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment.")
    base_url = os.getenv("OPENAI_BASE_URL")  # optional (for proxies/Azure-compatible endpoints)
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


SYSTEM_PROMPT = """You are a precise AI assistant that turns raw evaluation outputs (HTML or JSON-derived text)
into actionable, **concise** insights for engineers and QA leaders.

If asked for a SUMMARY:
- Start with a 3–5 line executive overview.
- Then include sections for: Key Findings, Pass/Fail/Deviations, Regressions, and Action Items.
- Call out safety or schema violations prominently.

If asked for NOTES:
- Produce a crisp bullet list of practical, high-signal items someone should keep in mind.
- Prefer imperative phrasing (“Verify …”, “Add guardrails for …”).
- Keep bullets short; no fluff.

General rules:
- Be faithful to the provided content only.
- Use Markdown. Avoid code blocks unless necessary.
- Don’t invent metrics; if unknown, say so clearly.
- If content looks like Promptfoo or DeepEval, surface: failing tests, rubrics, eval scores, and examples.
"""

def _build_user_prompt(chunk_text: str, mode: str, desired_length: str,
                       notes_bullets: int, extra_instructions: str) -> str:
    add = f"\nExtra instructions: {extra_instructions}" if extra_instructions else ""
    if mode == "Summarize the content":
        goal = f"Write a {desired_length} summary."
    elif mode == "Notes for you":
        goal = f"Write exactly {notes_bullets} concise bullets of notes."
    else:
        goal = f"Write a {desired_length} summary **and** {notes_bullets} concise bullets of notes."

    return f"""{goal}
Source content:
{chunk_text}

{add}
"""

def _chat_complete(client, model: str, messages: list, temperature: float) -> str:
    # OpenAI Chat Completions (compatible with many providers)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()

def _map_reduce_summary(raw_text: str, mode: str, desired_length: str,
                        notes_bullets: int, temperature: float, extra_instructions: str) -> ResultBundle:
    client = _get_openai_client()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    chunks = chunk_text_by_tokens(raw_text, max_tokens=6000)
    partials: List[str] = []

    # Map step
    for i, c in enumerate(chunks, 1):
        uprompt = _build_user_prompt(c, mode, desired_length, notes_bullets, extra_instructions)
        content = _chat_complete(
            client,
            model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": uprompt},
            ],
            temperature=temperature,
        )
        partials.append(f"### Part {i}\n\n{content}")

    # Reduce step
    reduce_prompt = f"""Combine these partial analyses into a single cohesive output.
Maintain the requested mode: "{mode}". Respect length and bullet constraints.

PARTIALS:
{"\n\n".join(partials)}
"""
    combined = _chat_complete(
        client,
        model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": reduce_prompt},
        ],
        temperature=temperature,
    )

    # Split into summary vs notes (best-effort)
    summary_md, notes_md = None, None
    if mode == "Summarize the content":
        summary_md = combined
    elif mode == "Notes for you":
        notes_md = combined
    else:
        # Try to split if user asked for both
        # Heuristic: headings or bullet sections
        parts = re.split(r"(?i)^#*\s*(notes|notes for you)\s*[:\-]*\s*$", combined, maxsplit=1, flags=re.MULTILINE)
        if len(parts) >= 3:
            # parts = [before, 'notes', after]
            summary_md = parts[0].strip()
            notes_md = parts[2].strip()
        else:
            # fallback: attempt bullet extraction
            bullets = re.findall(r"(?m)^\s*[-*]\s+.+$", combined)
            if bullets:
                notes_md = "\n".join(bullets)
                summary_md = combined
            else:
                summary_md = combined  # give everything; better than dropping info

    return ResultBundle(summary_md=summary_md, notes_md=notes_md)


def summarize_with_llm(raw_text: str, mode: str, desired_length: str = "medium",
                       notes_bullets: int = 7, temperature: float = 0.2,
                       extra_instructions: str = "") -> ResultBundle:
    """
    Main entrypoint for the app. Handles long input via map-reduce chunking.
    """
    return _map_reduce_summary(
        raw_text=raw_text,
        mode=mode,
        desired_length=desired_length,
        notes_bullets=notes_bullets,
        temperature=temperature,
        extra_instructions=extra_instructions,
    )
