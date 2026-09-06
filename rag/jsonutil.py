"""LLM 출력에서 JSON 객체를 추출한다 (코드펜스·앞뒤 설명문 허용)."""

from __future__ import annotations

import json
import re

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.DOTALL)
_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


class JSONExtractError(ValueError):
    pass


def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = _FENCE_RE.sub("", text).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = _OBJ_RE.search(text)
        if not m:
            raise JSONExtractError(
                f"no JSON object in output: {text[:200]!r}"
            ) from None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            raise JSONExtractError(f"invalid JSON: {e}") from None
    if not isinstance(obj, dict):
        raise JSONExtractError("top-level JSON is not an object")
    return obj
