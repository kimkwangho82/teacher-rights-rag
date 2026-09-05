"""LLM-as-a-Judge 클라이언트.

- temperature 0 + seed 고정, JSON 구조화 출력 (파싱 실패 시 1회 재시도)
- 프롬프트는 eval/prompts/*.txt (string.Template, ${var}) → 파일 해시를 run_config 에 기록
- 디스크 캐시: key = sha256(model, seed, prompt 해시, 렌더된 입력, repeat 인덱스)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from string import Template

from langchain_openai import ChatOpenAI

from rag.config import settings

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class JudgeError(RuntimeError):
    pass


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


def prompt_hashes() -> dict[str, str]:
    return {
        p.stem: hashlib.sha256(p.read_bytes()).hexdigest()[:12]
        for p in sorted(PROMPTS_DIR.glob("*.txt"))
    }


def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_RE.search(text)
        if not m:
            raise JudgeError(f"no JSON object in judge output: {text[:200]!r}")
        return json.loads(m.group(0))


class Judge:
    def __init__(
        self,
        model: str | None = None,
        seed: int | None = None,
        cache_dir: Path | None = None,
        use_cache: bool = True,
        repeat_index: int = 0,
        llm=None,
    ):
        self.model = model or settings.judge_model
        self.seed = settings.seed if seed is None else seed
        self.cache_dir = cache_dir or settings.judge_cache_dir
        self.use_cache = use_cache
        self.repeat_index = repeat_index
        self._llm = llm
        self.calls = 0
        self.cache_hits = 0

    @property
    def llm(self):
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=self.model,
                temperature=0,
                seed=self.seed,
                api_key=settings.openai_api_key or None,
                base_url=settings.openai_base_url,
            )
        return self._llm

    def render(self, prompt_name: str, **vars) -> str:
        return Template(load_prompt(prompt_name)).safe_substitute(**vars)

    def _cache_key(self, prompt_name: str, rendered: str) -> str:
        h = hashlib.sha256()
        for part in (
            self.model,
            str(self.seed),
            prompt_hashes()[prompt_name],
            rendered,
            str(self.repeat_index),
        ):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def ask(self, prompt_name: str, **vars) -> dict:
        rendered = self.render(prompt_name, **vars)
        key = self._cache_key(prompt_name, rendered)
        path = self.cache_dir / f"{key}.json"
        if self.use_cache and path.exists():
            self.cache_hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["output"]

        output = None
        last_err: Exception | None = None
        for attempt in range(2):
            self.calls += 1
            raw = self.llm.invoke([{"role": "user", "content": rendered}]).content or ""
            try:
                output = parse_json(raw)
                break
            except (JudgeError, json.JSONDecodeError) as e:
                last_err = e
                logger.warning(
                    "judge JSON parse failed (attempt %d) for %s: %s",
                    attempt + 1,
                    prompt_name,
                    e,
                )
        if output is None:
            raise JudgeError(f"{prompt_name}: {last_err}")

        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "prompt": prompt_name,
                        "model": self.model,
                        "seed": self.seed,
                        "output": output,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        return output
