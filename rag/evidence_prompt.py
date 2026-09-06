"""Evidence-first 프롬프트 (개선 실험 2): 원문 인용을 먼저 추출하고 인용에 근거해서만 답한다.

출력은 JSON 하나:
{"quotes": [{"source": 1, "text": "문맥 원문 그대로"}], "answer": "…[1]…", "insufficient": false}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from rag.jsonutil import JSONExtractError, extract_json

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_EVIDENCE_FIRST = """당신은 교권(교사의 교육활동 보호) 관련 질문에 답하는 도우미입니다.
반드시 아래 [참고 자료]만 근거로 답하세요. 사전 지식이나 추측으로 내용을 보충하지 마세요.

작업 순서:
1. 자료 [1]부터 마지막까지 **모두** 읽고, 질문에 답하는 데 직접 근거가 되는 문장을 자료 원문 그대로 발췌해 quotes 에 넣으세요.
   - 여러 자료에 근거가 나뉘어 있으면 각각 발췌합니다. 요약하거나 표현을 바꾸지 마세요.
   - 질문이 묻는 세부 정보(기간, 인원, 금액, 절차 등)가 자료에 없으면 발췌하지 마세요.
2. quotes 에 담긴 내용만으로 답변(answer)을 작성하세요. 각 문장 끝에 근거 자료 번호를 [n] 으로 표시합니다.
   - quotes 에 없는 내용은 답변에 쓰지 마세요.
3. 질문에 답할 근거가 되는 quotes 가 하나도 없으면 insufficient 를 true 로 두고 answer 는 빈 문자열로 두세요.
   자료가 주제와 관련 있어 보여도, 질문이 묻는 구체적 내용이 없으면 insufficient 입니다.

출력 형식 (JSON 만 출력, 다른 텍스트 금지):
{{"quotes": [{{"source": 1, "text": "자료 원문 발췌"}}], "answer": "한국어 답변 … [1]", "insufficient": false}}

[참고 자료]
{context}"""


@dataclass
class EvidenceOutput:
    answer: str
    quotes: list[dict] = field(default_factory=list)
    insufficient: bool = False
    parsed: bool = True  # False 면 JSON 파싱 실패 → 원문을 answer 로 폴백


def parse_evidence_output(text: str) -> EvidenceOutput:
    text = (text or "").strip()
    try:
        obj = extract_json(text)
    except JSONExtractError as e:
        logger.warning(
            "evidence_first output is not JSON, falling back to raw text: %s", e
        )
        return EvidenceOutput(answer=text, parsed=False)

    quotes_raw = obj.get("quotes") or []
    quotes = []
    for q in quotes_raw:
        if isinstance(q, dict) and str(q.get("text", "")).strip():
            try:
                source = int(q.get("source", 0))
            except (TypeError, ValueError):
                source = 0
            quotes.append({"source": source, "text": str(q["text"]).strip()})
    answer = str(obj.get("answer") or "").strip()
    insufficient = bool(obj.get("insufficient")) or not quotes or not answer
    return EvidenceOutput(answer=answer, quotes=quotes, insufficient=insufficient)
