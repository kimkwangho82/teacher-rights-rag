"""질의 → 검색 → (abstain 판단) → 생성 → Citation 추출.

Hallucination 방지 2단계:
  1) 점수 기반: 최고 관련도가 similarity_threshold 미만이면 LLM 을 호출하지 않고 insufficient 반환.
     - 결정적이며 비용이 들지 않는다. 명백히 무관한 질의(날씨, 잡담)를 거른다.
  2) 프롬프트 기반: 문맥이 검색됐더라도 답을 구성할 수 없으면 모델이 INSUFFICIENT_CONTEXT 만 출력하도록 지시.
     - 주제는 관련 있으나 문서에 답이 없는 질의를 거른다.
"""

import logging
import re
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from rag.config import settings
from rag.evidence_prompt import SYSTEM_PROMPT_EVIDENCE_FIRST, parse_evidence_output
from rag.retriever import retrieve_with_gate

logger = logging.getLogger(__name__)

INSUFFICIENT_TOKEN = "INSUFFICIENT_CONTEXT"
INSUFFICIENT_MESSAGE = (
    "제공된 문서에서 질문에 대한 근거를 찾지 못해 답변할 수 없습니다."
)

SYSTEM_PROMPT = f"""당신은 교권(교사의 교육활동 보호) 관련 질문에 답하는 도우미입니다.
반드시 아래 [참고 자료]만 근거로 답하세요. 사전 지식이나 추측으로 내용을 보충하지 마세요.

규칙:
1. 답변의 각 문장 끝에 근거가 된 자료 번호를 [n] 형식으로 표시하세요. 여러 자료면 [1][3] 처럼 나열합니다.
2. 참고 자료로 질문에 답할 수 없거나 근거가 부족하면, 다른 말 없이 정확히 {INSUFFICIENT_TOKEN} 만 출력하세요.
3. 자료에 있는 절차·기준·수치는 그대로 인용하고, 없는 내용은 만들어내지 마세요.
4. 한국어로 간결하게 답하세요.

[참고 자료]
{{context}}"""

_CITE_RE = re.compile(r"\[(\d+)\]")


@dataclass
class Retrieved:
    index: int  # 프롬프트에 표시된 번호 (1-based)
    doc: Document
    score: float


@dataclass
class RagAnswer:
    answer: str
    status: str  # "answered" | "insufficient"
    citations: list[Retrieved] = field(default_factory=list)
    retrieved: list[Retrieved] = field(default_factory=list)
    model: str = settings.chat_model
    llm_called: bool = False
    quotes: list[dict] = field(default_factory=list)  # evidence_first 모드의 원문 인용
    gate_score: float = 0.0  # abstain 게이트에 쓴 dense 최고 점수


def format_context(items: list[Retrieved]) -> str:
    blocks = []
    for r in items:
        m = r.doc.metadata
        header = f"[{r.index}] ({m.get('title', m.get('source', ''))}, p.{m.get('page', '?')})"
        blocks.append(f"{header}\n{r.doc.page_content}")
    return "\n\n".join(blocks)


def extract_citations(answer: str, items: list[Retrieved]) -> list[Retrieved]:
    """답변에 등장한 [n] 만 골라 등장 순서대로 반환. 하나도 없으면 검색 결과 전체를 반환한다."""
    by_index = {r.index: r for r in items}
    seen: list[int] = []
    for n in _CITE_RE.findall(answer):
        i = int(n)
        if i in by_index and i not in seen:
            seen.append(i)
    return [by_index[i] for i in seen] if seen else list(items)


def get_chat_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.chat_model,
        temperature=settings.temperature,
        seed=settings.seed,
        api_key=settings.openai_api_key or None,
        base_url=settings.openai_base_url,
    )


def answer_question(question: str, top_k: int | None = None) -> RagAnswer:
    results, gate_score = retrieve_with_gate(question, k=top_k)
    items = [Retrieved(index=i + 1, doc=d, score=s) for i, (d, s) in enumerate(results)]

    # 1) 점수 기반 abstain (dense 최고 점수 기준 — hybrid 에서도 동일한 게이트)
    if not items or gate_score < settings.similarity_threshold:
        return RagAnswer(
            answer=INSUFFICIENT_MESSAGE,
            status="insufficient",
            retrieved=items,
            gate_score=gate_score,
        )

    context = format_context(items)
    if settings.prompt_mode == "evidence_first":
        return _answer_evidence_first(question, items, context, gate_score)
    return _answer_baseline(question, items, context, gate_score)


def _invoke(system_prompt: str, question: str) -> str:
    response = get_chat_model().invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
    )
    return (response.content or "").strip()


def _answer_baseline(
    question: str, items: list[Retrieved], context: str, gate_score: float
) -> RagAnswer:
    text = _invoke(SYSTEM_PROMPT.format(context=context), question)

    # 2) 프롬프트 기반 abstain
    if INSUFFICIENT_TOKEN in text and len(text) <= len(INSUFFICIENT_TOKEN) + 10:
        return RagAnswer(
            answer=INSUFFICIENT_MESSAGE,
            status="insufficient",
            retrieved=items,
            llm_called=True,
            gate_score=gate_score,
        )

    return RagAnswer(
        answer=text,
        status="answered",
        citations=extract_citations(text, items),
        retrieved=items,
        llm_called=True,
        gate_score=gate_score,
    )


def _answer_evidence_first(
    question: str, items: list[Retrieved], context: str, gate_score: float
) -> RagAnswer:
    """개선 실험 2: 원문 인용을 먼저 추출하고 인용에 근거해서만 답한다."""
    text = _invoke(SYSTEM_PROMPT_EVIDENCE_FIRST.format(context=context), question)
    out = parse_evidence_output(text)

    if not out.parsed:
        logger.warning("evidence_first: JSON 파싱 실패 → 원문 텍스트로 폴백")
        return RagAnswer(
            answer=out.answer,
            status="answered",
            citations=extract_citations(out.answer, items),
            retrieved=items,
            llm_called=True,
            gate_score=gate_score,
        )

    # 2) 인용이 없거나 모델이 insufficient 로 표시 → abstain
    if out.insufficient:
        return RagAnswer(
            answer=INSUFFICIENT_MESSAGE,
            status="insufficient",
            retrieved=items,
            llm_called=True,
            quotes=out.quotes,
            gate_score=gate_score,
        )

    # 답변에 [n] 이 없으면 인용 출처를 citation 으로 사용
    citations = extract_citations(out.answer, items)
    if not _CITE_RE.search(out.answer):
        by_index = {r.index: r for r in items}
        sources = [q["source"] for q in out.quotes if q["source"] in by_index]
        if sources:
            citations = [by_index[i] for i in dict.fromkeys(sources)]

    return RagAnswer(
        answer=out.answer,
        status="answered",
        citations=citations,
        retrieved=items,
        llm_called=True,
        quotes=out.quotes,
        gate_score=gate_score,
    )
