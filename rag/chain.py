"""질의 → 검색 → (abstain 판단) → 생성 → Citation 추출.

Hallucination 방지 2단계:
  1) 점수 기반: 최고 관련도가 similarity_threshold 미만이면 LLM 을 호출하지 않고 insufficient 반환.
     - 결정적이며 비용이 들지 않는다. 명백히 무관한 질의(날씨, 잡담)를 거른다.
  2) 프롬프트 기반: 문맥이 검색됐더라도 답을 구성할 수 없으면 모델이 INSUFFICIENT_CONTEXT 만 출력하도록 지시.
     - 주제는 관련 있으나 문서에 답이 없는 질의를 거른다.
"""

import re
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from rag.config import settings
from rag.retriever import retrieve

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
    results = retrieve(question, k=top_k)
    items = [Retrieved(index=i + 1, doc=d, score=s) for i, (d, s) in enumerate(results)]

    # 1) 점수 기반 abstain
    if not items or items[0].score < settings.similarity_threshold:
        return RagAnswer(
            answer=INSUFFICIENT_MESSAGE, status="insufficient", retrieved=items
        )

    response = get_chat_model().invoke(
        [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(context=format_context(items)),
            },
            {"role": "user", "content": question},
        ]
    )
    text = (response.content or "").strip()

    # 2) 프롬프트 기반 abstain
    if INSUFFICIENT_TOKEN in text and len(text) <= len(INSUFFICIENT_TOKEN) + 10:
        return RagAnswer(
            answer=INSUFFICIENT_MESSAGE,
            status="insufficient",
            retrieved=items,
            llm_called=True,
        )

    return RagAnswer(
        answer=text,
        status="answered",
        citations=extract_citations(text, items),
        retrieved=items,
        llm_called=True,
    )
