from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, description="사용자 질의")
    top_k: int | None = Field(default=None, ge=1, le=20, description="검색할 청크 수 (기본: 설정값)")

    model_config = {"json_schema_extra": {"examples": [{"question": "교육활동 침해 사안 발생 시 학교장의 조치 절차는?"}]}}


class Citation(BaseModel):
    index: int = Field(description="답변 본문의 [n] 번호")
    chunk_id: str
    doc_id: str
    title: str
    page: int
    snippet: str = Field(description="근거 청크 앞부분 (최대 200자)")
    score: float = Field(description="cosine relevance (0~1)")


class ChatResponse(BaseModel):
    answer: str
    status: Literal["answered", "insufficient"] = Field(
        description="answered: 문서 근거로 답변 / insufficient: 근거 부족으로 답변 불가"
    )
    citations: list[Citation]
    model: str
    latency_ms: int

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "answer": "학교장은 교육활동 침해 사안을 인지한 즉시 피해 교원을 보호하고 ... [1] 이후 교권보호위원회에 ... [2]",
                    "status": "answered",
                    "citations": [
                        {
                            "index": 1,
                            "chunk_id": "2025-교육활동-보호-매뉴얼-배포용:p45:c0",
                            "doc_id": "2025-교육활동-보호-매뉴얼-배포용",
                            "title": "2025 교육활동 보호 매뉴얼",
                            "page": 45,
                            "snippet": "학교장은 교육활동 침해행위를 인지한 경우 ...",
                            "score": 0.62,
                        }
                    ],
                    "model": "gpt-4.1-mini",
                    "latency_ms": 1830,
                }
            ]
        }
    }


class ErrorResponse(BaseModel):
    detail: str
