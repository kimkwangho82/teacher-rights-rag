import logging
import time

from fastapi import APIRouter, HTTPException

from api.schemas import ChatRequest, ChatResponse, Citation, ErrorResponse
from rag.chain import RagAnswer, answer_question

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])

SNIPPET_LEN = 200


def to_response(result: RagAnswer, latency_ms: int) -> ChatResponse:
    return ChatResponse(
        answer=result.answer,
        status=result.status,
        citations=[
            Citation(
                index=r.index,
                chunk_id=r.doc.metadata.get("chunk_id", ""),
                doc_id=r.doc.metadata.get("doc_id", ""),
                title=r.doc.metadata.get("title", ""),
                page=int(r.doc.metadata.get("page", 0)),
                snippet=r.doc.page_content[:SNIPPET_LEN],
                score=round(r.score, 4),
            )
            for r in result.citations
        ],
        model=result.model,
        latency_ms=latency_ms,
    )


@router.post("", response_model=ChatResponse, responses={502: {"model": ErrorResponse}})
def chat(req: ChatRequest) -> ChatResponse:
    start = time.perf_counter()
    try:
        result = answer_question(req.question, top_k=req.top_k)
    except Exception as exc:  # LLM / 벡터스토어 오류
        logger.exception("chat failed")
        raise HTTPException(status_code=502, detail=f"upstream error: {type(exc).__name__}") from exc
    return to_response(result, int((time.perf_counter() - start) * 1000))
