import logging

from fastapi import FastAPI

from api.routers import chat

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(
    title="Teacher Rights RAG API",
    description="교육활동 보호 매뉴얼 기반 Citation RAG QA 서비스",
    version="0.1.0",
)
app.include_router(chat.router)


@app.get("/health", tags=["ops"])
def health() -> dict:
    return {"status": "ok"}
