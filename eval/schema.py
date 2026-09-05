"""Eval Harness 데이터 모델."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

QuestionType = Literal[
    "factual",
    "procedural",
    "multi_evidence",
    "summary",
    "reasoning",
    "unanswerable_out",
    "unanswerable_in",
]


class Criterion(BaseModel):
    text: str
    required: bool = False


class GoldItem(BaseModel):
    id: str
    type: QuestionType
    question: str
    answerable: bool
    evidence_pages: list[int] = Field(default_factory=list)
    evidence_quotes: list[str] = Field(default_factory=list)
    acceptance_criteria: list[Criterion] = Field(default_factory=list)
    notes: str = ""


class RetrievedChunk(BaseModel):
    index: int
    chunk_id: str
    page: int
    score: float
    text: str


class Statement(BaseModel):
    text: str
    supported: bool | None = None
    reason: str = ""


class CriterionResult(BaseModel):
    text: str
    required: bool
    passed: bool | None = None
    reason: str = ""


class JudgeScores(BaseModel):
    """LLM Judge 지표. None 은 해당 없음(abstain 등) 또는 judge 미실행."""

    faithfulness: float | None = None
    statements: list[Statement] = Field(default_factory=list)
    context_relevance: float | None = None
    context_sentences_total: int | None = None
    context_sentences_relevant: int | None = None
    correctness: float | None = None
    criteria: list[CriterionResult] = Field(default_factory=list)
    answer_relevance: float | None = None
    generated_questions: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class ItemResult(BaseModel):
    id: str
    type: QuestionType
    question: str
    answerable: bool
    answer: str
    status: Literal["answered", "insufficient"]
    llm_called: bool
    retrieved: list[RetrievedChunk]
    cited_indices: list[int]
    evidence_pages: list[int]
    recall_at_k: float | None = None
    mrr: float | None = None
    judge: JudgeScores = Field(default_factory=JudgeScores)
    judge_repeats: list[JudgeScores] = Field(default_factory=list)


class RunConfig(BaseModel):
    name: str
    started_at: str
    git_sha: str
    chat_model: str
    embedding_model: str
    judge_model: str | None
    seed: int
    temperature: float
    chunk_size: int
    chunk_overlap: int
    min_chunk_chars: int
    top_k: int
    similarity_threshold: float
    prompt_hashes: dict[str, str]
    gold_set_hash: str
    gold_set_size: int
    index_chunks: int | None
    packages: dict[str, str]
    repeat: int = 1
    subset: int | None = None


class AbstainMatrix(BaseModel):
    answerable_answered: int = 0
    answerable_abstained: int = 0  # 과잉 거부 (false abstain)
    unanswerable_answered: int = 0  # hallucination 위험 (false answer)
    unanswerable_abstained: int = 0

    @property
    def answer_rate(self) -> float | None:
        n = self.answerable_answered + self.answerable_abstained
        return self.answerable_answered / n if n else None

    @property
    def abstain_rate(self) -> float | None:
        n = self.unanswerable_answered + self.unanswerable_abstained
        return self.unanswerable_abstained / n if n else None


class MetricSummary(BaseModel):
    mean: float | None
    n: int


class Report(BaseModel):
    config: RunConfig
    overall: dict[str, MetricSummary]
    by_type: dict[str, dict[str, MetricSummary]]
    abstain: AbstainMatrix
    abstain_answer_rate: float | None
    abstain_abstain_rate: float | None
    consistency: dict[str, float] | None = None
    human_alignment: dict[str, float | int] | None = None
    items: list[ItemResult]
