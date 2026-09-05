# Teacher Rights RAG

교육부 **『2025 교육활동 보호 매뉴얼』** 을 근거로 교권(교육활동 보호) 관련 질의에 답하는 Citation 기반 RAG QA 서비스와, 그 품질을 측정하는 Eval Harness, 그리고 Eval 로 검증한 개선 실험 기록.

## 목차
- [실행 방법](#실행-방법)
- [환경 변수](#환경-변수)
- [Corpus](#corpus)
- [아키텍처](#아키텍처)
- [RAG QA 서비스](#rag-qa-서비스)
  - [Ingest 파이프라인](#ingest-파이프라인)
  - [Chunking 전략](#chunking-전략)
  - [Embedding / Index](#embedding--index)
  - [검색·생성 및 Hallucination 방지 (Abstain)](#검색생성-및-hallucination-방지-abstain)
  - [Citation](#citation)
  - [API 스키마](#api-스키마)
- [Design Decision & Trade-off](#design-decision--trade-off)
- [한계 및 알려진 이슈](#한계-및-알려진-이슈)
- [Eval Harness](#eval-harness) (작성 예정)
- [개선 실험](#개선-실험) (작성 예정)

---

## 실행 방법

```bash
# 0. 요구사항: Python 3.12+, uv (https://docs.astral.sh/uv/)
uv sync                          # 1. 의존성 설치 (.venv 자동 생성)
cp .env.example .env             # 2. OPENAI_API_KEY 입력
uv run python -m scripts.ingest --reset   # 3. 문서 Ingest → data/vectorstore/
uv run uvicorn api.main:app --reload      # 4. API 서버 (http://localhost:8000/docs)
uv run pytest                    # 5. 단위 테스트 (외부 API 호출 없음)
# uv run python -m eval.run      # 6. Eval (작성 예정)
```

질의 예시:

```bash
curl -s -X POST localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"question": "교육활동 침해 사안 발생 시 학교장의 조치 절차는?"}' | jq
```

## 환경 변수

| 변수 | 필수 | 기본값 | 설명 |
|---|---|---|---|
| `OPENAI_API_KEY` | ✅ | – | OpenAI(또는 호환 엔드포인트) API 키. 코드에 하드코딩하지 않으며 `.env`는 gitignore. |
| `OPENAI_BASE_URL` | | (OpenAI) | OpenAI 호환 엔드포인트 (프록시/타 제공자) 사용 시 지정 |
| `CHAT_MODEL` | | `gpt-4.1-mini` | 답변 생성 모델 |
| `EMBEDDING_MODEL` | | `text-embedding-3-small` | 임베딩 모델 (인덱싱/검색 동일 모델 강제) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | | `1000` / `200` | 청킹 파라미터 (문자 단위) |
| `MIN_CHUNK_CHARS` | | `50` | 이보다 짧은 청크는 인덱스에서 제외 |
| `TOP_K` | | `4` | 검색 청크 수 |
| `SIMILARITY_THRESHOLD` | | `0.3` | 최고 관련도가 이 값 미만이면 LLM 호출 없이 '정보 불충분' 반환 |

전체 목록과 기본값은 `rag/config.py` 참고.

## Corpus

| 문서 | 발행 | 규모 | 파일 |
|---|---|---|---|
| 2025 교육활동 보호 매뉴얼 | 교육부, 2025 | 219p / 298 chunks | `data/raw/2025+교육활동+보호+매뉴얼(배포용).pdf` |

- **유형**: 공개 문서셋 (정부 기관 정책 매뉴얼), 219페이지.
- **선정 이유**
  - 교권 침해 대응 절차, 교권보호위원회 운영, 법령 근거, 서식 등이 **하나의 체계로 정리된 공식 문서**라 질의의 근거를 페이지 단위로 검증할 수 있다.
  - 절차·기준·수치가 많아 **Hallucination이 관측되기 쉬운** 도메인이다 (예: "며칠 이내에 보고해야 하는가").
  - 일반 상식으로는 답하기 어려운 행정 절차가 많아, 모델의 사전 지식이 아닌 **문서 근거 여부**를 평가하기 좋다.
- **재현성**: PDF 원본을 `data/raw/`에 포함. 메타데이터(제목·발행처·연도)는 `data/raw/manifest.json`에 기록하며 Citation과 이 표에 재사용된다.

## 아키텍처

```
                    ┌────────────────── Ingest (scripts/ingest.py) ──────────────────┐
 data/raw/*.pdf ──▶ rag.loaders ──▶ rag.splitter ──▶ rag.embeddings ──▶ rag.vectorstore
   + manifest.json  (pypdf, 정규화,   (Recursive,      (text-embedding-   (Chroma, cosine,
                     doc_id/page)     chunk_id 부여)    3-small)           data/vectorstore/)
                                          │
                                          └──▶ data/processed/chunks.jsonl (Gold Set 근거 참조용)

                    ┌────────────────── Serve (api/main.py) ─────────────────────────┐
 POST /chat ──▶ api.routers.chat ──▶ rag.chain.answer_question
                                        │ 1) rag.retriever.retrieve(q, k)  → (chunk, score)[]
                                        │ 2) top score < threshold ? → status=insufficient (LLM 미호출)
                                        │ 3) 번호 매긴 문맥 + 시스템 프롬프트 → ChatOpenAI
                                        │ 4) 출력 == INSUFFICIENT_CONTEXT ? → status=insufficient
                                        └ 5) 답변 내 [n] 파싱 → citations
                ◀── ChatResponse {answer, status, citations[], model, latency_ms}
```

## RAG QA 서비스

### Ingest 파이프라인
`uv run python -m scripts.ingest [--reset] [--dry-run]`

1. **로드** (`rag/loaders.py`): `data/raw/` 의 PDF/TXT/MD 를 페이지 단위 `Document` 로 읽는다. pypdf 를 직접 사용해 페이지 번호를 1-based 로 기록한다.
2. **정규화**: 공문 PDF 특유의 연속 공백·개행을 정리하고, 텍스트가 없는 페이지(이미지·빈 페이지)는 제외한다.
3. **메타데이터**: `doc_id`(파일명 slug), `title/publisher/year/url`(manifest), `source`, `page`.
4. **청킹** (`rag/splitter.py`) → 5. **임베딩·저장** (`rag/vectorstore.py`, 100개 배치).
6. 통계(문서/페이지/청크 수, 청크 길이 분포) 출력 및 `data/processed/chunks.jsonl` 덤프.

### Chunking 전략
`RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ". ", " ", ""])`

| 결정 | 근거 |
|---|---|
| 문단(`\n\n`) → 줄 → 문장 순 재귀 분할 | 매뉴얼은 "항목 제목 + 설명 문단 + 번호 목록" 구조. 문단 경계를 최우선으로 존중해야 하나의 절차가 청크 중간에서 끊기지 않는다. |
| `chunk_size=1000` (문자) | 한국어 1000자 ≈ 600~800 토큰. 임베딩 한도(8k) 안에서 절차 하나(보통 5~10문장)가 통째로 들어가는 크기. 실제 분포: 평균 688자, 298 청크. |
| `chunk_overlap=200` (20%) | 항목 경계에서 잘린 문장이 인접 청크에 중복 포함되어 검색 recall 손실을 완화. |
| `min_chunk_chars=50` | 쪽번호·머리글만 남은 조각(9~40자)이 상위 검색 결과를 오염시키는 것을 방지. |
| `chunk_id = {doc_id}:p{page}:c{idx}` | Citation 의 안정적 키. 같은 문서·설정으로 재인덱싱해도 동일 ID → Eval 결과 비교 가능. |

청킹 파라미터는 모두 환경변수로 조정 가능하며 개선 실험의 변수로 사용한다.

### Embedding / Index
- **Embedding**: `text-embedding-3-small`. 한국어 성능이 충분하면서 비용이 large 대비 1/6 수준. 인덱싱·검색 모두 `rag.embeddings.get_embeddings()` 단일 소스를 사용해 모델 불일치를 원천 차단.
- **Vector DB**: Chroma (로컬 영속, `data/vectorstore/`). 외부 인프라 없이 `uv sync` 만으로 재현 가능. `hnsw:space=cosine` 으로 설정해 relevance score 를 0~1 로 해석한다 (abstain 임계값과 Eval 에서 사용).

### 검색·생성 및 Hallucination 방지 (Abstain)
`rag/chain.py` 는 두 단계로 "답변 불가"를 판단한다.

| 단계 | 조건 | 처리 | 잡아내는 케이스 |
|---|---|---|---|
| 1. 점수 기반 | 최고 cosine relevance < `SIMILARITY_THRESHOLD` (0.3) | **LLM 미호출**, `status=insufficient` | 명백히 무관한 질의 (날씨, 잡담). 결정적이고 비용 0. |
| 2. 프롬프트 기반 | 모델 출력이 `INSUFFICIENT_CONTEXT` | `status=insufficient` | 주제는 관련 있으나 문서에 답이 없는 질의. |

시스템 프롬프트는 (1) 참고 자료만 근거로 답할 것, (2) 문장마다 `[n]` 인용, (3) 근거 부족 시 토큰만 출력, (4) 수치·절차는 그대로 인용할 것을 지시한다. `temperature=0`.

임계값 0.3은 초기값이며, Eval Harness 의 Gold Set(답변 불가 질의 포함)으로 abstain 정확도를 측정해 조정한다.

### Citation
- 검색된 청크를 `[1] (제목, p.45)` 형식으로 번호 매겨 프롬프트에 넣고, 답변 본문의 `[n]` 을 정규식으로 파싱해 **실제 인용된 청크만** `citations` 로 반환한다 (등장 순서 유지).
- 모델이 인용 표기를 하나도 하지 않으면 검색 결과 전체를 반환한다 (근거를 숨기지 않기 위함).
- 각 citation 은 `chunk_id / doc_id / title / page / snippet(200자) / score` 를 포함하므로 사용자는 원문 페이지로 바로 검증할 수 있다.

### API 스키마
`POST /chat` — 전체 스키마와 예시는 `http://localhost:8000/docs` 참고.

```jsonc
// Request
{ "question": "교육활동 침해 사안 발생 시 학교장의 조치 절차는?", "top_k": 4 }   // top_k 선택 (1~20)

// Response
{
  "answer": "학교장은 ... 즉시 피해 교원을 보호하고 [1] ... 교권보호위원회에 ... [2]",
  "status": "answered",                // "answered" | "insufficient"
  "citations": [
    { "index": 1, "chunk_id": "2025-교육활동-보호-매뉴얼-배포용:p45:c0",
      "doc_id": "2025-교육활동-보호-매뉴얼-배포용", "title": "2025 교육활동 보호 매뉴얼",
      "page": 45, "snippet": "학교장은 교육활동 침해행위를 인지한 경우 ...", "score": 0.62 }
  ],
  "model": "gpt-4.1-mini",
  "latency_ms": 1830
}
```

오류: 빈 질의 → `422`, LLM/벡터스토어 오류 → `502 {"detail": "upstream error: ..."}`.

## Design Decision & Trade-off

| 결정 | 대안 | 선택 이유 / 트레이드오프 |
|---|---|---|
| LangChain 을 얇게 사용 (splitter, embeddings, Chroma 래퍼, ChatOpenAI 만) | LangChain 체인/Agent 전면 사용 · 프레임워크 없이 직접 구현 | 파이프라인 각 단계를 함수로 노출해 내부 동작을 모두 설명·테스트할 수 있게 했다. 로더는 pypdf 직접 호출 (`langchain-community` sunset 회피). 대신 LangChain 의 편의 체인은 쓰지 않아 코드가 조금 더 길다. |
| OpenAI SDK + `OPENAI_BASE_URL` | 제공자별 전용 클라이언트 | 환경변수 하나로 OpenAI ↔ OpenAI 호환 엔드포인트를 전환. 비용에 따라 저렴한 모델/제공자로 바꾸기 쉽다. |
| `gpt-4.1-mini` 기본 | `gpt-4.1` | 제한된 예산 안에서 Eval 을 여러 번 돌려야 하므로 저비용 모델을 기본으로 두고, 모델 교체는 개선 실험의 변수로 남긴다. |
| Chroma 로컬 | Pinecone 등 관리형 · FAISS | 외부 계정 없이 재현 가능 + 메타데이터 필터/점수 조회 지원. 대규모 트래픽에는 부적합하나 현재 규모(수백 청크)에는 충분. |
| 문자 기반 청킹 | 토큰 기반 · 구조(제목) 기반 청킹 | 구현이 단순하고 파라미터를 실험 변수로 삼기 쉽다. 매뉴얼의 목차 구조를 활용한 구조 기반 청킹은 개선 실험 후보. |
| 2단계 abstain (점수 + 프롬프트) | 프롬프트만 | 점수 단계는 무관한 질의에 LLM 비용을 쓰지 않고 결정적으로 동작한다. 임계값이 너무 높으면 정답 가능한 질의를 거절하는 위험이 있어 Eval 로 조정한다. |
| 동기 응답 (Streaming 미구현) | SSE 스트리밍 | 서비스는 MVP 로 두고 Eval Harness 와 개선 실험에 시간을 배분. |

## 한계 및 알려진 이슈
- **단일 문서 Corpus**: 문서 수가 1개라 문서 간 충돌·출처 선택 문제는 다루지 못한다. 교육청 지침·법령 원문 추가가 후속 작업.
- **PDF 텍스트 추출 품질**: 표·서식 페이지는 pypdf 추출 시 열 순서가 섞일 수 있다. 서식 관련 질의 정확도가 낮을 수 있다.
- **고정 임계값**: `SIMILARITY_THRESHOLD` 는 임베딩 모델·도메인에 따라 달라지므로 Eval 없이는 보정이 어렵다.
- **Citation 검증 없음**: 모델이 `[n]` 을 붙였다고 해당 청크가 실제 근거임을 보장하지 않는다 (Eval Harness 의 faithfulness 지표로 측정).
- **단일 프로세스 Chroma**: 동시 요청·다중 워커 환경에서는 별도 벡터 DB 서버가 필요하다.

## Eval Harness
_작성 예정_

## 개선 실험
_작성 예정_
