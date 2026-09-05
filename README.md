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
- [Eval Harness](#eval-harness)
  - [Gold Set](#gold-set)
  - [지표 정의](#지표-정의)
  - [LLM-as-a-Judge 신뢰성](#llm-as-a-judge-신뢰성)
  - [실행과 재현성](#실행과-재현성)
  - [Baseline 결과](#baseline-결과)
  - [지표의 한계와 맹점](#지표의-한계와-맹점)
  - [CI 연동 설계](#ci-연동-설계)
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
uv run python -m eval.run --name baseline   # 6. Eval Harness 실행 → eval/reports/baseline_<시각>/report.md
```

같은 작업을 `make` 로도 실행할 수 있다 (`make help` 로 목록 확인):

```bash
make install && make env      # 설치 + .env 생성
make ingest-reset             # 인덱싱
make dev                      # 서버 (자동 리로드)
make test                     # 테스트
make ask Q="교권보호위원회 구성은?"   # 질의 전송
make eval                     # Eval Harness (make eval NAME=exp1)
make eval-smoke               # 유형별 1개씩 7개만
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

RAG 응답 품질을 **검색 단계**와 **생성 단계**로 나누어 측정한다. 점수가 낮을 때 "근거를 못 찾은 것"인지 "찾았는데 잘못 답한 것"인지 분리하기 위한 구조다. 지표 정의는 RAGAS 논문(Es et al., 2023)을 따르되 라이브러리를 쓰지 않고 `eval/metrics/` 에 직접 구현했다. 프레임워크를 블랙박스로 쓰지 않기 위해서이고, 프롬프트를 파일로 고정해 버전을 추적하기 위해서다.

```
eval/
├── gold_set.jsonl        # Gold Set 32개
├── human_labels.jsonl    # Judge 검증용 수동 라벨 32개
├── prompts/*.txt         # Judge 프롬프트 5개 (해시로 버전 기록)
├── metrics/              # retrieval(규칙) · abstain(규칙) · ragas(Judge) · correctness(Judge)
├── judge.py              # Judge 클라이언트: temperature 0, seed, JSON 파싱, 디스크 캐시
├── runner.py / report.py # 실행 · 집계 · report.json/md
├── run.py                # 단일 명령 진입점
├── compare.py            # Before/After 비교
└── reports/              # 실행 결과 (baseline 포함)
```

### Gold Set

`eval/gold_set.jsonl`, 32문항. 각 문항은 질의, 답 가능 여부, **근거 페이지**, **수용 기준 체크리스트**(필수/선택)를 가진다.

| 유형 | n | 예시 |
|---|---|---|
| factual | 8 | 지역교권보호위원회는 몇 명으로 구성되고 임기는? |
| procedural | 6 | 분리조치는 어떤 절차로 진행되고 기간은 어느 정도가 권장되나? |
| multi_evidence | 4 | 행정심판과 행정소송의 청구 기간은 각각 어떻게 다른가? (2개 절 교차) |
| summary | 4 | 교원보호공제사업 표준약관의 보장 항목과 한도를 정리해 주세요 |
| reasoning | 4 | 과자를 먹다 지도받고 멈췼지만 다시 먹은 학생, 침해행위인가? (사례 → 유형 판단) |
| unanswerable_out | 3 | 형사 고소 시 승소 확률은? (문서 범위 밖) |
| unanswerable_in | 3 | 회의록은 몇 년간 보존해야 하나? (주제는 범위 안, 세부 정보는 문서에 없음) |

- **구축 방법**: `data/processed/chunks.jsonl` 로 매뉴얼 원문을 장별로 읽으며 AI 보조로 초안을 작성하고, 근거 페이지를 원문과 대조해 확정했다. 검색 결과 검토 과정에서 같은 근거가 부록 법령·고시에도 실려 있는 경우(예: 침해학생 조치 7종은 p25·36·53·180)를 발견해 근거 페이지에 추가했다. 답 없는 질의 6개는 `chunks.jsonl` 전문 검색으로 부재를 확인했다.
- **근거 키를 chunk_id 가 아닌 page 로 둔 이유**: 청킹 파라미터를 바꾸는 실험에서도 gold 를 다시 매핑하지 않고 비교하기 위해서다. 대신 같은 페이지의 무관한 청크도 정답으로 처리되는 관대함이 있다.
- **편향·한계**: 단일 문서(교육부 매뉴얼 1종) 기반이며 작성자 1인이 검수했다. 질의 표현이 문서 용어와 가까워 실제 교사의 구어체 질의보다 검색이 쉬울 수 있다. 절차·위원회 장(Ⅱ·Ⅲ)에 문항이 집중되고 서식·부록은 거의 다루지 않는다. `unanswerable_in` 은 "문서에 없다" 는 판단을 문서 전문 검색에 의존하므로, 표현이 다른 형태로 존재할 가능성을 완전히 배제하지 못한다.

### 지표 정의

| 단계 | 지표 | 정의 | 판정 | 선정 이유 |
|---|---|---|---|---|
| 검색 | **Recall@K** | gold 근거 페이지 중 상위 K 청크의 페이지에 포함된 비율 | 규칙 | 결정적·재현 가능. 다중 근거 질의에서 Hit Rate 와 갈라진다. |
| 검색 | **MRR** | gold 페이지에 속하는 첫 청크의 1/rank (없으면 0) | 규칙 | 순위에 민감해 리랭킹·하이브리드 실험 효과를 Recall 보다 예민하게 잡는다. |
| 검색 | Context Relevance | 문맥에서 질문에 필요한 문장만 추출 → 추출 문장 수 / 전체 문장 수 (RAGAS 식 2) | Judge | gold 없이도 "불필요한 문맥이 얼마나 섞였나" 를 본다. K·청크 크기 실험의 부작용 감시용. |
| 생성 | **Faithfulness** | 답변을 원자적 주장으로 분해 → 각 주장이 *실제 프롬프트에 들어간 문맥*에서 추론 가능한지 → 지지 주장 수 / 전체 (RAGAS F = \|V\|/\|S\|) | Judge ×2 | Hallucination 직접 측정. 주장 단위라 전체 점수 방식보다 안정적이고 어떤 주장이 지어낸 것인지 로그가 남는다. |
| 생성 | **Correctness** | 수용 기준 체크리스트 항목별 pass/fail → 통과/전체. 필수 항목 실패 시 0 | Judge | 요약·추론형 질의에 참조 답변 유사도보다 적합. Faithfulness 와 교차하면 "충실하지만 틀림 = 검색 실패", "맞지만 불충실 = 사전지식 답변" 을 구분할 수 있다. |
| 생성 | **Abstain Accuracy** | 답 없는 질의의 거부율과 답 있는 질의의 응답률을 **따로** 보고 + 혼동행렬 | 규칙 | "전부 거부" 시스템이 Faithfulness 만점을 받는 것을 견제한다. 두 오류를 분리해야 임계값 실험의 트레이드오프가 보인다. |
| 생성 | Answer Relevance (보조) | 답변에서 질문 3개를 역생성 → 원 질문과 임베딩 cosine 평균 (RAGAS 식 1) | Judge + 임베딩 | 불완전하거나 장황한 답변 감지. 보조 지표. |

Abstain 응답은 주장이 0개이므로 Faithfulness·Answer Relevance 계산에서 제외하고 Abstain 지표로만 집계한다. 과잉 거부 케이스는 검색 문맥의 Context Relevance 만 측정해 검색 문제인지 생성 문제인지 분리한다.

### LLM-as-a-Judge 신뢰성

| 확보 방법 | 구현 |
|---|---|
| 결정성 | `temperature=0`, `seed` 고정(기본 42), JSON 구조화 출력. 파싱 실패 시 1회 재시도 후 `judge_error` 로 기록 |
| 프롬프트 고정 | `eval/prompts/*.txt` 파일로 분리하고 논문처럼 demonstration 1개 포함. 파일 해시를 run_config 에 기록해 프롬프트 변경을 추적 |
| 생성 모델과 분리 | `JUDGE_MODEL`(기본 gpt-4.1)을 생성 모델과 별도로 두어 self-preference 회피 |
| 일관성 측정 | `--repeat 3`: 같은 입력을 3회 채점해 항목별 표준편차와 이진 판정 일치율 보고 |
| Human alignment | `eval/human_labels.jsonl` 에 32문항 전부 수동 라벨(faithful / correct). Judge 이진 판정과의 일치율과 Cohen's κ 를 리포트에 표시 |

Baseline 측정값 (`eval/reports/baseline_20260905-150624`):

| 지표 | 3회 반복 표준편차 평균 | 이진 판정 일치율 | 수동 라벨 일치율 (n) | Cohen's κ |
|---|---|---|---|---|
| faithfulness (임계값 0.8) | 0.011 | 95.5% | 90.9% (22) | 0.62 |
| correctness (임계값 0.5) | 0.029 | 93.8% | 96.9% (32) | 0.92 |
| context_relevance | 0.008 | 100% | – | – |

Faithfulness 불일치 2건은 모두 **Judge 가 사람보다 엄격**한 경우였다. 문맥의 "즉시 분리 의사 확인서(서식2)" 를 답변이 "확인서를 즉시 작성" 으로 재구성한 것(q010), "1년을 경과하면 제기할 수 없다" 에서 "두 기간 중 하나라도 경과하면 불가" 를 추론한 것(q017)을 미지지로 판정했다. 요약·추론형 답변에서 재구성 표현을 보수적으로 보는 RAGAS 방식의 알려진 경향과 일치한다.

### 실행과 재현성

```bash
make eval                                     # = uv run python -m eval.run --name baseline
uv run python -m eval.run --name x --subset 7 # 유형별 1개 (스모크)
uv run python -m eval.run --name x --repeat 3 # Judge 일관성
uv run python -m eval.run --name x --no-judge # 규칙 지표만 (Judge 비용 0, 생성 호출은 발생)
make eval-compare A=eval/reports/baseline_x B=eval/reports/exp_y   # Before/After 표
```

- 단일 명령으로 Gold Set 전체를 평가하고 `report.md`(요약) · `report.json`(집계) · `items.jsonl`(문항별 상세: 검색 청크·점수·주장별 판정·기준별 판정)을 생성한다.
- **run_config 기록**: git SHA, 실행 시각, 생성/임베딩/Judge 모델, seed, temperature, 청킹·검색 설정, 프롬프트 해시, Gold Set 해시, 인덱스 청크 수, 주요 패키지 버전.
- **Judge 캐시**: `sha256(모델, seed, 프롬프트 해시, 입력, 반복 인덱스)` 키로 `eval/cache/` 에 저장. 같은 답변을 다시 채점하지 않으므로 재실행 비용이 크게 줄고(위 baseline 재실행 시 125회 중 97회 캐시 적중), 캐시를 지우고 재실행해도 seed 로 동일 결과를 기대한다.
- 생성 단계에도 seed 를 전달하지만 OpenAI 의 seed 는 best-effort 라 재실행 시 답변이 소폭 달라질 수 있다. 실제로 재실행 간 faithfulness 가 0.93 → 0.90 으로 변동했다. 그래서 실험 비교는 같은 Gold Set·같은 설정에서 **여러 번 실행한 평균**을 권장하며, 리포트에 항목별 점수를 모두 남겨 어느 문항이 흔들렸는지 추적할 수 있게 했다.

### Baseline 결과

설정: gpt-4.1 / text-embedding-3-small / chunk 1000·200 / top_k 4 / threshold 0.3 / Judge gpt-4.1 (`eval/reports/baseline_20260905-150624`)

| 단계 | 지표 | 값 | n |
|---|---|---|---|
| 검색 | Recall@4 | 0.513 | 26 |
| 검색 | MRR | 0.673 | 26 |
| 검색 | Context Relevance | 0.110 | 27 |
| 생성 | Faithfulness | 0.898 | 22 |
| 생성 | Correctness | 0.641 | 32 |
| 생성 | Answer Relevance | 0.742 | 22 |
| Abstain | 답 있는 질의 응답률 | 80.8% (21/26) | |
| Abstain | 답 없는 질의 거부율 | 83.3% (5/6) | |

| 유형 | n | Recall@4 | MRR | Faith | Correct | 거부율 |
|---|---|---|---|---|---|---|
| factual | 8 | 0.56 | 0.69 | 1.00 | 0.52 | 12.5% |
| procedural | 6 | 0.35 | 0.58 | 0.88 | 0.78 | 33.3% |
| multi_evidence | 4 | 0.50 | 0.75 | 0.83 | 0.17 | 25.0% |
| summary | 4 | 0.81 | 1.00 | 1.00 | 1.00 | 0.0% |
| reasoning | 4 | 0.38 | 0.38 | 0.92 | 0.50 | 25.0% |
| unanswerable_in | 3 | – | – | 0.00 | 0.67 | 66.7% |
| unanswerable_out | 3 | – | – | – | 1.00 | 100% |

**관찰**
- **과잉 거부 5건(q004, q009, q013, q016, q024)은 모두 Recall@4 = 0** 이다. 즉 생성 모델이 소극적인 것이 아니라 근거 페이지가 상위 4개에 들지 않아 프롬프트 규칙대로 거부한 것이다. 병목은 검색 단계다. 예: "시·도교권보호위원회 위원 정수" 질의에 p24·26·28(지역교권보호위원회)은 검색되고 p23(시·도)은 빠졌다. 용어가 비슷한 두 위원회를 dense 임베딩이 구분하지 못했다.
- **Hallucination 1건(q032)**: "특별휴가 기간 급여" 는 문서에 없는데, 특별휴가 관련 청크가 높은 점수(0.72)로 검색돼 점수 기반 abstain 을 통과했고 모델도 "정상 지급된다" 고 답했다. 프롬프트 기반 abstain 이 실패한 유일한 사례로, `unanswerable_in` 유형이 왜 필요한지 보여준다.
- **multi_evidence 의 Correctness 0.17**: 두 절을 교차해야 하는 질의에서 한쪽 근거만 검색되면 모델이 나머지를 "자료에 없다" 고 답한다(q015). 요약형(summary)은 근거가 한 페이지에 모여 있어 1.00 을 기록했다.
- **Faithfulness 는 높고(0.90) Correctness 는 낮다(0.64)**: 지어내지는 않지만 근거를 덜 가져오거나 일부만 답하는 패턴. 개선 실험은 생성보다 검색(Recall) 쪽을 겨냥해야 한다는 근거가 된다.
- Context Relevance 0.11 은 4개 청크(약 2,800자) 중 실제 필요한 문장이 10% 안팎이라는 뜻이다. 절대값보다 K·청크 크기 실험에서의 변화량으로 해석한다.

### 지표의 한계와 맹점

| 지표 | 맹점 |
|---|---|
| Recall@K / MRR | 페이지 단위라 같은 페이지의 무관 청크도 정답 처리. gold 근거가 여러 곳에 중복될 때 어느 하나만 찾아도 만점. MRR 은 첫 정답 순위만 반영해 다중 근거 질의에 불리. |
| Context Relevance | 문장 분할 규칙(종결부호·글머리표 기준)에 의존. 판정자가 추출한 문장 수와 규칙 분할 수의 단위가 완전히 일치하지 않아 근사값이다. 표·서식 페이지에서 낮게 나온다. |
| Faithfulness | 문맥을 그대로 복사하면 만점. 재구성 표현을 미지지로 판정하는 보수성(위 alignment 불일치 2건). 주장 분해 단계가 문장을 얼마나 잘게 나누느냐에 따라 분모가 달라진다. 숫자·주체가 미세하게 다른 오류를 supported 로 넘길 수 있다. |
| Correctness | 수용 기준이 작성자 1인의 판단. 기준에 없는 올바른 정보를 더해도 점수가 오르지 않고, 기준의 표현과 다르게 말하면 fail 될 수 있다. 필수 항목 실패 시 0 으로 만드는 규칙이 부분 정답을 과소평가한다. |
| Abstain | `status` 필드만 보므로 "답변은 했지만 본문에서 '자료에 없다' 고 얼버무린" 경우(q015)를 거부로 잡지 못한다. 거부 자체가 정답인 6문항은 Faithfulness 계산에서 빠져 생성 지표의 n 이 줄어든다. |
| Answer Relevance | 역생성 질문과 임베딩 모델에 의존. 답변이 틀려도 질문과 주제가 같으면 높게 나온다(사실성 미반영). |
| Judge 전반 | 사람 라벨도 작성자 1인이라 alignment 수치 자체에 편향이 있다. 같은 모델(gpt-4.1)이 생성과 판정을 모두 맡으므로 self-preference 가 완전히 제거되지 않는다(JUDGE_MODEL 로 교체 가능). |

### CI 연동 설계

Eval 파이프라인을 회귀 방지 장치로 쓰려면 비용과 비결정성을 통제해야 한다. 제안하는 구성:

| 단계 | 트리거 | 내용 | 비용 통제 |
|---|---|---|---|
| Unit | 모든 PR | `pytest` (지표 함수, 캐시 키, 리포트 생성 — LLM 호출 없음) + `ruff` | 0 |
| Smoke | 모든 PR | `eval.run --subset 7 --no-judge` → Recall@K·Abstain 만 검사. 임계값(예: Recall@4 ≥ baseline − 0.1, 답 없는 질의 거부율 = 100%) 미달 시 실패 | 생성 7회 + 임베딩 |
| Full | `main` 머지 후 야간 1회, 또는 `eval/`·`rag/` 변경 PR 에 수동 라벨 | `eval.run --repeat 2` 전체 → `compare.py` 로 직전 baseline 과 diff 를 PR 코멘트에 게시 | Judge 캐시를 GitHub Actions cache 에 보존해 변경 없는 문항은 재채점하지 않음 |
| Gate | Full 결과 | Correctness·Faithfulness 가 baseline 대비 −0.05 이상 하락하거나 hallucination(답 없는 질의 답변) 건수가 증가하면 실패. 단일 실행 변동(±0.03)보다 큰 폭만 게이트로 사용 | |

- 프롬프트·Gold Set 해시가 바뀐 PR 은 baseline 갱신 PR 로 취급해 게이트를 건너뛰고 새 baseline 을 커밋한다.
- API 키는 저장소 secret 으로만 주입하고 fork PR 에서는 Smoke/Full 을 실행하지 않는다.
- `.github/workflows/eval.yml` 에 Unit 단계와 수동 트리거 Smoke 를 구현해 두었다. Full 단계는 크레딧 사정에 따라 스케줄을 켜는 것을 전제로 설계만 기록한다.

## 개선 실험
_작성 예정_
