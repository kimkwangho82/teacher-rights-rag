# 교권 RAG QA 서비스 — 자주 쓰는 명령 모음
# 사용법: make <target>   (목록: make help)

UV      ?= uv
PORT    ?= 8000
HOST    ?= 0.0.0.0
Q       ?= 교육활동 침해 사안 발생 시 학교장의 조치 절차는?

.DEFAULT_GOAL := help
.PHONY: help install env ingest ingest-reset ingest-dry serve dev test lint format ask health eval eval-smoke eval-nojudge eval-compare eval-exp1 eval-exp2 eval-exp3 eval-rebuild clean clean-index

help: ## 타겟 목록 출력
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## 의존성 설치 (.venv 생성)
	$(UV) sync

env: ## .env 가 없으면 .env.example 로 생성
	@test -f .env && echo ".env 이미 존재" || (cp .env.example .env && echo ".env 생성됨 — OPENAI_API_KEY 를 입력하세요")

ingest: ## data/raw 문서를 인덱싱 (기존 컬렉션에 추가)
	$(UV) run python -m scripts.ingest

ingest-reset: ## 벡터스토어 삭제 후 재인덱싱
	$(UV) run python -m scripts.ingest --reset

ingest-dry: ## 임베딩 없이 로드/청킹 통계만 출력
	$(UV) run python -m scripts.ingest --dry-run

serve: ## API 서버 실행
	$(UV) run uvicorn api.main:app --host $(HOST) --port $(PORT)

dev: ## API 서버 실행 (자동 리로드)
	$(UV) run uvicorn api.main:app --host $(HOST) --port $(PORT) --reload

test: ## 단위 테스트 (외부 API 호출 없음)
	$(UV) run pytest -q

lint: ## ruff 정적 검사
	$(UV) run --with ruff ruff check .

format: ## ruff 포맷 적용
	$(UV) run --with ruff ruff format .

health: ## 서버 헬스체크
	@curl -sf http://localhost:$(PORT)/health && echo || \
		{ echo "서버가 응답하지 않습니다 (localhost:$(PORT)). 먼저 'make dev' 를 실행하세요."; exit 1; }

ask: ## 질의 전송 (make ask Q="질문")
	@out=$$(curl -sS --fail-with-body -X POST http://localhost:$(PORT)/chat \
		-H 'Content-Type: application/json' -d '{"question": "$(Q)"}'); rc=$$?; \
	if [ $$rc -eq 7 ]; then echo "서버에 연결할 수 없습니다 (localhost:$(PORT)). 먼저 'make dev' 를 실행하세요."; exit 1; fi; \
	if [ $$rc -ne 0 ]; then echo "요청 실패 (HTTP 오류):"; printf '%s\n' "$$out"; exit 1; fi; \
	printf '%s\n' "$$out" | $(UV) run python -m json.tool --no-ensure-ascii

eval: ## Gold Set 전체 평가 + 리포트 (make eval NAME=baseline)
	$(UV) run python -m eval.run --name $(or $(NAME),baseline)

eval-smoke: ## 유형별 1개씩 7개만 평가 (빠른 확인)
	$(UV) run python -m eval.run --name smoke --subset 7

eval-nojudge: ## 규칙 기반 지표만 (Judge 호출 없음)
	$(UV) run python -m eval.run --name $(or $(NAME),nojudge) --no-judge

eval-compare: ## 리포트 비교, 여러 개면 평균 (make eval-compare A="eval/reports/baseline-*" B="eval/reports/exp1-*")
	$(UV) run python -m eval.compare --before $(A) --after $(B)

eval-exp1: ## 개선 실험 1: hybrid 검색 (make eval-exp1 NAME=exp1-hybrid-a)
	RETRIEVAL_MODE=hybrid $(UV) run python -m eval.run --name $(or $(NAME),exp1-hybrid)

eval-exp2: ## 개선 실험 2: evidence-first 프롬프트 (make eval-exp2 NAME=exp2-evidence-a)
	PROMPT_MODE=evidence_first $(UV) run python -m eval.run --name $(or $(NAME),exp2-evidence)

eval-exp3: ## 개선 실험 3: hybrid + Kiwi 형태소 BM25 (make eval-exp3 NAME=exp3-kiwi-a)
	RETRIEVAL_MODE=hybrid BM25_TOKENIZER=kiwi $(UV) run python -m eval.run --name $(or $(NAME),exp3-kiwi)

eval-rebuild: ## 기존 리포트를 현재 집계 규칙으로 재집계 (make eval-rebuild DIRS="eval/reports/baseline-*")
	$(UV) run python -m eval.rebuild $(DIRS)

clean: ## 캐시/임시 파일 삭제
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache

clean-index: ## 벡터스토어와 processed 산출물 삭제
	rm -rf data/vectorstore data/processed/chunks.jsonl
