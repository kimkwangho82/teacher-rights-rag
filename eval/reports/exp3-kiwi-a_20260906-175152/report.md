# Eval Report: exp3-kiwi-a

## Run config

| 항목 | 값 |
|---|---|
| started_at | 2026-09-06T17:47:40+09:00 |
| git_sha | 704fc98 |
| chat_model | gpt-4.1 |
| embedding_model | text-embedding-3-small |
| judge_model | gpt-4.1 |
| seed | 42 |
| temperature | 0.0 |
| chunk_size | 1000 |
| chunk_overlap | 200 |
| min_chunk_chars | 50 |
| top_k | 5 |
| similarity_threshold | 0.3 |
| retrieval_mode | hybrid |
| bm25_tokenizer | kiwi |
| prompt_mode | baseline |
| gold_set_hash | 8676a6a7aede |
| gold_set_size | 32 |
| index_chunks | 298 |
| repeat | 1 |
| subset | None |
| prompt_hashes | answer_relevance=a1a0b1fa3390, context_relevance=587985e98b7b, correctness=413e68c22ae7, faithfulness_statements=e6803e91d998, faithfulness_verify=92ddaa87ef69 |
| packages | langchain=1.4.0, langchain-openai=1.6.0, langchain-chroma=1.1.0, chromadb=1.5.9, openai=3.8.0, pypdf=6.17.0 |

## 검색 단계 (Retrieval)

| 지표 | 평균 | n |
|---|---|---|
| recall_at_k | 0.734 | 26 |
| recall_at_3 | 0.638 | 26 |
| mrr | 0.767 | 26 |
| context_relevance | 0.098 | 27 |

## 생성 단계 (Generation)

| 지표 | 평균 | n |
|---|---|---|
| faithfulness | 0.940 | 27 |
| correctness | 0.771 | 32 |
| answer_relevance | 0.752 | 27 |

## Abstain

- 답 있는 질의 응답률: **100.0%** (26/26)
- 답 없는 질의 거부율: **83.3%** (5/6)

| | 시스템: 답변 | 시스템: 거부 |
|---|---|---|
| Gold: 답 있음 | 26 | 0 (과잉 거부) |
| Gold: 답 없음 | 1 (hallucination 위험) | 5 |

## 유형별

| 유형 | n | recall@k | mrr | ctx_rel | faith | correct | ans_rel | 거부율 |
|---|---|---|---|---|---|---|---|---|
| factual | 8 | 0.833 | 0.688 | 0.078 | 1.000 | 0.833 | 0.802 | 0.0% |
| multi_evidence | 4 | 0.646 | 0.800 | 0.166 | 0.908 | 0.667 | 0.777 | 0.0% |
| procedural | 6 | 0.514 | 0.667 | 0.069 | 0.844 | 0.611 | 0.732 | 0.0% |
| reasoning | 4 | 0.875 | 1.000 | 0.047 | 1.000 | 0.792 | 0.668 | 0.0% |
| summary | 4 | 0.812 | 0.812 | 0.171 | 1.000 | 0.875 | 0.717 | 0.0% |
| unanswerable_in | 3 | - | - | 0.073 | 0.667 | 0.667 | 0.847 | 66.7% |
| unanswerable_out | 3 | - | - | - | - | 1.000 | - | 100.0% |

## Human alignment (수동 라벨 vs Judge)

| 지표 | n | 일치율 | Cohen's κ |
|---|---|---|---|
| faithfulness (임계값 0.8) | 23 | 91.3% | -0.045 |
| correctness (임계값 0.5) | 32 | 84.4% | 0.459 |

## 주의가 필요한 항목

- **q009** (procedural) 교육활동 침해 사안이 접수되면 학교장은 언제까지 어떤 방법으로 교육지원청에 보고해야 하나요?  
  → correct 0.00 (실패: 사안 발생보고서(서식6)는 사안 보고 후 5일 이내(주말·공휴일 제외) 공문 제출)
- **q011** (procedural) 지역교권보호위원회는 사안 발생보고서 접수 후 며칠 이내에 열어야 하며, 연장할 수 있나요?  
  → faith 0.50 (미지지: 지역교권보호위원회는 사안 발생보고서 접수 후 21일 이내에 개최해야 한다.)
- **q012** (procedural) 지역교권보호위원회가 조치를 의결한 뒤 교육장은 언제까지 조치하고 결과를 어떻게 통지하나요?  
  → recall 0.25
- **q013** (procedural) 침해학생이 조치를 이행하지 않을 경우 학교와 교육장은 어떻게 대응하나요?  
  → recall 0.00; faith 0.67 (미지지: 교육장은 해당 학생이 전학할 학교의 배정을 지체 없이 요청해야 한다.); correct 0.00 (실패: 통보받은 날부터 90일 내 미이행 시 학교장이 미이행 학생 명단을 교육장(교권보호위원회)에 보고 / 교육장은 보고 후 21일 이내에 30일 이내 이행할 것을 서면 안내)
- **q016** (multi_evidence) 교육활동 침해학생에 대한 조치와 침해 보호자 등에 대한 조치는 종류가 어떻게 다른가요?  
  → recall 0.25
- **q018** (multi_evidence) 특별휴가 5일을 다 쓴 뒤에도 요양이 더 필요하면 어떤 제도를 이용할 수 있고 기간은 얼마인가요?  
  → correct 0.00 (실패: 공무원연금공단 공무상 요양 승인 시 연 180일 범위에서 공무상 병가 가능)
- **q020** (summary) 피해교원에 대한 보호조치에는 어떤 유형이 있는지 정리해 주세요.  
  → recall 0.25
- **q032** (unanswerable_in) 교육활동 침해 피해교원이 특별휴가를 사용하는 기간의 급여는 어떻게 처리되나요?  
  → 답 없는 질의에 답변; faith 0.67 (미지지: 특별휴가 사용 시 급여는 정상적으로 지급된다.); correct 0.00 (실패: 답변 불가 또는 정보 불충분을 표시)
