# Eval Report: exp1-hybrid-b

## Run config

| 항목 | 값 |
|---|---|
| started_at | 2026-09-06T17:21:43+09:00 |
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
| bm25_tokenizer | bigram |
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
| recall_at_k | 0.692 | 26 |
| recall_at_3 | 0.606 | 26 |
| mrr | 0.735 | 26 |
| context_relevance | 0.088 | 27 |

## 생성 단계 (Generation)

| 지표 | 평균 | n |
|---|---|---|
| faithfulness | 0.957 | 25 |
| correctness | 0.786 | 32 |
| answer_relevance | 0.741 | 26 |

## Abstain

- 답 있는 질의 응답률: **96.2%** (25/26)
- 답 없는 질의 거부율: **83.3%** (5/6)

| | 시스템: 답변 | 시스템: 거부 |
|---|---|---|
| Gold: 답 있음 | 25 | 1 (과잉 거부) |
| Gold: 답 없음 | 1 (hallucination 위험) | 5 |

## 유형별

| 유형 | n | recall@k | mrr | ctx_rel | faith | correct | ans_rel | 거부율 |
|---|---|---|---|---|---|---|---|---|
| factual | 8 | 0.729 | 0.573 | 0.105 | 0.865 | 0.833 | 0.793 | 0.0% |
| multi_evidence | 4 | 0.646 | 1.000 | 0.082 | 1.000 | 0.417 | 0.768 | 0.0% |
| procedural | 6 | 0.472 | 0.700 | 0.066 | 1.000 | 0.611 | 0.756 | 16.7% |
| reasoning | 4 | 0.875 | 0.875 | 0.064 | 1.000 | 0.792 | 0.688 | 0.0% |
| summary | 4 | 0.812 | 0.708 | 0.126 | 1.000 | 1.000 | 0.698 | 0.0% |
| unanswerable_in | 3 | - | - | 0.040 | - | 1.000 | 0.528 | 66.7% |
| unanswerable_out | 3 | - | - | - | - | 1.000 | - | 100.0% |

## Human alignment (수동 라벨 vs Judge)

| 지표 | n | 일치율 | Cohen's κ |
|---|---|---|---|
| faithfulness (임계값 0.8) | 23 | 87.0% | -0.062 |
| correctness (임계값 0.5) | 32 | 90.6% | 0.676 |

## 주의가 필요한 항목

- **q002** (factual) 교원지위법 제19조가 정한 교육활동 침해행위 유형에는 어떤 것들이 있나요?  
  → recall 0.33
- **q006** (factual) 지역교권보호위원회가 교육활동 침해학생에 대해 교육장에게 요청할 수 있는 조치는 무엇인가요?  
  → faith 0.67 (미지지: 지역교권보호위원회가 교육장에게 요청할 수 있는 조치에는 학급교체가 있다. / 지역교권보호위원회가 교육장에게 요청할 수 있는 조치에는 전학이 있다.)
- **q008** (factual) 보호자가 정당한 사유 없이 특별교육이나 심리치료에 참여하지 않으면 과태료는 얼마인가요?  
  → faith 0.25 (미지지: 보호자가 1년간 1회 위반 시 100만원의 과태료가 부과된다. / 보호자가 2회 위반 시 150만원의 과태료가 부과된다.)
- **q009** (procedural) 교육활동 침해 사안이 접수되면 학교장은 언제까지 어떤 방법으로 교육지원청에 보고해야 하나요?  
  → recall 0.25; correct 0.00 (실패: 사안 발생보고서(서식6)는 사안 보고 후 5일 이내(주말·공휴일 제외) 공문 제출)
- **q012** (procedural) 지역교권보호위원회가 조치를 의결한 뒤 교육장은 언제까지 조치하고 결과를 어떻게 통지하나요?  
  → recall 0.25
- **q013** (procedural) 침해학생이 조치를 이행하지 않을 경우 학교와 교육장은 어떻게 대응하나요?  
  → 과잉 거부; recall 0.00; correct 0.00 (실패: 통보받은 날부터 90일 내 미이행 시 학교장이 미이행 학생 명단을 교육장(교권보호위원회)에 보고 / 교육장은 보고 후 21일 이내에 30일 이내 이행할 것을 서면 안내)
- **q016** (multi_evidence) 교육활동 침해학생에 대한 조치와 침해 보호자 등에 대한 조치는 종류가 어떻게 다른가요?  
  → recall 0.25; correct 0.00 (실패: 학생: 학교봉사~퇴학 등 7가지 조치(4개 이상 열거) / 보호자 등: 서면사과 및 재발방지 서약, 특별교육 이수 또는 심리치료 2가지)
- **q018** (multi_evidence) 특별휴가 5일을 다 쓴 뒤에도 요양이 더 필요하면 어떤 제도를 이용할 수 있고 기간은 얼마인가요?  
  → correct 0.00 (실패: 공무원연금공단 공무상 요양 승인 시 연 180일 범위에서 공무상 병가 가능)
- **q020** (summary) 피해교원에 대한 보호조치에는 어떤 유형이 있는지 정리해 주세요.  
  → recall 0.25
- **q032** (unanswerable_in) 교육활동 침해 피해교원이 특별휴가를 사용하는 기간의 급여는 어떻게 처리되나요?  
  → 답 없는 질의에 답변
