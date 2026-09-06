# Eval Report: baseline-b

## Run config

| 항목 | 값 |
|---|---|
| started_at | 2026-09-06T17:14:01+09:00 |
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
| retrieval_mode | dense |
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
| recall_at_k | 0.545 | 26 |
| recall_at_3 | 0.449 | 26 |
| mrr | 0.681 | 26 |
| context_relevance | 0.115 | 27 |

## 생성 단계 (Generation)

| 지표 | 평균 | n |
|---|---|---|
| faithfulness | 0.911 | 23 |
| correctness | 0.651 | 32 |
| answer_relevance | 0.749 | 24 |

## Abstain

- 답 있는 질의 응답률: **88.5%** (23/26)
- 답 없는 질의 거부율: **83.3%** (5/6)

| | 시스템: 답변 | 시스템: 거부 |
|---|---|---|
| Gold: 답 있음 | 23 | 3 (과잉 거부) |
| Gold: 답 없음 | 1 (hallucination 위험) | 5 |

## 유형별

| 유형 | n | recall@k | mrr | ctx_rel | faith | correct | ans_rel | 거부율 |
|---|---|---|---|---|---|---|---|---|
| factual | 8 | 0.604 | 0.688 | 0.083 | 1.000 | 0.521 | 0.825 | 12.5% |
| multi_evidence | 4 | 0.500 | 0.750 | 0.097 | 0.864 | 0.167 | 0.784 | 0.0% |
| procedural | 6 | 0.347 | 0.583 | 0.093 | 0.875 | 0.556 | 0.677 | 33.3% |
| reasoning | 4 | 0.500 | 0.425 | 0.057 | 0.750 | 0.667 | 0.685 | 0.0% |
| summary | 4 | 0.812 | 1.000 | 0.321 | 1.000 | 1.000 | 0.714 | 0.0% |
| unanswerable_in | 3 | - | - | 0.000 | - | 1.000 | 0.770 | 66.7% |
| unanswerable_out | 3 | - | - | - | - | 1.000 | - | 100.0% |

## Human alignment (수동 라벨 vs Judge)

| 지표 | n | 일치율 | Cohen's κ |
|---|---|---|---|
| faithfulness (임계값 0.8) | 23 | 82.6% | -0.070 |
| correctness (임계값 0.5) | 32 | 100.0% | 1.000 |

## 주의가 필요한 항목

- **q002** (factual) 교원지위법 제19조가 정한 교육활동 침해행위 유형에는 어떤 것들이 있나요?  
  → recall 0.33
- **q004** (factual) 시·도교권보호위원회의 위원 정수와 임기는 어떻게 정해져 있나요?  
  → 과잉 거부; recall 0.00; correct 0.00 (실패: 10명 이상 20명 이하(위원장 포함) / 3년의 범위에서 교육감이 정하며 1회 연임 가능)
- **q006** (factual) 지역교권보호위원회가 교육활동 침해학생에 대해 교육장에게 요청할 수 있는 조치는 무엇인가요?  
  → correct 0.00 (실패: 학교에서의 봉사, 사회봉사, 특별교육 또는 심리치료, 출석정지, 학급교체, 전학, 퇴학처분 중 6개 이상 열거)
- **q009** (procedural) 교육활동 침해 사안이 접수되면 학교장은 언제까지 어떤 방법으로 교육지원청에 보고해야 하나요?  
  → 과잉 거부; recall 0.00; correct 0.00 (실패: 사안 접수 후 24시간 이내 보고(유선·구두 또는 사안 신고서) / 사안 발생보고서(서식6)는 사안 보고 후 5일 이내(주말·공휴일 제외) 공문 제출)
- **q011** (procedural) 지역교권보호위원회는 사안 발생보고서 접수 후 며칠 이내에 열어야 하며, 연장할 수 있나요?  
  → faith 0.50 (미지지: 지역교권보호위원회는 사안 발생보고서 접수 후 21일 이내에 개최해야 한다.)
- **q012** (procedural) 지역교권보호위원회가 조치를 의결한 뒤 교육장은 언제까지 조치하고 결과를 어떻게 통지하나요?  
  → recall 0.25
- **q013** (procedural) 침해학생이 조치를 이행하지 않을 경우 학교와 교육장은 어떻게 대응하나요?  
  → 과잉 거부; recall 0.00; correct 0.00 (실패: 통보받은 날부터 90일 내 미이행 시 학교장이 미이행 학생 명단을 교육장(교권보호위원회)에 보고 / 교육장은 보고 후 21일 이내에 30일 이내 이행할 것을 서면 안내)
- **q015** (multi_evidence) 시·도교권보호위원회와 지역교권보호위원회의 심의사항은 각각 무엇이고 어떻게 다른가요?  
  → recall 0.33; correct 0.00 (실패: 지역: 침해 기준 마련·예방 대책, 침해학생 조치, 침해 보호자 등 조치, 분쟁 조정을 언급)
- **q016** (multi_evidence) 교육활동 침해학생에 대한 조치와 침해 보호자 등에 대한 조치는 종류가 어떻게 다른가요?  
  → recall 0.00; faith 0.60 (미지지: 보호자에게는 보호자 역할에 맞는 교육·치료적 조치가 적용된다. / 교육활동 침해학생에 대한 조치와 침해 보호자 등에 대한 조치는 종류와 적용 대상이 다르다.); correct 0.00 (실패: 보호자 등: 서면사과 및 재발방지 서약, 특별교육 이수 또는 심리치료 2가지)
- **q018** (multi_evidence) 특별휴가 5일을 다 쓴 뒤에도 요양이 더 필요하면 어떤 제도를 이용할 수 있고 기간은 얼마인가요?  
  → correct 0.00 (실패: 공무원연금공단 공무상 요양 승인 시 연 180일 범위에서 공무상 병가 가능)
- **q020** (summary) 피해교원에 대한 보호조치에는 어떤 유형이 있는지 정리해 주세요.  
  → recall 0.25
- **q023** (reasoning) 퇴근 후 학부모가 교사의 개인 휴대폰으로 전화해 자녀 학업 상담 중 폭언을 하고 부당한 요구를 반복했습니다. 교육활동 침해로 볼 수 있나요?  
  → recall 0.00; faith 0.00 (미지지: 학부모가 교사의 개인 휴대폰으로 퇴근 후 연락하여 폭언을 하거나 부당한 요구를 반복하는 행위는 교원의 교육활동을 부당하게 간섭하거나 제한하는 행위이다. / 학부모가 교사의 개인 휴대폰으로 퇴근 후 연락하여 폭언을 하거나 부당한 요구를 반복하는 행위는 교육활동 침해에 해당할 수 있다.)
- **q032** (unanswerable_in) 교육활동 침해 피해교원이 특별휴가를 사용하는 기간의 급여는 어떻게 처리되나요?  
  → 답 없는 질의에 답변
