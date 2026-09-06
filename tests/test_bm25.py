import pytest

from rag.bm25 import BM25Index, rrf_fuse, tokenize, tokenize_kiwi
from tests.conftest import make_doc


def test_tokenize_char_bigrams_ignore_whitespace_and_case():
    assert tokenize("시·도 교권") == ["시·", "·도", "도교", "교권"]
    assert tokenize("AB c") == ["ab", "bc"]
    assert tokenize("가") == ["가"]
    assert tokenize("  ") == []


def test_bm25_index_prefers_lexical_match():
    docs = [
        make_doc(
            "시·도교권보호위원회의 위원은 10명 이상 20명 이하로 구성한다.",
            page=23,
            idx=0,
        ),
        make_doc(
            "지역교권보호위원회의 위원은 10명 이상 50명 이하로 구성한다.",
            page=26,
            idx=0,
        ),
        make_doc(
            "분쟁조정은 30일 이내에 성립하지 않으면 종료할 수 있다.", page=65, idx=0
        ),
    ]
    idx = BM25Index(docs)
    results = idx.search("시·도교권보호위원회 위원 정수", k=3)
    assert results[0][0].metadata["page"] == 23
    assert all(score > 0 for _, score in results)
    assert idx.by_chunk_id["manual:p23:c0"].metadata["page"] == 23


def test_bm25_search_drops_zero_scores():
    idx = BM25Index([make_doc("완전히 다른 내용입니다.", page=1)])
    assert idx.search("zzzz", k=5) == []


def test_rrf_fuse_rewards_documents_ranked_in_both_lists():
    fused = rrf_fuse([["a", "b", "c"], ["b", "d", "a"]], k=60)
    order = [cid for cid, _ in fused]
    assert order[:2] == ["b", "a"]  # 양쪽 상위인 b, a 가 한쪽만 상위인 d, c 보다 앞
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_fuse_single_list_keeps_order():
    assert [cid for cid, _ in rrf_fuse([["x", "y"]])] == ["x", "y"]


def test_tokenize_kiwi_keeps_content_words_and_compound_nouns():
    toks = tokenize_kiwi(
        "시·도교권보호위원회의 위원 정수와 임기는 어떻게 정해져 있나요?"
    )
    assert "시·도" in toks and "위원회" in toks and "임기" in toks
    assert (
        "의" not in toks and "는" not in toks and "?" not in toks
    )  # 조사·어미·기호 제거


def test_tokenize_kiwi_keeps_numbers_and_prefixes():
    toks = tokenize_kiwi("사안 접수 후 24시간 이내 보고, 제25조제7항")
    assert "24" in toks and "25" in toks and "제" in toks and "이내" in toks


def test_tokenize_dispatch_by_name():
    assert tokenize("교권 보호", "bigram") == ["교권", "권보", "보호"]
    assert tokenize("교권 보호", "kiwi") == ["교권", "보호"]


def test_bm25_kiwi_distinguishes_similar_committees():
    docs = [
        make_doc(
            "시·도교권보호위원회의 위원은 10명 이상 20명 이하로 구성한다.",
            page=23,
            idx=0,
        ),
        make_doc(
            "지역교권보호위원회의 위원은 10명 이상 50명 이하로 구성한다.",
            page=26,
            idx=0,
        ),
        make_doc(
            "분쟁조정은 30일 이내에 성립하지 않으면 종료할 수 있다.", page=65, idx=0
        ),  # 2개뿐이면 IDF 가 퇴화
    ]
    idx = BM25Index(docs, tokenizer=tokenize_kiwi)
    assert idx.search("시·도교권보호위원회 위원 정수", k=2)[0][0].metadata["page"] == 23
    assert idx.search("지역교권보호위원회 위원 정수", k=2)[0][0].metadata["page"] == 26
