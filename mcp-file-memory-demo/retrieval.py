import re
from collections import Counter

TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]+")


def tokenize(text):
    return [token.lower() for token in TOKEN_RE.findall(text)]


def lexical_rank(query, docs):
    query_terms = tokenize(query)
    ranked = []
    for doc in docs:
        doc_terms = Counter(tokenize(doc.get("title", "") + "\n" + doc.get("content", "")))
        score = sum(doc_terms.get(term, 0) for term in query_terms)
        if score > 0:
            ranked.append((doc["doc_id"], score))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return [doc_id for doc_id, _score in ranked]


def semantic_mock_rank(query, docs):
    """Tiny semantic mock: uses set overlap as a stand-in for vector similarity."""
    query_terms = set(tokenize(query))
    ranked = []
    for doc in docs:
        doc_terms = set(tokenize(doc.get("title", "") + "\n" + doc.get("content", "")))
        score = len(query_terms & doc_terms)
        if score > 0:
            ranked.append((doc["doc_id"], score))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return [doc_id for doc_id, _score in ranked]


def rrf_fuse(rank_lists, k=60):
    scores = {}
    for rank_list in rank_lists:
        for rank, doc_id in enumerate(rank_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return [doc_id for doc_id, _score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]
