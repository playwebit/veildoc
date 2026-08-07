import numpy as np

from veildoc.embeddings.base import EmbeddingBackend
from veildoc.retrieval import Retriever


class FakeEmbedding(EmbeddingBackend):
    """Deterministic fake backend for testing retrieval logic without
    downloading a real model. Returns pre-defined vectors keyed by
    exact text match, falling back to a zero vector otherwise.
    """

    def __init__(self, vectors: dict[str, list[float]]):
        self._vectors = {k: np.array(v, dtype=float) for k, v in vectors.items()}

    def embed(self, texts):
        out = np.array([self._vectors.get(t, np.zeros(2)) for t in texts])
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(norms, 1e-12, None)

    def embed_query(self, query):
        return self.embed([query])[0]


def test_retrieve_returns_most_similar_chunk():
    chunks = ["about cats", "about dogs"]
    backend = FakeEmbedding({
        "about cats": [1, 0],
        "about dogs": [0, 1],
        "tell me about cats please": [1, 0],
    })
    retriever = Retriever(backend, chunks)
    results = retriever.retrieve("tell me about cats please", k=1)
    assert results[0].text == "about cats"
    assert results[0].index == 0


def test_retrieve_k_greater_than_one_returns_ranked_list():
    chunks = ["alpha", "beta", "gamma"]
    backend = FakeEmbedding({
        "alpha": [1, 0],
        "beta": [0.7, 0.7],
        "gamma": [0, 1],
        "query": [1, 0],
    })
    retriever = Retriever(backend, chunks)
    results = retriever.retrieve("query", k=2)
    assert len(results) == 2
    assert results[0].text == "alpha"  # closest match first
    assert results[0].score >= results[1].score  # ranked descending


def test_rank_of_finds_correct_position():
    chunks = ["alpha", "beta", "gamma"]
    backend = FakeEmbedding({
        "alpha": [1, 0],
        "beta": [0.7, 0.7],
        "gamma": [0, 1],
        "query": [0, 1],  # closest to gamma
    })
    retriever = Retriever(backend, chunks)
    assert retriever.rank_of("query", chunk_index=2) == 1  # gamma is rank 1
    assert retriever.rank_of("query", chunk_index=0) == 3  # alpha is rank 3 (least similar)


def test_exposure_pct_computes_correctly():
    chunks = ["short chunk"]
    backend = FakeEmbedding({"short chunk": [1, 0], "q": [1, 0]})
    retriever = Retriever(backend, chunks)
    results = retriever.retrieve("q", k=1)
    full_doc = "short chunk" * 10  # retrieved chunk is 1/10th of "document"
    pct = retriever.exposure_pct(results, full_doc)
    assert abs(pct - 10.0) < 0.01


def test_empty_chunks_raises():
    backend = FakeEmbedding({})
    try:
        Retriever(backend, [])
        assert False, "expected ValueError"
    except ValueError:
        pass
