"""
Tests for HnswStore — the scaled vector index.

Model-free: FAISS and numpy are real dependencies, but no embedding model is
needed. Random unit vectors exercise the same contract real embeddings do,
because the store only ever sees normalized float32 arrays.
"""

import numpy as np
import pytest

from engine.vector_store import HnswStore

DIM = 32


def _vecs(n, seed=0):
    """n random UNIT vectors — the store's cosine contract assumes normalized
    input (inner product == cosine), exactly like ingest's encode(..., normalize)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, DIM)).astype("float32")
    return x / np.linalg.norm(x, axis=1, keepdims=True)


@pytest.fixture()
def store():
    return HnswStore(dim=DIM, m=16, ef_construction=100, ef_search=64)


def test_add_returns_the_starting_faiss_id(store):
    """The id contract with SQLite: chunk rows are keyed off this offset, so if
    it ever stopped being the pre-add ntotal every citation would shift."""
    x = _vecs(10)
    assert store.add_vectors(x[:4]) == 0
    assert store.add_vectors(x[4:]) == 4
    assert len(store) == 10


def test_search_finds_the_vector_itself(store):
    x = _vecs(200)
    store.add_vectors(x)
    hits = store.search(x[7], k=5)
    assert hits[0]["index"] == 7
    assert hits[0]["score"] == pytest.approx(1.0, abs=1e-4)   # cosine with itself
    assert [h["index"] for h in hits] == sorted(
        [h["index"] for h in hits], key=lambda i: -next(
            h["score"] for h in hits if h["index"] == i))     # descending


def test_search_on_an_empty_index_is_empty_not_an_error(store):
    assert store.search(_vecs(1)[0], k=5) == []


def test_search_never_returns_more_than_exists(store):
    store.add_vectors(_vecs(3))
    assert len(store.search(_vecs(1)[0], k=50)) == 3


def test_wrong_dimension_raises_immediately(store):
    """An index built at the wrong dim must fail on add, not silently produce
    meaningless scores later."""
    with pytest.raises(ValueError):
        store.add_vectors(np.zeros((2, DIM + 1), dtype="float32"))


def test_allowed_ids_restricts_the_result_set(store):
    """The department/date/language filter. Pushed INTO the search rather than
    applied after it, so filtering cannot silently empty the result."""
    x = _vecs(300)
    store.add_vectors(x)
    allowed = {5, 11, 42, 77, 150}
    hits = store.search(x[7], k=5, allowed_ids=allowed)
    assert hits, "a filtered search must still return the best allowed matches"
    assert {h["index"] for h in hits} <= allowed
    assert 7 not in {h["index"] for h in hits}      # the true nearest is excluded


def test_an_empty_filter_returns_nothing(store):
    """A filter that matches no document must abstain, NOT fall back to the
    unfiltered corpus — that would answer from GRs the officer excluded."""
    store.add_vectors(_vecs(50))
    assert store.search(_vecs(1)[0], k=5, allowed_ids=set()) == []


def test_no_filter_searches_everything(store):
    x = _vecs(100)
    store.add_vectors(x)
    assert store.search(x[3], k=1, allowed_ids=None)[0]["index"] == 3


def test_score_for_index_is_exact_cosine(store):
    """BM25-only chunks get their score this way. IndexHNSWFlat keeps the full
    vectors, so reconstruct() is exact — the score is comparable to a dense hit's
    and the calibrated thresholds still mean what they were calibrated to mean."""
    x = _vecs(50)
    store.add_vectors(x)
    q = x[9]
    assert store.score_for_index(9, q) == pytest.approx(1.0, abs=1e-5)
    assert store.score_for_index(3, q) == pytest.approx(float(np.dot(x[3], q)), abs=1e-5)


def test_save_and_load_round_trip(tmp_path, store):
    x = _vecs(120)
    store.add_vectors(x)
    store.save(str(tmp_path))

    loaded = HnswStore.load(str(tmp_path))
    assert len(loaded) == 120
    assert loaded.dim == DIM
    assert loaded.corpus_db_path.endswith("corpus.db")
    assert loaded.search(x[42], k=1)[0]["index"] == 42


def test_load_or_new_resumes_an_existing_index(tmp_path, store):
    """What makes ingestion resumable: a second run must continue the same
    index, not start a fresh one and renumber every vector."""
    store.add_vectors(_vecs(20))
    store.save(str(tmp_path))

    resumed = HnswStore.load_or_new(str(tmp_path))
    assert len(resumed) == 20
    assert resumed.add_vectors(_vecs(5, seed=1)) == 20   # ids continue, not restart

    fresh = HnswStore.load_or_new(str(tmp_path / "elsewhere"))
    assert len(fresh) == 0


def test_ef_search_is_reapplied_on_load(tmp_path, store):
    """efSearch is a QUERY-time dial: it must be retunable without the hour of
    GPU time a rebuild would cost, so load() takes it from config, not the file."""
    store.add_vectors(_vecs(30))
    store.save(str(tmp_path))
    assert HnswStore.load(str(tmp_path), ef_search=256).index.hnsw.efSearch == 256


def test_recall_against_exact_search_is_high(store):
    """HNSW is APPROXIMATE — this pins how approximate. If a config change ever
    tanks recall, this fails instead of quietly returning worse GRs."""
    import faiss

    x = _vecs(2000, seed=3)
    store.add_vectors(x)
    exact = faiss.IndexFlatIP(DIM)
    exact.add(x)

    queries = _vecs(50, seed=99)
    hit = 0
    for q in queries:
        truth = exact.search(q.reshape(1, DIM), 10)[1][0]
        got = [h["index"] for h in store.search(q, k=10)]
        hit += len(set(truth) & set(got))
    assert hit / (10 * len(queries)) > 0.9


# --------------------------------------------------------------------------- #
# The FILTERED-search efSearch boost. It silently became a no-op once the base
# efSearch was raised to meet the ceiling that was meant to bound it.
# --------------------------------------------------------------------------- #

def test_filtered_search_ef_boost_scales_with_the_base():
    """A narrow filter makes the greedy walk spend most of its budget on nodes
    the selector rejects, so the filtered path asks for 4x efSearch. The ceiling
    that bounds it MUST stay above the base — it was hardcoded to 1024, written
    when the base was 128, and collapsed to a no-op when the base reached 1024.
    """
    from engine import config
    boosted = min(4 * config.HNSW_EF_SEARCH, config.HNSW_EF_SEARCH_MAX)
    assert boosted > config.HNSW_EF_SEARCH, (
        f"filtered efSearch {boosted} is not above the base "
        f"{config.HNSW_EF_SEARCH} — the boost is a no-op")
    assert config.HNSW_EF_SEARCH_MAX >= config.HNSW_EF_SEARCH


def test_filtered_search_still_returns_only_allowed_ids_at_a_high_ef():
    """Raising efSearch must not weaken the selector contract: a filtered search
    returns allowed ids ONLY, never a near-miss the walk happened to visit."""
    import numpy as np
    from engine.vector_store import HnswStore
    store = HnswStore(dim=8)
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(200, 8)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    store.add_vectors(vecs)
    allowed = {3, 17, 42, 99}
    hits = store.search(vecs[3], k=10, allowed_ids=allowed)
    assert hits, "a non-empty filter must return something"
    assert {h["index"] for h in hits} <= allowed
