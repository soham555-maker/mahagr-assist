"""
Tests for engine/graph.py — the supersede / citation knowledge graph.

Model-free and corpus-free: every fixture is a handful of synthetic documents in
a tmp SQLite file, which is the whole point — the graph is built from metadata
Phase 2 already stored, so none of this needs the 18k-GR corpus or a GPU.
"""

import pytest

from engine import corpus_db, graph
from engine.gr_metadata import canonical_number


@pytest.fixture()
def conn(tmp_path):
    path = str(tmp_path / "corpus.db")
    corpus_db.init(path)
    with corpus_db.connect(path) as c:
        yield c


def _doc(conn, doc_id, number, refs=(), supersedes=False, date="2020-01-01"):
    corpus_db.upsert_document(
        conn, doc_id,
        {"gr_number": number, "department": "Higher and Technical Education Department",
         "date": date, "category": "government resolution (GR)", "language": "mr",
         "title": f"title {doc_id}", "references": list(refs),
         "supersedes": supersedes},
        f"text {doc_id}", 1)


# --------------------------------------------------------------------------- #
# canonical_number — the actual hard part
# --------------------------------------------------------------------------- #

def test_canonical_number_absorbs_ocr_variation():
    """The same GR number, printed two ways by OCR, must canonicalise equal."""
    a = canonical_number("संकीर्ण-२०२३/प्र.क्र.४५/तांशि-४")
    b = canonical_number("संकीर्ण 2023 / प्र. क्र. ४५ / तांशि ४")
    assert a == b and a


def test_canonical_number_keeps_different_numbers_different():
    """Over-normalising would silently merge unrelated orders — worse than
    missing an edge, because it would fabricate a supersession."""
    assert canonical_number("GR-12") != canonical_number("GR-123")
    assert canonical_number("अ-२०२३/१/x") != canonical_number("अ-२०२३/२/x")
    # '/' is structural and must survive
    assert "/" in canonical_number("संकीर्ण-२०२३/प्र.क्र.४५")


def test_canonical_number_rejects_meaningless_input():
    for junk in ("", None, "---", "  ", "..."):
        assert canonical_number(junk) is None


# --------------------------------------------------------------------------- #
# building
# --------------------------------------------------------------------------- #

def test_edges_resolve_across_ocr_spelling(conn):
    """The end-to-end reason canonicalisation exists: B references A in a
    differently-spaced form and the edge must still resolve."""
    _doc(conn, "a", "संकीर्ण-२०२३/प्र.क्र.४५/तांशि-४")
    _doc(conn, "b", "संकीर्ण-२०२४/प्र.क्र.१२/तांशि-४",
         refs=["संकीर्ण 2023 / प्र. क्र. ४५ / तांशि ४"], supersedes=True)

    counts = graph.build_edges(conn)
    assert counts["resolved"] == 1 and counts["dangling"] == 0
    assert [e["dst_id"] for e in graph.outgoing(conn, "b")] == ["a"]
    assert [e["src_id"] for e in graph.incoming(conn, "a")] == ["b"]


def test_unresolvable_references_are_kept_as_dangling(conn):
    """A reference to a GR we don't hold is INFORMATION, not an error — and the
    dangling rate is how we measure corpus completeness honestly."""
    _doc(conn, "a", "GR-1", refs=["GR-NOT-IN-CORPUS"])
    counts = graph.build_edges(conn)
    assert counts["dangling"] == 1 and counts["resolved"] == 0
    edge = graph.outgoing(conn, "a")[0]
    assert edge["resolution"] == "dangling" and edge["dst_id"] is None
    assert edge["dst_number"] == "GR-NOT-IN-CORPUS"   # printed form preserved


def test_colliding_numbers_are_ambiguous_not_guessed(conn):
    """Two documents sharing a canonical number must NOT produce a confident
    edge to an arbitrary one of them."""
    _doc(conn, "a1", "GR-7")
    _doc(conn, "a2", "GR 7")            # canonicalises to the same thing
    _doc(conn, "b", "GR-9", refs=["GR-7"])
    counts = graph.build_edges(conn)
    assert counts["ambiguous"] == 1 and counts["resolved"] == 0
    assert graph.outgoing(conn, "b")[0]["dst_id"] is None


def test_the_cited_date_breaks_a_number_collision(conn):
    """Two GRs share a number; the reference names a date. That date is the only
    thing that can say which one is meant — and it is printed on nearly every
    reference line, so using it converts ambiguity into real edges."""
    _doc(conn, "a1", "GR-7", date="2019-05-04")
    _doc(conn, "a2", "GR 7", date="2021-11-30")
    _doc(conn, "b", "GR-9", date="2022-01-01",
         refs=[{"number": "GR-7", "date": "2021-11-30"}])
    counts = graph.build_edges(conn)
    assert counts["resolved"] == 1 and counts["resolved_by_date"] == 1
    assert counts["ambiguous"] == 0
    assert graph.outgoing(conn, "b")[0]["dst_id"] == "a2"


def test_a_cited_date_matching_nobody_stays_ambiguous(conn):
    """Never guess. If the date rules out every candidate, the edge is
    unresolved — a wrong GR in front of an officer is worse than a gap."""
    _doc(conn, "a1", "GR-7", date="2019-05-04")
    _doc(conn, "a2", "GR 7", date="2021-11-30")
    _doc(conn, "b", "GR-9", refs=[{"number": "GR-7", "date": "2015-01-01"}])
    counts = graph.build_edges(conn)
    assert counts["ambiguous"] == 1 and counts["resolved"] == 0


def test_the_cited_date_is_stored_on_the_edge(conn):
    """Kept even when the edge dangles: 'GR-X of 30/10/2010, which we do not
    hold' is what the officer needs to go and find it."""
    _doc(conn, "b", "GR-9", refs=[{"number": "GR-MISSING", "date": "2010-10-30"}])
    graph.build_edges(conn)
    assert graph.outgoing(conn, "b")[0]["dst_date"] == "2010-10-30"


def test_a_corpus_of_plain_string_refs_still_builds(conn):
    """Backwards compatibility: a corpus ingested before dates were parsed holds
    refs as bare strings. Re-embedding 18k documents to change a JSON shape is
    not an acceptable price, so the reader absorbs both."""
    _doc(conn, "a", "GR-1")
    _doc(conn, "b", "GR-2", refs=["GR-1"])          # old shape
    _doc(conn, "c", "GR-3", refs=[{"number": "GR-1", "date": None}])   # new shape
    counts = graph.build_edges(conn)
    assert counts["resolved"] == 2


def test_a_document_never_cites_itself(conn):
    _doc(conn, "a", "GR-1", refs=["GR-1"])
    graph.build_edges(conn)
    assert graph.outgoing(conn, "a") == []


def test_build_is_idempotent(conn):
    _doc(conn, "a", "GR-1")
    _doc(conn, "b", "GR-2", refs=["GR-1"], supersedes=True)
    first = graph.build_edges(conn)
    second = graph.build_edges(conn)
    assert first == second
    assert graph.stats(conn)["edges"] == 1


def test_kind_reflects_the_supersede_flag(conn):
    _doc(conn, "a", "GR-1")
    _doc(conn, "b", "GR-2", refs=["GR-1"], supersedes=True)
    _doc(conn, "c", "GR-3", refs=["GR-1"], supersedes=False)
    graph.build_edges(conn)
    kinds = {e["src_id"]: e["kind"] for e in graph.incoming(conn, "a")}
    assert kinds == {"b": graph.SUPERSEDES, "c": graph.CITES}


def test_duplicate_references_collapse(conn):
    _doc(conn, "a", "GR-1")
    _doc(conn, "b", "GR-2", refs=["GR-1", "GR 1", "GR-1"])
    graph.build_edges(conn)
    assert len(graph.outgoing(conn, "b")) == 1


# --------------------------------------------------------------------------- #
# traversal
# --------------------------------------------------------------------------- #

def test_supersede_chain_is_transitive(conn):
    """C supersedes B supersedes A. Asking about A must surface BOTH — telling
    an officer 'replaced by B' is misleading when B is itself dead."""
    _doc(conn, "a", "GR-1", date="2019-01-01")
    _doc(conn, "b", "GR-2", refs=["GR-1"], supersedes=True, date="2021-01-01")
    _doc(conn, "c", "GR-3", refs=["GR-2"], supersedes=True, date="2023-01-01")
    graph.build_edges(conn)
    assert [n["id"] for n in graph.supersede_chain(conn, "a")] == ["b", "c"]
    assert graph.supersede_chain(conn, "c") == []


def test_supersede_chain_survives_a_cycle(conn):
    """Two GRs superseding each other must not hang the traversal — this runs
    behind an HTTP request."""
    _doc(conn, "a", "GR-1", refs=["GR-2"], supersedes=True)
    _doc(conn, "b", "GR-2", refs=["GR-1"], supersedes=True)
    graph.build_edges(conn)
    chain = graph.supersede_chain(conn, "a")
    assert len(chain) <= 2
    assert len({n["id"] for n in chain}) == len(chain)   # no repeats


def test_supersede_chain_respects_max_depth(conn):
    for i in range(8):
        _doc(conn, f"d{i}", f"GR-{i}",
             refs=[f"GR-{i-1}"] if i else [], supersedes=bool(i),
             date=f"20{10+i}-01-01")
    graph.build_edges(conn)
    assert len(graph.supersede_chain(conn, "d0", max_depth=3)) == 3


def test_chain_picks_the_newest_when_several_claim_supersession(conn):
    _doc(conn, "a", "GR-1", date="2019-01-01")
    _doc(conn, "old", "GR-8", refs=["GR-1"], supersedes=True, date="2020-01-01")
    _doc(conn, "new", "GR-9", refs=["GR-1"], supersedes=True, date="2024-01-01")
    graph.build_edges(conn)
    assert graph.supersede_chain(conn, "a")[0]["id"] == "new"


def test_neighbourhood_returns_nodes_edges_and_ghosts(conn):
    _doc(conn, "a", "GR-1", refs=["GR-MISSING"])
    _doc(conn, "b", "GR-2", refs=["GR-1"], supersedes=True)
    graph.build_edges(conn)

    n = graph.neighbourhood(conn, "a", depth=1)
    assert n["found"] is True
    assert {x["id"] for x in n["nodes"]} == {"a", "b"}
    assert n["edges"] == [{"src": "b", "dst": "a", "kind": graph.SUPERSEDES}]
    assert [d["gr_number"] for d in n["dangling"]] == ["GR-MISSING"]
    assert [c["id"] for c in n["chain"]] == ["b"]


def test_neighbourhood_of_an_unknown_document(conn):
    assert graph.neighbourhood(conn, "nope")["found"] is False


def test_neighbourhood_depth_expands_the_graph(conn):
    _doc(conn, "a", "GR-1")
    _doc(conn, "b", "GR-2", refs=["GR-1"])
    _doc(conn, "c", "GR-3", refs=["GR-2"])
    graph.build_edges(conn)
    assert {x["id"] for x in graph.neighbourhood(conn, "a", depth=1)["nodes"]} == {"a", "b"}
    assert {x["id"] for x in graph.neighbourhood(conn, "a", depth=2)["nodes"]} == {"a", "b", "c"}


def test_neighbourhood_caps_node_count_and_keeps_edges_consistent(conn):
    """A hub GR in an 18k corpus can pull in hundreds of neighbours. The cap
    must never leave an edge pointing at a node the client wasn't given."""
    _doc(conn, "hub", "GR-0")
    for i in range(1, 30):
        _doc(conn, f"n{i}", f"GR-{i}", refs=["GR-0"])
    graph.build_edges(conn)
    n = graph.neighbourhood(conn, "hub", depth=1, max_nodes=10)
    ids = {x["id"] for x in n["nodes"]}
    assert len(ids) <= 10
    assert all(e["src"] in ids and e["dst"] in ids for e in n["edges"])


def test_neighbourhood_terminates_on_a_cycle(conn):
    _doc(conn, "a", "GR-1", refs=["GR-2"])
    _doc(conn, "b", "GR-2", refs=["GR-1"])
    graph.build_edges(conn)
    n = graph.neighbourhood(conn, "a", depth=5)
    assert {x["id"] for x in n["nodes"]} == {"a", "b"}


def test_stats_reports_resolution_breakdown(conn):
    _doc(conn, "a", "GR-1")
    _doc(conn, "b", "GR-2", refs=["GR-1", "GR-GONE"], supersedes=True)
    graph.build_edges(conn)
    s = graph.stats(conn)
    assert s["edges"] == 2
    assert s["by_resolution"] == {"resolved": 1, "dangling": 1}
    assert s["by_kind"] == {graph.SUPERSEDES: 2}
    assert s["documents_with_resolved_edges"] == 1


def test_backfill_populates_canonical_numbers(conn):
    """Existing corpora were ingested before this column existed; the migration
    must fill them in rather than force a 25-minute re-embed."""
    _doc(conn, "a", "संकीर्ण-२०२३/४५")
    conn.execute("UPDATE gr_documents SET gr_number_canon=NULL")
    assert corpus_db.backfill_canonical_numbers(conn) == 1
    row = conn.execute("SELECT gr_number_canon FROM gr_documents WHERE id='a'").fetchone()
    assert row[0] == canonical_number("संकीर्ण-२०२३/४५")


# --------------------------------------------------------------------------- #
# officer.supersession / supersede_warnings on the graph
# --------------------------------------------------------------------------- #

class _Store:
    """Minimal stand-in for HnswStore — officer only reads corpus_db_path."""
    def __init__(self, path):
        self.corpus_db_path = path


def _chain_corpus(conn):
    """A -> B -> C: each supersedes the previous."""
    _doc(conn, "a", "GR-1", date="2019-01-01")
    _doc(conn, "b", "GR-2", refs=["GR-1"], supersedes=True, date="2021-01-01")
    _doc(conn, "c", "GR-3", refs=["GR-2"], supersedes=True, date="2023-01-01")


def test_supersession_uses_the_graph_when_it_is_built(tmp_path, conn):
    from engine import officer
    _chain_corpus(conn)
    graph.build_edges(conn)
    store = _Store(conn.execute("PRAGMA database_list").fetchone()[2])

    info = officer.supersession(store, "a")
    assert info["found"] is True
    assert [c["gr_number"] for c in info["supersede_chain"]] == ["GR-2", "GR-3"]
    assert info["superseded_by"][0]["gr_number"] == "GR-2"


def test_supersession_falls_back_when_the_graph_is_not_built(conn):
    """A freshly ingested corpus has no edges yet — the endpoint must still
    answer rather than silently going blank."""
    from engine import officer
    _chain_corpus(conn)                       # note: build_edges NOT called
    # officer opens its OWN connection, so the rows must be committed first.
    # (The graph-backed tests get this for free because build_edges commits.)
    conn.commit()
    store = _Store(conn.execute("PRAGMA database_list").fetchone()[2])

    info = officer.supersession(store, "a")
    assert info["found"] is True
    assert info["supersede_chain"] == []
    assert [s["gr_number"] for s in info["superseded_by"]] == ["GR-2"]


def test_warning_names_the_order_actually_in_force(conn):
    """The point of the chain: 'replaced by B' is misleading when B is dead too."""
    from engine import officer
    _chain_corpus(conn)
    graph.build_edges(conn)
    store = _Store(conn.execute("PRAGMA database_list").fetchone()[2])

    warnings = officer.supersede_warnings(store, [{"gr_number": "GR-1"}])
    assert len(warnings) == 1
    assert "GR-2" in warnings[0] and "GR-3" in warnings[0]
    assert "latest" in warnings[0]


def test_warning_is_simple_for_a_single_supersession(conn):
    from engine import officer
    _doc(conn, "a", "GR-1")
    _doc(conn, "b", "GR-2", refs=["GR-1"], supersedes=True, date="2022-02-02")
    graph.build_edges(conn)
    store = _Store(conn.execute("PRAGMA database_list").fetchone()[2])

    warnings = officer.supersede_warnings(store, [{"gr_number": "GR-1"}])
    assert len(warnings) == 1 and "GR-2" in warnings[0] and "latest" not in warnings[0]


def test_no_warning_for_a_current_gr(conn):
    from engine import officer
    _chain_corpus(conn)
    graph.build_edges(conn)
    store = _Store(conn.execute("PRAGMA database_list").fetchone()[2])
    assert officer.supersede_warnings(store, [{"gr_number": "GR-3"}]) == []
