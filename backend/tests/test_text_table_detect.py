"""Tests for text_table_detect — the OCR'd Marathi GR table detector."""

from engine import text_table_detect as ttd


# -------------------------------------------------------------------------
# fixtures: real GR text snippets (from the orgpedia corpus)
# -------------------------------------------------------------------------

PIPE_TABLE_TEXT = """\
शासन निर्णय क्रमांकः मुवाढ -२०१६/(३८/१६)/मशि-१
-------------------------------------------------------------
| अ.क्र. | विभागाचे नांव | एकूण पदे |
| पूर्णवेळ | अर्धवेळ |
| १ | उच्च शिक्षण संचालनालय, महाराष्ट्र राज्य, पुणे | ७७ |  |
| २ | सहसंचालक, उच्च शिक्षण, मुंबई विभाग, मुंबई | ४९८ |  |
| ३ | सहसंचालक, उच्च शिक्षण, पुणे विभाग, पुणे | ४६ |  |
| एकूण | १६४० | ९४  |  |
सदर शासन निर्णय गुणनियंत्रण कक्षाकडून प्राप्त झाला.
"""

PROSE_ONLY_TEXT = """\
शासन निर्णय क्रमांकः एनजीसी २०१७/(२२९/१७)/मशि-४
विद्यापीठाने शासनाकडे प्रस्ताव पाठविताना त्यात नवीन महाविद्यालय
अभ्यासक्रम विषय व वाढीव तुकड्या असे वर्गीकरण केलेले असेल.
शासनाने विहित केलेल्या निकषांची पूर्तता न करणारे प्रस्ताव
विद्यापीठाने शिफारशीत करु नयेत.
"""

MIXED_TEXT = """\
# Page 1
शासन निर्णय
This is a heading paragraph before the table.
# Page 2
शासन निर्णय क्रमांकः
| अ.क्र. | वसतीगृहाची मान्य संख्या | किमान जागा |
| १ | २४ | ४००० |
| २ | ४८ | ६००० |
| ३ | ७५ | ९१२० |
This paragraph follows the table.
# Page 3
Another page with no tables.
"""


# -------------------------------------------------------------------------
# pipe table detection
# -------------------------------------------------------------------------

def test_detect_pipe_tables_finds_a_single_table():
    lines = PIPE_TABLE_TEXT.split("\n")
    tables = ttd.detect_pipe_tables(lines)
    assert len(tables) == 1
    start, end = tables[0]
    # The range should include all pipe rows (but not prose)
    assert any("|" in lines[j] and "अ.क्र." in lines[j] for j in range(start, end))
    assert any("|" in lines[j] and "एकूण" in lines[j] for j in range(start, end))


def test_detect_pipe_tables_returns_empty_for_prose():
    lines = PROSE_ONLY_TEXT.split("\n")
    tables = ttd.detect_pipe_tables(lines)
    assert tables == []


def test_detect_pipe_tables_ignores_single_pipe_line():
    """A single pipe row is not a table — it's probably a stray OCR artifact."""
    lines = ["some text", "| single | pipe | line |", "more text"]
    tables = ttd.detect_pipe_tables(lines)
    assert tables == []


def test_detect_pipe_tables_handles_dash_separators():
    """Dash lines between pipe rows should be included in the range."""
    lines = [
        "| header1 | header2 |",
        "|----|-----|",
        "| data1 | data2 |",
    ]
    tables = ttd.detect_pipe_tables(lines)
    assert len(tables) == 1
    start, end = tables[0]
    assert end - start == 3  # all three lines


# -------------------------------------------------------------------------
# pipe table parsing
# -------------------------------------------------------------------------

def test_parse_pipe_table_extracts_header_and_rows():
    lines = [
        "| अ.क्र. | विभागाचे नांव | एकूण पदे |",
        "| १ | उच्च शिक्षण संचालनालय | ७७ |",
        "| २ | सहसंचालक | ४६ |",
    ]
    header, rows = ttd.parse_pipe_table(lines)
    assert header == ["अ.क्र.", "विभागाचे नांव", "एकूण पदे"]
    assert len(rows) == 2
    assert rows[0][0] == "१"
    assert rows[0][2] == "७७"


def test_parse_pipe_table_skips_dash_lines():
    lines = [
        "| H1 | H2 |",
        "|----|-----|",
        "| d1 | d2 |",
    ]
    header, rows = ttd.parse_pipe_table(lines)
    assert header == ["H1", "H2"]
    assert len(rows) == 1
    assert rows[0] == ["d1", "d2"]


def test_parse_pipe_table_needs_at_least_2_rows():
    lines = ["| only | one | row |"]
    header, rows = ttd.parse_pipe_table(lines)
    assert header is None
    assert rows == []


# -------------------------------------------------------------------------
# split_prose_and_tables (the full pipeline)
# -------------------------------------------------------------------------

def test_split_prose_and_tables_creates_table_chunks():
    pages = [(1, PIPE_TABLE_TEXT)]
    prose_pages, table_chunks = ttd.split_prose_and_tables(pages)
    assert len(table_chunks) >= 1
    for tc in table_chunks:
        assert tc["content_type"] == "table"
        assert "In this row:" in tc["text"]  # row_to_sentence format


def test_split_prose_and_tables_removes_table_from_prose():
    pages = [(1, PIPE_TABLE_TEXT)]
    prose_pages, _ = ttd.split_prose_and_tables(pages)
    prose_text = prose_pages[0][1]
    # The pipe table rows should NOT be in the prose anymore
    assert "| अ.क्र." not in prose_text
    assert "| १ |" not in prose_text
    # But surrounding prose should remain
    assert "शासन निर्णय" in prose_text


def test_split_prose_and_tables_no_tables():
    pages = [(1, PROSE_ONLY_TEXT)]
    prose_pages, table_chunks = ttd.split_prose_and_tables(pages)
    assert table_chunks == []
    assert prose_pages == pages  # unchanged


def test_split_prose_and_tables_multi_page():
    """Tables on page 2 should have correct page attribution."""
    import re
    text = MIXED_TEXT
    parts = re.split(r"(?m)^#\s*Page\s*(\d+)\s*$", text)
    pages = []
    i = 1
    while i < len(parts):
        pages.append((int(parts[i]), parts[i + 1] if i + 1 < len(parts) else ""))
        i += 2

    prose_pages, table_chunks = ttd.split_prose_and_tables(pages)
    assert len(table_chunks) >= 1
    # The table is on page 2
    assert table_chunks[0]["page_start"] == 2
    # Page 1 and page 3 prose should be intact
    page1_prose = next(t for n, t in prose_pages if n == 1)
    assert "heading paragraph" in page1_prose
    page3_prose = next(t for n, t in prose_pages if n == 3)
    assert "no tables" in page3_prose


def test_split_prose_and_tables_preserves_prose_after_table():
    """Text following a table on the same page must survive in prose."""
    import re
    parts = re.split(r"(?m)^#\s*Page\s*(\d+)\s*$", MIXED_TEXT)
    pages = []
    i = 1
    while i < len(parts):
        pages.append((int(parts[i]), parts[i + 1] if i + 1 < len(parts) else ""))
        i += 2

    prose_pages, _ = ttd.split_prose_and_tables(pages)
    page2_prose = next(t for n, t in prose_pages if n == 2)
    assert "follows the table" in page2_prose
    assert "| अ.क्र." not in page2_prose


# -------------------------------------------------------------------------
# row_to_sentence integration
# -------------------------------------------------------------------------

def test_table_chunks_contain_prose_sentences():
    """The table chunks should use row_to_sentence, not raw pipe text."""
    pages = [(1, PIPE_TABLE_TEXT)]
    _, table_chunks = ttd.split_prose_and_tables(pages)
    assert len(table_chunks) >= 1
    chunk_text = table_chunks[0]["text"]
    # Must start with a caption
    assert chunk_text.startswith("Table:")
    # Must contain "In this row:" from row_to_sentence
    assert "In this row:" in chunk_text
    # Must NOT contain raw pipes
    assert " | " not in chunk_text.split("\n", 1)[1]  # after the caption line


def test_devanagari_numerals_in_table():
    """Devanagari numerals (१, २, ...) should be treated as cell values."""
    lines = [
        "| क्र. | रक्कम |",
        "| १ | ५,००० |",
        "| २ | १०,००० |",
    ]
    header, rows = ttd.parse_pipe_table(lines)
    assert rows[0][0] == "१"
    assert rows[0][1] == "५,०००"
    assert rows[1][1] == "१०,०००"
