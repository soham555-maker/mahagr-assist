"""
text_table_detect.py — detect and extract tables from OCR'd Marathi GR text.

The 18,078-GR corpus is pre-extracted `.mr.txt` files from orgpedia's OCR.
These have no PDF structure, so pdfplumber cannot help. Instead we detect
table-like patterns directly in the text and convert them to the same
`row_to_sentence` prose that the PDF pipeline uses for born-digital tables.

WHY THIS EXISTS
---------------
Tables embedded as raw OCR text (column headers jumbled into a word stream)
embed poorly with bge-m3: the model is trained on prose, not grids. A fee
schedule like "| अ.क्र. | प्रवर्ग | शुल्क | | १ | खुला | ५,००० |" becomes
"In this row: अ.क्र. is 1, प्रवर्ग is खुला, शुल्क is 5,000" — a sentence
a question about fees actually matches.

FORMATS HANDLED (measured across 18,078 GRs)
---------------------------------------------
1. PIPE-DELIMITED TABLES (7,611 GRs, 42% of the corpus):
   | अ.क्र. | विभागाचे नांव | एकूण पदे |
   | १ | उच्च शिक्षण संचालनालय | ७७ |
   The dominant format. orgpedia's OCR consistently uses '|' for table cells.

2. DASH-SEPARATED TABLES (1,805 GRs):
   Lines of '----' marking table boundaries, with data rows between them.
   Handled as a variant of pipe tables (the dashes mark the extent, cells
   are whitespace-separated or pipe-separated between them).

Serial-number lists (१., २., ...) are NOT parsed as tables — they are too
often just numbered prose paragraphs, and the false-positive rate is high.
They are left in prose where the embedding model handles them reasonably.

CONSERVATIVE BY DESIGN
----------------------
A false positive (prose parsed as a table, then converted to broken
row_to_sentence output) is worse than a false negative (a real table left
as raw text in a prose chunk, which is no worse than today). So:
- Minimum 3 pipe characters per line to count as a table row
- Minimum 2 rows to form a table
- Lines must have CONSISTENT pipe count (±1, allowing for OCR errors)
"""

import re
from engine.table_extract import row_to_sentence


# --------------------------------------------------------------------------- #
# pipe-table detection
# --------------------------------------------------------------------------- #

# A line that looks like a pipe-delimited table row: starts with optional
# whitespace, then '|', and has at least 2 more '|' characters.
_PIPE_ROW_RE = re.compile(r"^\s*\|.*\|.*\|")

# A dash separator line (e.g. "-----" or "| --- | --- |")
_DASH_LINE_RE = re.compile(r"^\s*[-|─—\s]{4,}\s*$")


def _is_pipe_row(line):
    """True if this line is a pipe-delimited table row."""
    return bool(_PIPE_ROW_RE.match(line)) and not bool(_DASH_LINE_RE.match(line))


def _parse_pipe_row(line):
    """Split a pipe-delimited line into cell values, stripping outer pipes."""
    # "| अ.क्र. | विभागाचे नांव | एकूण |" -> ["अ.क्र.", "विभागाचे नांव", "एकूण"]
    cells = line.split("|")
    # Drop the empty strings from the leading and trailing pipes.
    return [c.strip() for c in cells if c.strip() or (c and not c.strip())]


def _clean_cells(cells):
    """Normalize whitespace in each cell, dropping fully empty ones only from
    the tail (inner empty cells are meaningful — an empty column)."""
    cleaned = [" ".join(c.split()) for c in cells]
    # Trim trailing empties (OCR sometimes adds extra pipes at the end)
    while cleaned and not cleaned[-1]:
        cleaned.pop()
    return cleaned


def detect_pipe_tables(lines):
    """Find contiguous runs of pipe-delimited rows in the text.

    Returns [(start_line_idx, end_line_idx), ...] — half-open ranges into
    `lines`. Dash-separator lines between pipe rows are included in the range
    (they are header/footer separators, not data). A run must have at least
    2 pipe rows to qualify.
    """
    tables = []
    i = 0
    while i < len(lines):
        if _is_pipe_row(lines[i]):
            start = i
            while i < len(lines) and (_is_pipe_row(lines[i]) or _DASH_LINE_RE.match(lines[i])):
                i += 1
            # Count actual pipe rows (not dash lines)
            pipe_count = sum(1 for j in range(start, i) if _is_pipe_row(lines[j]))
            if pipe_count >= 2:
                tables.append((start, i))
        else:
            i += 1
    return tables


def parse_pipe_table(lines):
    """Parse a block of pipe-delimited lines into (header, [data_rows]).

    The first pipe row is taken as the header. Dash-separator lines are
    skipped. Returns (header_cells, [row_cells, ...]).
    """
    pipe_rows = [_clean_cells(_parse_pipe_row(line))
                 for line in lines if _is_pipe_row(line)]
    if len(pipe_rows) < 2:
        return None, []

    header = pipe_rows[0]
    data = pipe_rows[1:]

    # Normalize: pad shorter rows to header width
    width = len(header)
    data = [row + [""] * max(0, width - len(row)) for row in data]

    return header, data


# --------------------------------------------------------------------------- #
# the public interface: split prose and tables
# --------------------------------------------------------------------------- #

def _page_for_line(line_idx, page_boundaries):
    """Which page does this line index belong to?

    page_boundaries: [(page_number, start_line_idx), ...] sorted by start_line.
    Returns the page number.
    """
    page = 1
    for pnum, pstart in page_boundaries:
        if line_idx >= pstart:
            page = pnum
        else:
            break
    return page


def _table_to_chunks(header, data_rows, page_start, page_end, chunk_size=250):
    """Convert a parsed table into one or more chunks of row sentences.

    Mirrors ingest._chunk_one_table: packs rows until chunk_size words, then
    starts a new chunk. The header caption is repeated on each chunk.
    """
    if not header or not data_rows:
        return []

    # Build a caption from the header row
    caption = "Table: " + " | ".join(h for h in header if h)

    sentences = [s for s in (row_to_sentence(header, row) for row in data_rows) if s]
    if not sentences:
        return []

    cap_len = len(caption.split())
    budget = max(1, chunk_size - cap_len)

    chunks, current, current_len = [], [], 0
    for sent in sentences:
        slen = len(sent.split())
        if current and current_len + slen > budget:
            chunks.append({
                "text": (caption + "\n" + "\n".join(current)).strip(),
                "page_start": page_start,
                "page_end": page_end,
                "content_type": "table",
            })
            current, current_len = [], 0
        current.append(sent)
        current_len += slen

    if current:
        chunks.append({
            "text": (caption + "\n" + "\n".join(current)).strip(),
            "page_start": page_start,
            "page_end": page_end,
            "content_type": "table",
        })
    return chunks


def split_prose_and_tables(pages, chunk_size=250):
    """Split (page_number, text) pages into prose pages and table chunks.

    Returns (prose_pages, table_chunks):
      prose_pages  — [(page_number, text_with_tables_removed), ...]
      table_chunks — [{'text', 'page_start', 'page_end', 'content_type'}, ...]

    The table text is REMOVED from the prose pages so it doesn't appear in
    both a prose chunk and a table chunk. This mirrors what process_pdf does.
    """
    # Flatten all pages into a single list of lines, tracking page boundaries
    all_lines = []
    page_boundaries = []  # [(page_number, start_line_idx)]
    for page_number, text in pages:
        page_boundaries.append((page_number, len(all_lines)))
        all_lines.extend(text.split("\n"))

    # Detect pipe tables across the full text
    table_ranges = detect_pipe_tables(all_lines)

    if not table_ranges:
        return pages, []

    # Build the set of line indices that belong to tables
    table_line_set = set()
    for start, end in table_ranges:
        table_line_set.update(range(start, end))

    # Parse each table and convert to chunks
    table_chunks = []
    for start, end in table_ranges:
        table_lines = all_lines[start:end]
        header, data_rows = parse_pipe_table(table_lines)
        if header is None:
            continue

        page_start = _page_for_line(start, page_boundaries)
        page_end = _page_for_line(end - 1, page_boundaries)
        table_chunks.extend(
            _table_to_chunks(header, data_rows, page_start, page_end, chunk_size)
        )

    # Rebuild prose pages with table lines removed
    prose_pages = []
    for page_number, text in pages:
        lines = text.split("\n")
        # Find the absolute line indices for this page's lines
        page_start_idx = None
        for pnum, pidx in page_boundaries:
            if pnum == page_number:
                page_start_idx = pidx
                break
        if page_start_idx is None:
            prose_pages.append((page_number, text))
            continue

        kept = []
        for i, line in enumerate(lines):
            abs_idx = page_start_idx + i
            if abs_idx not in table_line_set:
                kept.append(line)
        prose_pages.append((page_number, "\n".join(kept)))

    return prose_pages, table_chunks
