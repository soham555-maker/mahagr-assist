"""
Phase 2 — standalone table extraction experiment.

Separate from ingest.py. Does NOT embed anything. Its job: open a PDF, find the
*real* data tables, and print each row next to the sentence we'd embed.

WHY TWO DETECTION METHODS (the Phase 2 story)
---------------------------------------------
Phase 1 used one method: page-level detection with vertical_strategy="lines",
horizontal_strategy="text". That works for tables with ruled column lines
(e.g. the summary-statistics table in the limit-order-book paper), but it fails
on two very common cases:

  * Borderless tables (no vertical rules) -- columns collapse into one, because
    "lines" has no column separators to work with. This hid the results table
    in the Transformer paper (sample.pdf).

  * We can't just switch columns to "text", because on a page of prose,
    text-alignment invents spurious columns and turns paragraphs (and the
    bibliography) into fake tables.

So Phase 2 adds a second, more surgical method: CAPTION-GUIDED extraction. We
find where the paper literally says "Table N", crop the region just below that
caption, and extract only that crop with the text strategy. Because we only
look where the paper declares a table, we sidestep prose and references, and
the text strategy can recover borderless columns inside the crop.

Final pipeline = caption-guided crops  +  page-level ruled detection,
both passed through one content-aware filter, then de-duplicated.

KNOWN LIMITS (honest)
---------------------
  * Two-column journal layouts (e.g. CoinTossX): a table in the left column
    sits beside body prose in the right column at the same height, so a crop
    picks up both. Isolating one column needs layout detection -- out of scope.
  * A paper with genuinely no tables (e.g. a 2-page abstract) correctly yields
    zero -- that is not a bug.

Usage:
    python table_extract.py test_pdfs/<file>.pdf
"""

import re
import sys
# pyrefly: ignore [missing-import]
import pdfplumber


# Separator strategies in PRIORITY order. We do not mix blindly: we try ruled
# strategies first and only fall back to text-based detection when the lines
# don't yield a valid table. Rationale: ruling lines, when a table has them,
# are an unambiguous ground truth for where columns/rows are; text-alignment is
# an inference we only trust when there are no lines to go on.
#
#   1. lines / lines   -- fully ruled grid (rules between every row and column).
#   2. lines / text    -- vertical rules for columns, text alignment for rows
#                         (the common "booktabs with column rules" case; also
#                         what keeps prose from being chopped into columns).
#   3. text  / text    -- borderless table: no rules at all, infer everything
#                         from text alignment.
STRATEGY_PRIORITY = [
    {"vertical_strategy": "lines", "horizontal_strategy": "lines"},
    {"vertical_strategy": "lines", "horizontal_strategy": "text", "snap_y_tolerance": 5},
    {"vertical_strategy": "text", "horizontal_strategy": "text", "snap_y_tolerance": 4},
]

# How far below a "Table N" caption to crop, in PDF points. Big enough for a
# tall table, small enough to avoid running deep into following prose.
CAPTION_CROP_HEIGHT = 230


def looks_like_real_table(rows):
    """
    Reject figures, prose, and bibliographies that got detected as tables.

    Three content signals separate a real data table from a false positive:

      1. SHAPE. At least 2 columns, not absurdly wide (>20 cols = figure), and
         at least two "substantial" rows whose filled cells cover >= 40% of the
         width. (A figure carved into a wide grid has rows that are almost all
         empty; a real table's rows are mostly full.)

      2. DATA DENSITY. A good fraction of cells contain a digit. Data tables are
         numeric (scores, counts, percentages); prose paragraphs are not. Real
         table ~0.7-0.8; prose ~0.15. We require >= 0.30.

      3. NOT PROSE. Few cells are long free text, AND no single cell is huge.
         A data cell is short ("0.0117", "512", "MSFT"); a prose/bibliography
         cell is a long string. Real table cells top out around ~45 chars, so a
         single 179-char cell means prose leaked in (this is how a two-column
         layout contaminates a crop). We cap both the long-cell fraction and the
         single longest cell.
    """
    if not rows or len(rows) < 2:
        return False

    width = max(len(row) for row in rows)
    if width < 2 or width > 20:
        return False

    cells = [c.strip() for row in rows for c in row if c and c.strip()]
    if len(cells) < 4:
        return False

    numeric_frac = sum(1 for c in cells if re.search(r"\d", c)) / len(cells)
    long_frac = sum(1 for c in cells if len(c) > 30) / len(cells)
    longest_cell = max(len(c) for c in cells)

    # "Wordy" cells are long, space-free, all-letter tokens -- i.e. prose words
    # that got their spaces stripped and chopped into fake cells ("abulated-
    # measurements"). Real data tables have none of these; a two-column layout
    # that leaked a description paragraph into the crop is full of them.
    wordy_frac = sum(
        1 for c in cells if len(c) > 8 and " " not in c and c.isalpha()
    ) / len(cells)

    substantial_rows = 0
    for row in rows:
        filled = sum(1 for cell in row if cell and len(cell.strip()) > 1)
        if filled >= 2 and filled >= 0.4 * width:
            substantial_rows += 1

    return (
        substantial_rows >= 2
        and numeric_frac >= 0.30
        and long_frac <= 0.35
        and longest_cell <= 80
        and wordy_frac <= 0.15
    )


def merge_multiline_header(rows):
    """
    Recover a header that PDF layout split across two rows.

    A table like the Transformer results table has a header spanning two lines:
        row 0: ['Model', '',      '',      '',      ''     ]
        row 1: ['',      'EN-DE', 'EN-FR', 'EN-DE', 'EN-FR']
    Neither row alone names all the columns; merged column-by-column they do:
        ['Model', 'EN-DE', 'EN-FR', 'EN-DE', 'EN-FR'].

    We only merge when row 0 has empty cells AND row 1 is still header-like
    (has content but NO digits) -- that guards against merging a real data row,
    which would carry numbers.
    """
    if len(rows) < 3:
        return rows

    r0, r1 = rows[0], rows[1]
    r0_has_gaps = any(not (c and c.strip()) for c in r0)
    r1_has_content = any(c and c.strip() for c in r1)
    r1_has_digits = any(c and re.search(r"\d", c) for c in r1)

    if r0_has_gaps and r1_has_content and not r1_has_digits:
        merged = []
        for a, b in zip(r0, r1):
            parts = [x.strip() for x in (a, b) if x and x.strip()]
            merged.append(" ".join(parts))
        return [merged] + rows[2:]

    return rows


def merge_sparse_header(rows):
    """
    Merge a two-row header that survives after caption stripping.

    Some headers split across two rows where BOTH halves carry labels, e.g.
    Table 3's header is
        row 0: [ '', '', '',  '',   ...,  'train', 'PPL', 'BLEU', 'params']
        row 1: [ '', '', 'N', 'd_model d_ff', ..., 'steps', '(dev)', '(dev)', 'x10^6']
    merge_multiline_header can't fix this (its second row carries a digit, x10^6),
    so we handle it here with a numeric-fraction guard instead: merge row 0 into
    row 1 only when row 0 is sparse (mostly empty) AND row 1 is header-like
    (few of its cells are numeric). A real data row is mostly numbers, so this
    never swallows one.
    """
    if len(rows) < 3:
        return rows

    r0, r1 = rows[0], rows[1]
    r0_gaps = sum(1 for c in r0 if not (c and c.strip())) >= 0.4 * len(r0)
    r1_cells = [c.strip() for c in r1 if c and c.strip()]
    if not r1_cells:
        return rows
    r1_numeric = sum(1 for c in r1_cells if re.search(r"\d", c)) / len(r1_cells)

    if r0_gaps and r1_numeric < 0.4:
        merged = [
            " ".join(x.strip() for x in (a, b) if x and x.strip())
            for a, b in zip(r0, r1)
        ]
        return [merged] + rows[2:]

    return rows


def _is_data_row(row):
    """A body row is real data if it has a number and no prose word-fragments."""
    cells = [c.strip() for c in row if c and c.strip()]
    if not cells:
        return False
    has_number = any(re.search(r"\d", c) for c in cells)
    wordy = any(len(c) > 8 and " " not in c and c.isalpha() for c in cells)
    return has_number and not wordy


def drop_prose_rows(rows):
    """
    Keep the header row, then keep only genuine data rows after it.

    A long wrapped caption leaks into a caption crop as prose rows sitting above
    the real grid (Table 3's "Variations on the Transformer architecture...").
    Those rows carry word-fragments and no numbers, so _is_data_row drops them
    while the numeric data rows survive.
    """
    if len(rows) < 2:
        return rows
    body = [row for row in rows[1:] if _is_data_row(row)]
    return [rows[0]] + body


def strip_caption_rows(rows):
    """
    Drop leading caption-prose rows so the real header surfaces as row 0.

    A caption crop starts at the "Table N" caption, and a long caption wraps
    across one or more rows before the grid begins (Table 3's caption is three
    rows: "Table3: Variations on the Transformer architecture...", "...model.
    All metrics are on the English-to-German...", "...per-word perplexities.").
    Chopped finely enough, no single row looks obviously like prose, so instead
    of scoring the caption rows we ANCHOR on the data: find the first genuine
    data row (numbers, no word-fragments) and treat the row directly above it as
    the header. Everything above that header is caption and gets dropped.
    """
    first_data = next((i for i, row in enumerate(rows) if _is_data_row(row)), None)
    if first_data is None or first_data == 0:
        return rows  # no data row, or data starts at the top -- nothing to trim
    return rows[first_data - 1:]


# def _max_alpha_run(row):
#     """Longest run of consecutive letters in any cell of the row."""
#     longest = 0
#     for cell in row:
#         if not cell:
#             continue
#         for run in re.findall(r"[A-Za-z]+", cell):
#             longest = max(longest, len(run))
#     return longest


# def blank_caption_header(rows):
#     """
#     If the header row is actually wrapped caption prose, blank it.

#     In some two-column papers the caption is a long paragraph with no real
#     header row beneath it (CoinTossX Table 4), so the header ends up holding
#     caption fragments like "abulatedmeasurements". A real header cell is a short
#     label; caption prose has a long unbroken run of letters. When the header's
#     longest letter-run exceeds 18 we replace it with blanks, so row_to_sentence
#     falls back to generic "row_name"/"more info" labels instead of embedding
#     caption garbage as column names. Real headers (Model, d_model, MSFT) are
#     short-run and untouched.
#     """
#     if not rows:
#         return rows
#     if _max_alpha_run(rows[0]) > 18:
#         return [[""] * len(rows[0])] + rows[1:]
#     return rows


def _fold_cell_subscripts(cell):
    """
    Rejoin a subscript that pdfplumber split onto a second line inside a cell.

    Subscripts sit in a smaller font just below the baseline, so pdfplumber
    reads them as a second line within the cell:
        "d d\\nmodel ff"      (base "d d",  subscripts "model ff")
        "d P e\\nv drop ls"   (base "d P e", subscripts "v drop ls")
    The two lines line up token-for-token, so when they have equal token counts
    and the lower tokens are short (subscript-sized), we pair them with "_":
        -> "d_model d_ff"
        -> "d_v P_drop e_ls"
    Anything else (single line, or a wrapped multi-word cell) is just space-
    normalised, so ordinary cells are untouched.
    """
    if not cell or "\n" not in cell:
        return cell

    lines = [ln.strip() for ln in cell.split("\n") if ln.strip()]
    if len(lines) == 2:
        top, bottom = lines[0].split(), lines[1].split()
        if top and len(top) == len(bottom) and all(len(b) <= 6 for b in bottom):
            return " ".join(f"{t}_{b}" for t, b in zip(top, bottom))

    return " ".join(cell.split())


def fold_subscripts(rows):
    """Apply subscript folding to every cell of every row."""
    return [[_fold_cell_subscripts(cell) if cell else cell for cell in row] for row in rows]


def _clean_rows(rows):
    """Run the full row-repair pipeline on one extracted table."""
    rows = fold_subscripts(rows)
    rows = merge_multiline_header(rows)
    rows = strip_caption_rows(rows)
    rows = merge_sparse_header(rows)
    rows = drop_prose_rows(rows)
    # rows = blank_caption_header(rows)
    return rows


def find_table_captions(page):
    """Return (y_top, x0, label) for every 'Table N' caption word on the page."""
    captions = []
    for word in page.extract_words():
        if re.match(r"^Table\s*\d", word["text"]):
            captions.append((word["top"], word["x0"], word["text"]))
    return captions


def _caption_crop_boxes(page, y_top, x0):
    """
    Crop boxes to search around a caption at (y_top, x0), in priority order.

    Two independent axes, so a table is found whether it's above or below the
    caption and whether the page is one or two columns:

      * VERTICAL -- BELOW the caption first, then ABOVE. Academic *table*
        captions are conventionally placed above their table (so the table is
        below the caption), but that is a convention, not a guarantee -- some
        tables caption below. Trying below-then-above handles both without
        assuming a layout.

      * HORIZONTAL -- FULL WIDTH first, then the caption's OWN COLUMN. Full width
        is right for single-column papers and must go first so a single-column
        table is never sliced in half; the column crop (left half if the caption
        sits left of the page midline, else right half) only runs as a fallback,
        to rescue two-column journals where a table sits beside body prose.

    First candidate that yields a valid table wins, so the common layout
    (table below caption, full width) is tried first and the rarer ones are only
    reached when it fails.
    """
    mid = page.width / 2
    height = CAPTION_CROP_HEIGHT
    col_left, col_right = (0, mid) if x0 < mid else (mid, page.width)

    below = (y_top, min(y_top + height, page.height))
    above = (max(0, y_top - height), y_top)

    boxes = []
    for top, bottom in (below, above):
        boxes.append((0, top, page.width, bottom))              # full width
        boxes.append((col_left, top, col_right, bottom))        # caption's column
    return boxes


def extract_caption_tables(page, page_number):
    """
    Caption-guided extraction: for each 'Table N' caption, search the crop boxes
    around it (see _caption_crop_boxes) and extract with STRATEGY_PRIORITY (ruled
    first, text as fallback). Rows are cleaned before the validity check so
    caption prose doesn't sink an otherwise-real table. Yields
    (page_number, label, rows).
    """
    results = []
    for y_top, x0, label in find_table_captions(page):
        found = None
        for box in _caption_crop_boxes(page, y_top, x0):
            crop = page.crop(box)
            for settings in STRATEGY_PRIORITY:
                raw = crop.extract_table(settings)
                if not raw:
                    continue
                rows = _clean_rows(raw)
                if looks_like_real_table(rows):
                    found = rows
                    break
            if found:
                break

        if found:
            results.append((page_number, label, found))
    return results


def extract_page_tables(page, page_number):
    """
    Page-level detection for tables without a findable caption. We only use the
    ruled tiers here (lines/lines, then lines/text): text-based column detection
    on a whole page invents tables out of prose, and any genuinely borderless
    table is caught by the caption-guided path instead. First tier that finds
    real tables wins (lines get priority).
    """
    results = []
    for settings in STRATEGY_PRIORITY[:2]:
        found = []
        for raw in page.extract_tables(settings):
            rows = _clean_rows(raw)
            if looks_like_real_table(rows):
                found.append(rows)
        if found:
            results = [(page_number, None, rows) for rows in found]
            break
    return results


def _signature(rows):
    """A cheap fingerprint of a table's data, for de-duplication."""
    flat = [c.strip() for row in rows for c in row if c and c.strip()]
    return tuple(sorted(flat))[:12]


def extract_tables(pdf_path):
    """
    Combine caption-guided and page-level detection across the whole PDF,
    filter to real tables, and de-duplicate tables found by both methods.
    Returns a list of (page_number, label, rows).
    """
    extracted = []
    seen = set()
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            # Caption-guided first -- it's the more precise method.
            candidates = extract_caption_tables(page, page_number)
            candidates += extract_page_tables(page, page_number)
            for page_no, label, rows in candidates:
                sig = _signature(rows)
                if sig in seen:
                    continue
                seen.add(sig)
                extracted.append((page_no, label, rows))
    return extracted


def row_to_sentence(headers, row):
    """
    Convert one data row into a natural-language sentence with a generic
    template -- no hardcoded column names, since we don't know a paper's schema.

        headers = ["", "MSFT", "INTC"]
        row     = ["Mean spread", "0.0117", "0.0118"]
      ->  "In this row: row_name is Mean spread, MSFT is 0.0117, INTC is 0.0118."

    Unnamed columns still carry meaning, so we name them instead of dropping
    them: the first column is almost always the row label, so we call it
    "row_name"; any other unnamed column is "more info".
    """
    pairs = []
    for col_idx, (header, value) in enumerate(zip(headers, row)):
        if not value:
            continue
        clean_value = " ".join(str(value).split())
        if not clean_value:
            continue

        if header and str(header).strip():
            clean_header = " ".join(str(header).split())
        else:
            clean_header = "row_name" if col_idx == 0 else "more info"

        pairs.append(f"{clean_header} is {clean_value}")

    if not pairs:
        return None
    return "In this row: " + ", ".join(pairs) + "."


def print_table(page_number, label, rows):
    """Print one table: location, raw rows, and the sentences we'd embed."""
    tag = f" ({label})" if label else ""
    print("=" * 70)
    print(f"PAGE {page_number}{tag}  |  {len(rows)} rows")
    print("=" * 70)

    headers = rows[0]
    print(f"HEADER ROW (raw): {headers}\n")

    for i, row in enumerate(rows[1:], start=1):
        sentence = row_to_sentence(headers, row)
        if sentence:
            print(f"  data row {i} (raw): {row}")
            print(f"  -> sentence     : {sentence}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python table_extract.py <path-to-pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    print(f"Extracting tables from: {pdf_path}\n")

    tables = extract_tables(pdf_path)
    print(f"Kept {len(tables)} real table(s) after filtering.\n")

    for page_number, label, rows in tables:
        print_table(page_number, label, rows)


if __name__ == "__main__":
    main()
