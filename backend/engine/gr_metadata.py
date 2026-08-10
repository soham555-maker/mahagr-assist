"""
gr_metadata.py — pull document-level metadata out of a Government Resolution's
own text, so every chunk (and therefore every citation) can carry the GR
number, date, department, category and language (SRS FR 3.1.3 / 3.1.4).

This is a PURE function over already-extracted text — no model, no I/O — so it's
cheap, deterministic, and unit-testable. The ingest scripts call it once per
document and thread the result onto every chunk via `extra_metadata`.

WHY REGEX, NOT AN LLM
---------------------
Maharashtra GRs follow a fixed administrative header: a "शासन निर्णय क्रमांक"
line, a "दिनांक" line, a department line ending in "विभाग", a "विषय" (subject)
block, and often a "वाचा" (references) section. That structure is regular
enough to parse reliably and for free — spending an LLM call per document here
would be slower, costlier, and less deterministic. It is best-effort: any field
that doesn't match is simply omitted, never guessed.

BILINGUAL
---------
Every label has an English alternative (Marathi GRs dominate, but some
circulars are in English), and Devanagari digits/months are normalised so a
Marathi date like "१५ जून, २०२३" resolves to an ISO "2023-06-15".
"""

import datetime
import re

# Devanagari digits -> ASCII, so "२०२३" becomes "2023" for date/number parsing.
_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

_MONTHS = {
    # Marathi
    "जानेवारी": 1, "फेब्रुवारी": 2, "मार्च": 3, "एप्रिल": 4, "मे": 5, "जून": 6,
    "जुलै": 7, "ऑगस्ट": 8, "सप्टेंबर": 9, "ऑक्टोबर": 10, "नोव्हेंबर": 11, "डिसेंबर": 12,
    # English (lowercased, first three letters matched below too)
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

# Document category by header keyword, highest-priority first.
_CATEGORIES = [
    (("शासन निर्णय", "government resolution"), "government resolution (GR)"),
    (("शासन आदेश", "government order"), "government order (GO)"),
    (("परिपत्रक", "circular"), "circular"),
    (("अधिसूचना", "notification"), "notification"),
    (("कार्यालयीन आदेश", "office order"), "office order"),
]

_DEVA = re.compile(r"[ऀ-ॿ]")


def _norm_digits(s):
    return s.translate(_DEVA_DIGITS)


# Punctuation that OCR renders inconsistently inside a GR number. '/' is NOT in
# this set: it is structural (it separates the subject code, the proposal number
# and the desk code), so dropping it would collapse genuinely different numbers.
_NUMBER_NOISE = re.compile(r"[\s​‌‍.\-–—_,;:()\[\]]+")


def canonical_number(number):
    """Canonical form of a GR number, for MATCHING one document against another
    document's reference list (PLAN Phase 3).

    A GR number appears in two places that must be compared: `gr_number`, parsed
    from the document's own header, and the entries of another GR's `references`
    list. Both come out of OCR, so the SAME number routinely differs by
    whitespace, by Devanagari vs ASCII digits ('२०२३' vs '2023'), and by
    hyphens/dots the scanner did or didn't pick up. Comparing the raw strings
    finds almost nothing; comparing canonical forms finds the real edges.

    Returns None for an empty/meaningless input so callers can treat "no usable
    number" as a distinct case rather than matching on ''.

    >>> canonical_number("संकीर्ण-२०२३/प्र.क्र.४५/तांशि-४")
    'संकीर्ण2023/प्रक45/तांशि4'
    >>> canonical_number("संकीर्ण 2023 / प्र. क्र. ४५ / तांशि ४") == \
        canonical_number("संकीर्ण-२०२३/प्र.क्र.४५/तांशि-४")
    True
    """
    if not number:
        return None
    s = _norm_digits(str(number)).lower()
    s = _NUMBER_NOISE.sub("", s)
    s = re.sub(r"/+", "/", s).strip("/")
    return s or None


# A Government Resolution cannot predate the Bombay/Maharashtra record-keeping
# this corpus draws on, and cannot be issued far in the future. Anything outside
# this window is OCR noise, not a date. Measured on the 99,410-GR corpus: the
# real span is 1962-02-28 .. 2027-03-31, and only 9 documents fell outside it.
_MIN_GR_YEAR = 1900
_MAX_GR_YEAR = 2035


def _iso(year, month, day):
    """'YYYY-MM-DD' if this is a real, plausible GR date — otherwise None.

    TWO checks, because they catch different failures and neither alone is
    enough (both were found in the corpus):

      * CALENDAR validity. The day-month-name branch matches day as `\\d{1,2}`,
        so OCR noise produced **'2028-09-94'** — day 94. datetime rejects it.
      * PLAUSIBLE YEAR. '0201-01-21' is a perfectly valid date and a nonsense
        GR; a calendar check waves it straight through.

    Returning None matters as much as the check: ingest_corpus.py falls back to
    the order id (whose first 8 digits are YYYYMMDD and are far more reliable
    than an OCR'd header line), so a rejected parse becomes a CORRECT date
    rather than a missing one.
    """
    if not _MIN_GR_YEAR <= year <= _MAX_GR_YEAR:
        return None
    try:
        return datetime.date(year, month, day).isoformat()
    except ValueError:
        return None


def _parse_date(raw):
    """Best-effort ISO date from a GR 'दिनांक' / 'Date' value. Handles
    '१५ जून, २०२३', '15 June 2023', and numeric dd/mm/yyyy or yyyy-mm-dd.
    Returns 'YYYY-MM-DD' or None (see _iso for what is rejected and why)."""
    s = _norm_digits(raw).strip()

    # day  month-word  year   (Marathi or English month name)
    m = re.search(r"(\d{1,2})\s+([^\s,]+)[,\s]+(\d{4})", s)
    if m:
        month = _MONTHS.get(m.group(2).lower()) or _MONTHS.get(m.group(2))
        if month:
            iso = _iso(int(m.group(3)), month, int(m.group(1)))
            if iso:
                return iso

    # dd/mm/yyyy, dd-mm-yyyy and dd.mm.yyyy. The DOTTED form is the one GR
    # reference lines actually use ('दि. ३०.१०.२०१०'); omitting it left 2 of
    # every 3 cited dates unparsed, and the cited date is what tells two
    # same-numbered GRs apart when the graph resolves a reference.
    # The lookaround stops a GR number ('१४०२५/११/२०२३') being read as a date:
    # a date component may not sit next to another digit or separator.
    m = re.search(r"(?<![\d/.\-])(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})(?![\d])", s)
    if m:
        year = int(m.group(3))
        if year < 100:                       # '02.09.13' -> 2013
            year += 2000 if year < 70 else 1900
        iso = _iso(year, int(m.group(2)), int(m.group(1)))
        if iso:
            return iso

    # yyyy-mm-dd already
    m = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", s)
    if m:
        return _iso(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


def _first(pattern, text, flags=0):
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else None


# label alternatives (Marathi | English), used in several patterns below
_L_NUMBER = r"(?:शासन निर्णय क्रमांक|शासन निर्णय क्र\.?|क्रमांक|Government Resolution No\.?|G\.?\s*R\.?\s*No\.?)"
_L_DATE = r"(?:दिनांक|तारीख|Dated|Date)"
_L_SUBJECT = r"(?:विषय|Subject)"
_L_SECTION_END = r"(?:वाचा|संदर्भ|प्रस्तावना|शासन निर्णय|Reference|Preamble|Read)"


def _language(text):
    """'mr' if the text is Devanagari-dominant, else 'en'. (Marathi vs Hindi is
    not distinguished — both are Devanagari; Marathi is assumed for GRs.)"""
    deva = len(_DEVA.findall(text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if deva == 0 and latin == 0:
        return None
    return "mr" if deva >= latin else "en"


# --------------------------------------------------------------------------- #
# reference extraction
#
# A GR number CONTAINS SPACES ('एनजीसी-२०१०/(१९३/१०) /मशि-४'), so it cannot be
# matched as a whitespace-bounded token — that is what the first version did,
# and it produced fragments ('२०११/प्रक्र', '१३६/विशि-३') that resolved against
# nothing: 2% of graph edges. What actually delimits a number in a reference
# line is punctuation and the date that follows it:
#
#   वाचा : १) शासन निर्णय, उच्च व तंत्र शिक्षण विभाग, क्र. एनजीसी-२०१०/(१९३/१०) /मशि-४, दि. ३०.१०.२०१०.
#          └item┘└─ document type ─┘└──── department ────┘└label┘└──── the number ────┘  └── the date ──┘
#
# So the block is cut into numbered ITEMS, each item into comma-separated
# SEGMENTS, and each segment is trimmed at its date. That mirrors what extract()
# already does for the document's own number.
# --------------------------------------------------------------------------- #

# The date that trails a number: 'दि. ३०.१०.२०१०', 'दिनांक - १८/०७/२०११', 'Dated 3 May 2024'.
# The lookbehind keeps 'दि' from firing inside a Devanagari word.
_DATE_TAIL = re.compile(r"(?<![ऀ-ॿ])(?:दिनांक|दिनाक|दि|Dated|Dt|Date)\s*[.:：ः\-–]*\s*(?=[\d०-९])")
# A date at the START of a piece, left over after the split above.
_LEADING_DATE = re.compile(r"^\s*[\d०-९]{1,4}[/.\-][\d०-९]{1,2}[/.\-][\d०-९]{2,4}\.?\s*")
# Item numbering: '१)', '(२)', '3.' — at the start of a line or inline.
_ITEM_LEAD = re.compile(r"^\s*[(\[]?\s*[\d०-९]{1,2}\s*[)\].]\s*")
_ITEM_SPLIT = re.compile(r"\n|(?<=[।.\s])[(\[]?[\d०-९]{1,2}[)\]](?=\s)")
_SEG_SPLIT = re.compile(r"[,;।]")
# The label a number follows: 'शासन निर्णय, <विभाग>, क्रमांक : <number>'.
_NUM_LABEL = re.compile(r"(?:क्रमांक|क्रमाक|क्र|No|Number)\s*[.:：ः\-–]*\s*")
# A date is slash-bearing too, so it must be rejected explicitly.
_BARE_DATE = re.compile(r"^[\d०-९]{1,4}[/.\-][\d०-९]{1,2}[/.\-][\d०-९]{2,4}\.?$")


def _trim_number(s):
    """Cut a captured GR number at the date that follows it and strip the
    punctuation OCR leaves behind. Shared by extract() and the reference scan so
    both sides of a match are normalised the same way."""
    if not s:
        return None
    s = _DATE_TAIL.split(s)[0]
    s = re.split(r"\s{2,}|।", s)[0]
    return " ".join(s.split()).strip(" .,;:।-–") or None


def _after_label(piece):
    """Drop the 'शासन निर्णय ... क्रमांक :' preamble in front of a number.

    The trap: 'क्र' also occurs INSIDE most GR numbers ('संकीर्ण-२०२३/प्र.क्र.४५'),
    so stripping at the first label match truncates the number to '४५'. The
    discriminator is the slash — a number's internal 'क्र' always sits after
    one, an introducing label never does. So: take the LAST label whose preamble
    contains no '/'. Last, not first, because a reference line can carry two
    ('शासन निर्णय क्र. <विभाग> क्र. <number>').
    """
    cut = None
    for m in _NUM_LABEL.finditer(piece):
        if "/" in piece[:m.start()]:
            break
        cut = m.end()
    return piece[cut:] if cut is not None else None


def _is_number(s):
    """Does this look like a GR/letter number rather than prose or a date?
    Deliberately shape-based: a number has a slash (structural in every GR
    number), at least one digit, and is short."""
    if not s or "/" not in s or len(s) > 80:
        return False
    if not re.search(r"[\d०-९]", s):
        return False
    return not _BARE_DATE.match(s)


def reference_block(text):
    """The raw text of the 'वाचा' / 'संदर्भ' (references) section, or "".

    Public because more than one thing needs the same scope: `_reference_entries`
    pulls GR numbers out of it, and `scripts/cited_departments.py` counts which
    DEPARTMENTS are named in it (which is how the ingestion order for PLAN
    Phase 6 was chosen — see CHECKLIST). Keeping one definition of "where the
    references are" means those two measurements can never drift apart.

    Terminators must be a section HEADER (word + colon). Without the colon, the
    '.*?' would stop at the "शासन निर्णय क्रमांक ..." that OPENS the वाचा value
    itself and capture nothing — the reference lives inside that phrase.
    """
    m = re.search(r"(?:वाचा|संदर्भ|Reference|Read)\s*[:：ःन.\-]?(.*?)"
                  r"(?=\n\s*(?:विषय|प्रस्तावना|शासन निर्णय|Subject|Preamble)\s*[:：ः]|\Z)",
                  text, re.DOTALL)
    return m.group(1) if m else ""


def _reference_entries(text, own=None):
    """References in the 'वाचा' / 'संदर्भ' block, as {'number', 'date'} dicts.

    The date matters: 2,138 canonical numbers in the corpus are shared by more
    than one document (a GR and its corrigendum, or an OCR collision), and the
    cited date is what tells them apart when the graph resolves the reference.

    Scoped to the reference section so body prose doesn't leak in; `own` (this
    GR's own number) is dropped so a document never lists itself. De-duplicated
    on the canonical form, order preserved.
    """
    scope = reference_block(text)
    own_canon = canonical_number(own)
    seen, out = set(), []

    for item in _ITEM_SPLIT.split(scope):
        item = _ITEM_LEAD.sub("", item)
        # One date per item — the date of the order being cited.
        date = None
        for tail in _DATE_TAIL.split(item)[1:]:
            date = _parse_date(tail[:40])
            if date:
                break
        for seg in _SEG_SPLIT.split(item):
            for piece in _DATE_TAIL.split(seg):
                piece = _LEADING_DATE.sub("", piece)
                # Prefer the label-stripped form, but fall back to the raw piece:
                # plenty of references print the number with no label at all.
                cand = _trim_number(_after_label(piece))
                if not _is_number(cand):
                    cand = _trim_number(piece)
                if not _is_number(cand):
                    continue
                canon = canonical_number(cand)
                if not canon or canon in seen or canon == own_canon:
                    continue
                seen.add(canon)
                out.append({"number": cand, "date": date})
    return out


def _references(text, own=None):
    """The cited GR numbers only — the historical shape of this field."""
    return [e["number"] for e in _reference_entries(text, own=own)]


def extract(text):
    """Parse a GR's header text into a metadata dict. Only fields that actually
    match are included, so a document that omits (say) a department line simply
    has no 'department' key rather than an empty one.

    Keys (all optional): title, gr_number, date (ISO), date_raw, department,
    category, language ('mr'/'en'), references (list), supersedes (bool).
    """
    meta = {}

    # GR number: 'शासन निर्णय' immediately followed by क्रमांक/क्र — this anchors
    # on the MAIN number line and skips reference lines (where a comma follows
    # 'शासन निर्णय'). Real numbers contain spaces ('एनजीसी २०१७/(२२९/१७)/मशि-४'),
    # so capture to end of line, not just the first token.
    number = _first(r"(?:शासन निर्णय|शासन आदेश|आदेश|परिपत्रक|अधिसूचना)\s*(?:क्रमांक|क्र)\s*[.:：ः]*\s*([^\n]+)", text) \
        or _first(r"(?:Government Resolution No\.?|Government Order No\.?|G\.?\s*R\.?\s*No\.?)\s*[:：]?\s*([^\n]+)", text)
    number = _trim_number(number)
    if number:
        meta["gr_number"] = number[:80]

    date_raw = _first(rf"{_L_DATE}\s*[:：-]?\s*([^\n]+)", text)
    if date_raw:
        date_raw = date_raw.strip(" .,।")
        meta["date_raw"] = date_raw
        iso = _parse_date(date_raw)
        if iso:
            meta["date"] = iso

    # department: a line ending in 'विभाग' (or containing 'Department')
    dept = _first(r"(?m)^\s*(.{0,60}?विभाग)\s*$", text) \
        or _first(r"(?m)^\s*(.{0,60}?Department)\s*$", text)
    if dept:
        meta["department"] = " ".join(dept.split())

    # title: prefer an explicit 'विषय:' header — but ONLY as a header (colon
    # required), so the common noun 'विषय' inside a sentence
    # ("अभ्यासक्रम, विषय व वाढीव") doesn't get mistaken for one. Many real GRs
    # have no विषय label and put the subject as the opening line(s); fall back
    # to the first substantive line for those.
    subject = _first(rf"{_L_SUBJECT}\s*[:：ः]\s*(.+?)(?=\n\s*{_L_SECTION_END}\s*[:：ः]|\Z)",
                     text, re.DOTALL)
    if not subject:
        for line in text.splitlines():
            s = line.strip()
            if s and not s.startswith("#") and "महाराष्ट्र शासन" not in s and len(s) > 25:
                subject = s
                break
    if subject:
        meta["title"] = " ".join(subject.split())[:300]

    low = text.lower()
    for keys, label in _CATEGORIES:
        if any(k in text or k in low for k in keys):
            meta["category"] = label
            break

    lang = _language(text)
    if lang:
        meta["language"] = lang

    # Both shapes are returned: `references` (numbers only) is what officer.py,
    # the API and the older corpora already speak; `reference_details` adds the
    # cited date, which the knowledge graph uses to disambiguate two documents
    # that share a GR number.
    entries = _reference_entries(text, own=meta.get("gr_number"))
    if entries:
        meta["reference_details"] = entries
        refs = [e["number"] for e in entries]
        meta["references"] = refs
    else:
        refs = []

    # A GR that both cites another and says 'अधिक्रमित'/'superseded' is replacing it.
    if refs and re.search(r"अधिक्रमित|अधिक्रमीत|supersed", text, re.IGNORECASE):
        meta["supersedes"] = True

    return meta


if __name__ == "__main__":
    import sys
    # Quick manual check: python engine/gr_metadata.py < some_extracted_text.txt
    data = sys.stdin.read() if not sys.stdin.isatty() else ""
    from pprint import pprint
    pprint(extract(data))
