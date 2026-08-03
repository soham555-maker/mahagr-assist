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


def _parse_date(raw):
    """Best-effort ISO date from a GR 'दिनांक' / 'Date' value. Handles
    '१५ जून, २०२३', '15 June 2023', and numeric dd/mm/yyyy or yyyy-mm-dd.
    Returns 'YYYY-MM-DD' or None."""
    s = _norm_digits(raw).strip()

    # day  month-word  year   (Marathi or English month name)
    m = re.search(r"(\d{1,2})\s+([^\s,]+)[,\s]+(\d{4})", s)
    if m:
        month = _MONTHS.get(m.group(2).lower()) or _MONTHS.get(m.group(2))
        if month:
            return f"{int(m.group(3)):04d}-{month:02d}-{int(m.group(1)):02d}"

    # dd/mm/yyyy or dd-mm-yyyy
    m = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", s)
    if m:
        return f"{int(m.group(3)):04d}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"

    # yyyy-mm-dd already
    m = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", s)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    return None


def _first(pattern, text, flags=0):
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else None


# label alternatives (Marathi | English), used in several patterns below
_L_NUMBER = r"(?:शासन निर्णय क्रमांक|शासन निर्णय क्र\.?|क्रमांक|Government Resolution No\.?|G\.?\s*R\.?\s*No\.?)"
_L_DATE = r"(?:दिनांक|Dated|Date)"
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


def _references(text, own=None):
    """GR numbers cited in the 'वाचा' / 'संदर्भ' reference section — the
    documents this GR builds on or replaces. A GR number is a slash-bearing
    token like 'संकीर्ण-२०२३/प्र.क्र.४५/तांशि-४'. Scoped to the reference
    section so body prose doesn't leak in; `own` (this GR's own number) is
    excluded so a document never lists itself. De-duplicated, order preserved."""
    # Terminators must be a section HEADER (word + colon). Without the colon,
    # the '.*?' would stop at the "शासन निर्णय क्रमांक ..." that OPENS the वाचा
    # value itself and capture nothing — the reference lives inside that phrase.
    m = re.search(rf"(?:वाचा|संदर्भ|Reference|Read)\s*[:：ःन.\-]?(.*?)(?=\n\s*(?:विषय|प्रस्तावना|शासन निर्णय|Subject|Preamble)\s*[:：ः]|\Z)",
                  text, re.DOTALL)
    scope = m.group(1) if m else ""
    seen, out = set(), []
    # A GR number token starts with a letter/Devanagari char and contains a
    # slash — e.g. 'एनजीसी-२०१०/(१९३/१०)'. Requiring the leading char rejects
    # OCR fragments like '/मशि-४' that a space split off the real number.
    # NOTE: reference sections are also full of DATES ('दि.०८/०३/२०१७'), which
    # are slash-bearing too — filter those out so references stay GR numbers.
    date_tok = re.compile(r"^(?:दि\.?|दिनांक)?\s*[\d०-९]{1,2}[/.\-][\d०-९]{1,2}[/.\-][\d०-९]{2,4}\.?$")
    for tok in re.findall(r"[A-Za-zऀ-ॿ][^\s,]*/[^\s,]+", scope):
        tok = tok.strip(" .,।")
        if not tok or tok in seen:
            continue
        if tok.startswith("दि") or date_tok.match(tok):   # a date, not a GR number
            continue
        if own is not None and (tok in own or own in tok):  # never list itself
            continue
        seen.add(tok)
        out.append(tok)
    return out


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
    if number:
        number = re.split(r"\s{2,}|।", number)[0].strip(" .,।-")
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

    refs = _references(text, own=meta.get("gr_number"))
    if refs:
        meta["references"] = refs

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
