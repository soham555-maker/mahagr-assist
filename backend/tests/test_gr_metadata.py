"""GR header parsing — the fields that power citations and the supersede graph."""

from engine import gr_metadata

# A realistic Marathi GR header (multi-word GR number, Marathi date, संदर्भ refs).
GR_MR = """# Page 1
शासकीय तंत्रनिकेतनांमधील प्रथम वर्ष पदविका अभ्यासक्रमांच्या शुल्काबाबत.
महाराष्ट्र शासन
उच्च व तंत्र शिक्षण विभाग
शासन निर्णय क्रमांक: संकीर्ण-२०२४/प्र.क्र.१२/तांशि-४
दिनांक: १० एप्रिल, २०२४
संदर्भ: शासन निर्णय क्रमांक संकीर्ण-२०२३/प्र.क्र.४५/तांशि-४, दिनांक १५ जून, २०२३.
प्रस्तावना: सदर शासन निर्णय अधिक्रमित करण्यात येत आहे.
शासन निर्णय: खालीलप्रमाणे सुधारित शुल्क लागू राहील.
"""


def test_gr_number_multiword():
    # must capture the whole number incl. spaces, not stop at the first token
    assert gr_metadata.extract(GR_MR)["gr_number"] == "संकीर्ण-२०२४/प्र.क्र.१२/तांशि-४"


def test_marathi_date_to_iso():
    assert gr_metadata.extract(GR_MR)["date"] == "2024-04-10"


def test_department_and_language_and_category():
    m = gr_metadata.extract(GR_MR)
    assert m["department"] == "उच्च व तंत्र शिक्षण विभाग"
    assert m["language"] == "mr"
    assert m["category"] == "government resolution (GR)"


def test_references_and_supersedes():
    m = gr_metadata.extract(GR_MR)
    assert m.get("supersedes") is True
    assert any("२०२३" in r for r in m["references"])
    # its own number must never appear in references
    assert all("२०२४" not in r for r in m["references"])
    # a date must not be captured as a reference
    assert all("१५" not in r for r in m["references"])


def test_title_falls_back_to_first_line_when_no_subject_header():
    # GR_MR has no 'विषय:' header -> title is the opening subject line.
    assert gr_metadata.extract(GR_MR)["title"].startswith("शासकीय तंत्रनिकेतनांमधील")


def test_subject_word_midsentence_is_not_mistaken_for_a_header():
    txt = "# Page 1\nनवीन अभ्यासक्रम, विषय व तुकड्या मंजुरीबाबत निर्णय.\nमहाराष्ट्र शासन\n"
    title = gr_metadata.extract(txt).get("title", "")
    # must NOT slice from the mid-sentence 'विषय'; should be the whole first line
    assert title.startswith("नवीन अभ्यासक्रम")


def test_english_date_formats():
    assert gr_metadata.extract("Dated: 15 June 2023")["date"] == "2023-06-15"
    assert gr_metadata.extract("Date: 10/04/2024")["date"] == "2024-04-10"


def test_language_english():
    assert gr_metadata.extract("Government Resolution No. ABC/2024 Dated: 1 May 2024")["language"] == "en"


# --------------------------------------------------------------------------
# Reference extraction.
#
# These are VERBATIM reference lines from the 18,078-GR corpus. The first
# implementation matched a slash-bearing token bounded by WHITESPACE, but a real
# GR number CONTAINS spaces ('एनजीसी-२०१०/(१९३/१०) /मशि-४'), so every one of
# these came out truncated — '२०११/प्रक्र', '१३६/विशि-३', 'अनौस-२०२०/प्र.क्र.१०२/'
# — and the knowledge graph resolved only 2% of its edges as a result.
# --------------------------------------------------------------------------

REF_SPACED = """# Page 1
महाराष्ट्र शासन
उच्च व तंत्र शिक्षण विभाग
शासन निर्णय क्र. एनजीसी २०१७/(२२९/१७)/मशि-४
दिनांक : १२ ऑक्टोबर, २०१७
संदर्भ : १) शासन निर्णय, उच्च व तंत्र शिक्षण विभाग, क्र. एनजीसी-२०१०/(१९३/१०) /मशि-४, दि. ३०.१०.२०१०.
२) शासन निर्णय, उच्च व तंत्र शिक्षण विभाग, क्र. एनजीसी-२०१२/(२४७/१२) /मशि-४, दि. ०२.०९.२०१३.
शासन निर्णय : खालीलप्रमाणे निर्णय घेतला आहे.
"""

REF_TWO_LABELS = """# Page 1
शासन निर्णय क्रमांक:- संकीर्ण २०१५/प्र.क्र.२१९/१५/विशि-३
दिनांक - १२ ऑक्टोबर, २०१७.
वाचा -१) शासन निर्णय क्र. उच्च व तंत्र शिक्षण विभाग, क्र. संकीर्ण २०११/प्र.क्र. १३६/विशि-३, दिनांक - १८/०७/२०११.
प्रस्तावना : सदर समिती पुनर्गठीत करण्यात येत आहे.
"""


def test_reference_with_internal_spaces_is_captured_whole():
    refs = gr_metadata.extract(REF_SPACED)["references"]
    assert refs == ["एनजीसी-२०१०/(१९३/१०) /मशि-४", "एनजीसी-२०१२/(२४७/१२) /मशि-४"]


def test_reference_does_not_swallow_the_date_that_follows_it():
    for r in gr_metadata.extract(REF_SPACED)["references"]:
        assert "दि" not in r.split("/")[0][:3]   # no 'दि. ३०.१०.२०१०' tail
        assert "२०१०." not in r


def test_reference_skips_the_department_between_two_labels():
    # 'शासन निर्णय क्र. उच्च व तंत्र शिक्षण विभाग, क्र. संकीर्ण २०११/...' — the
    # FIRST 'क्र.' is followed by the department, not by a number.
    refs = gr_metadata.extract(REF_TWO_LABELS)["references"]
    assert refs == ["संकीर्ण २०११/प्र.क्र. १३६/विशि-३"]


def test_reference_is_not_truncated_at_the_inner_label():
    # regression: 'प्र.क्र.' inside a number must not be mistaken for the label
    # that introduces one, which produced fragments like '१३६/विशि-३'.
    refs = gr_metadata.extract(REF_TWO_LABELS)["references"]
    assert not any(r.startswith("१३६") for r in refs)


def test_label_after_a_long_preamble_still_introduces_the_number():
    # no commas to lean on: the number is the LAST label on the line.
    txt = ("शासन निर्णय क्रमांक: अ-१/२०२४\n"
           "वाचा : शासन निर्णय उच्च व तंत्र शिक्षण विभाग क्र. संकीर्ण २०११/प्र.क्र. १३६/विशि-३ दि. १८/०७/२०११.\n")
    assert gr_metadata.extract(txt)["references"] == ["संकीर्ण २०११/प्र.क्र. १३६/विशि-३"]


def test_reference_details_carry_the_cited_date():
    details = gr_metadata.extract(REF_SPACED)["reference_details"]
    assert [d["number"] for d in details] == gr_metadata.extract(REF_SPACED)["references"]
    assert details[0]["date"] == "2010-10-30"      # 'दि. ३०.१०.२०१०'
    assert details[1]["date"] == "2013-09-02"


def test_dotted_numeric_dates_parse():
    # GR reference lines write dates as ३०.१०.२०१० far more often than 30/10/2010
    assert gr_metadata.extract("दिनांक: ३०.१०.२०१०")["date"] == "2010-10-30"
    assert gr_metadata.extract("Dated: 02.09.13")["date"] == "2013-09-02"


def test_gr_number_stops_before_a_trailing_date():
    # real corpus row: 'शासन निर्णय क्रमांक: ११११/(२०/१५)/एलबी/तांशि-२, दि. २३ मे, २०१८'
    txt = "शासन निर्णय क्रमांक: ११११/(२०/१५)/एलबी/तांशि-२, दि. २३ मे, २०१८\n"
    assert gr_metadata.extract(txt)["gr_number"] == "११११/(२०/१५)/एलबी/तांशि-२"


def test_a_bare_date_is_never_a_reference():
    txt = "शासन निर्णय क्रमांक: अ-१/२०२४\nवाचा : शासन परिपत्रक क्र. अनौस-२०२०/प्र.क्र.१०२/तांशि-४, दि.०८/०३/२०१७.\n"
    assert gr_metadata.extract(txt)["references"] == ["अनौस-२०२०/प्र.क्र.१०२/तांशि-४"]


def test_prose_with_a_slash_is_not_a_reference():
    txt = ("शासन निर्णय क्रमांक: अ-१/२०२४\n"
           "वाचा : शासन निर्णय क्र. संकीर्ण-२०२३/प्र.क्र.४५/तांशि-४, दि.०८/०३/२०१७.\n"
           "डोंगराळ / दुर्गम भागासाठी सर्वोच्च प्राथम्य द्यावे व त्या प्राथम्यक्रमानुसार कार्यवाही करावी.\n")
    assert gr_metadata.extract(txt)["references"] == ["संकीर्ण-२०२३/प्र.क्र.४५/तांशि-४"]


# --------------------------------------------------------------------------- #
# reference_block — public since PLAN Phase 6, so it needs its own guard
# --------------------------------------------------------------------------- #

def test_reference_block_returns_only_the_vacha_section():
    """Two consumers now share this scope — the edge builder (which reads GR
    NUMBERS out of it) and scripts/cited_departments.py (which reads DEPARTMENT
    names out of it). If the scope silently widened to the whole document, the
    department ranking would start counting the ISSUING department of every GR
    and the ingestion order for Phase 6 would be derived from noise.
    """
    txt = ("महाराष्ट्र शासन\n"
           "शासन निर्णय क्रमांक: अ-१/२०२४\n"
           "वाचा : १) शासन निर्णय, वित्त विभाग, क्र. संकीर्ण-२०२३/प्र.क्र.४५/तांशि-४, दि. ०८.०३.२०१७.\n"
           "विषय : नियोजन विभाग यांचेकडील प्रस्तावाबाबत\n"
           "प्रस्तावना : सामान्य प्रशासन विभाग यांनी कळविले आहे.\n")
    block = gr_metadata.reference_block(txt)
    # the cited department is inside the block ...
    assert "वित्त विभाग" in block
    # ... and departments named in the SUBJECT and PREAMBLE are not
    assert "नियोजन विभाग" not in block
    assert "सामान्य प्रशासन" not in block


def test_reference_block_is_empty_when_there_is_no_vacha():
    assert gr_metadata.reference_block("शासन निर्णय क्रमांक: अ-१/२०२४\nविषय : काही तरी\n") == ""


def test_reference_block_keeps_the_number_that_opens_the_vacha_value():
    """The terminator must be a section HEADER (word + colon). Without the
    colon the lazy '.*?' stops at the 'शासन निर्णय' that OPENS the वाचा value
    itself and captures nothing — the reference lives inside that phrase.
    """
    txt = "वाचा : शासन निर्णय क्र. संकीर्ण-२०२३/प्र.क्र.४५/तांशि-४, दि. ०८.०३.२०१७.\n"
    assert "संकीर्ण-२०२३" in gr_metadata.reference_block(txt)


# --------------------------------------------------------------------------- #
# Date validation. Found on the 99,410-GR corpus, where /corpus/stats reported a
# span of "0201-01-21 .. 9202-01-19" and fed that range to the portal's date
# filter. Only 9 documents were affected, but the SPAN is user-visible.
# --------------------------------------------------------------------------- #

def test_impossible_calendar_dates_are_rejected():
    """Day is matched as \\d{1,2}, so OCR noise produced a real corpus row dated
    '2028-09-94'. Formatting without validating is what let it through."""
    assert gr_metadata._parse_date("दिनांक: 94 सप्टेंबर, 2028") is None
    assert gr_metadata._parse_date("Dated: 30/02/2023") is None      # Feb 30
    assert gr_metadata._parse_date("Dated: 15/13/2023") is None      # month 13


def test_implausible_years_are_rejected_even_when_calendar_valid():
    """'0201-01-21' is a VALID date and a nonsense GR — a calendar check alone
    waves it through. That row's order id began 20190121, so rejecting the parse
    lets ingest fall back to the correct 2019-01-21."""
    assert gr_metadata._parse_date("दिनांक: 21/01/0201") is None
    assert gr_metadata._parse_date("Dated: 19 January 9202") is None


def test_real_dates_still_parse():
    """The validation must not cost any true positive."""
    assert gr_metadata._parse_date("दिनांक: १५ जून, २०२३") == "2023-06-15"
    assert gr_metadata._parse_date("Dated: 15 June 2023") == "2023-06-15"
    assert gr_metadata._parse_date("दि. ३०.१०.२०१०") == "2010-10-30"
    assert gr_metadata._parse_date("Dated: 02.09.13") == "2013-09-02"
    assert gr_metadata._parse_date("2023-06-15") == "2023-06-15"
    # boundaries of the plausible window
    assert gr_metadata._parse_date("Dated: 01/01/1900") == "1900-01-01"
    assert gr_metadata._parse_date("Dated: 31/12/2035") == "2035-12-31"
