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
