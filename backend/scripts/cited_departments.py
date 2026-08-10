"""
cited_departments.py — which departments does this corpus CITE but not HOLD?

WHY THIS EXISTS
---------------
The knowledge graph resolves only ~9.5% of its references, and the standing
explanation is "we hold 6 of the dataset's 33 departments, so most references
point at orders we simply don't have". That is a plausible story. This script
turns it into a measurement, and — more usefully — says WHICH departments to
ingest next for the biggest gain, instead of ingesting 27 of them in an
arbitrary order and hoping.

HOW IT MEASURES
---------------
A GR's 'वाचा' (references) block names the department of each order it cites,
usually right before the number:

    वाचा : १) शासन निर्णय, उच्च व तंत्र शिक्षण विभाग, क्र. एनजीसी-२०१०/... , दि. ३०.१०.२०१०.
                         └──── the cited department ────┘

So: take the reference block only (gr_metadata.reference_block — the SAME scope
the edge builder uses, so the two numbers are about the same text), and count
which department names appear in it. Restricting to that block is the whole
point — scanning the full document would mostly count the ISSUING department,
which is not what we want to know.

WHAT THE NUMBERS MEAN, AND WHAT THEY DON'T
------------------------------------------
A department name in a reference block is EVIDENCE of a citation, not proof of
one: a block may name a department without a parsable number, and a reference
may print a number with no department at all (those are counted separately as
"unattributed"). So read the output as a RANKING to drive ingestion order, not
as a count of edges that will resolve. The honest check on the ranking is to
ingest the top departments and re-run scripts/build_graph.py — which is exactly
what PLAN Phase 6 does.

Usage:
    python scripts/cited_departments.py                 # rank all 33
    python scripts/cited_departments.py --top 10
"""

import argparse
import os
import sys

from engine import corpus_db, gr_metadata

DEFAULT_INDEX = os.environ.get("MAHAGR_INDEX_DIR", "/mnt/win/mahagr/index")

# The 33 departments in orgpedia/mahGRs, keyed by the folder name the corpus
# uses, with the Marathi fragments a reference line actually prints.
#
# Fragments, NOT full official names, on purpose: the printed form varies by
# decade and by OCR quality ("महसूल व वन विभाग" / "महसुल व वन विभाग" / "महसूल
# आणि वन विभाग"), so matching a canonical full name would under-count badly.
# Each fragment is chosen to be long enough to be unambiguous between the 33.
DEPARTMENTS = {
    "Agriculture,_Dairy_Development,_Animal_Husbandry_and_Fisheries_Department":
        ["कृषि, पशुसंवर्धन", "कृषी, पशुसंवर्धन", "पशुसंवर्धन", "मत्स्यव्यवसाय"],
    "Co-operation,_Textiles_and_Marketing_Department":
        ["सहकार, पणन", "पणन व वस्त्रोद्योग", "वस्त्रोद्योग"],
    "Environment_Department": ["पर्यावरण विभाग", "पर्यावरण व वातावरणीय"],
    "Finance_Department": ["वित्त विभाग", "वित्त विभागा"],
    "Food,_Civil_Supplies_and_Consumer_Protection_Department":
        ["अन्न, नागरी पुरवठा", "नागरी पुरवठा व ग्राहक"],
    "General_Administration_Department": ["सामान्य प्रशासन"],
    "Higher_and_Technical_Education_Department": ["उच्च व तंत्र शिक्षण", "उच्च आणि तंत्र शिक्षण"],
    "Home_Department": ["गृह विभाग", "गृह विभागा"],
    "Housing_Department": ["गृहनिर्माण विभाग", "गृहनिर्माण विभागा"],
    "Industries,_Energy_and_Labour_Department": ["उद्योग, ऊर्जा", "ऊर्जा व कामगार", "कामगार विभाग"],
    "Information_Technology_Department": ["माहिती तंत्रज्ञान"],
    "Law_and_Judiciary_Department": ["विधि व न्याय", "विधी व न्याय"],
    "Marathi_Language_Department": ["मराठी भाषा विभाग"],
    "Medical_Education_and_Drugs_Department": ["वैद्यकीय शिक्षण", "औषधी द्रव्ये"],
    "Minorities_Development_Department": ["अल्पसंख्याक विकास"],
    "Other_Backward_Bahujan_Welfare_Department": ["इतर मागास बहुजन", "बहुजन कल्याण"],
    "Parliamentary_Affairs_Department": ["संसदीय कार्य"],
    "Persons_with_Disabilities_Welfare_Department": ["दिव्यांग कल्याण"],
    "Planning_Department": ["नियोजन विभाग", "नियोजन विभागा"],
    "Public_Health_Department": ["सार्वजनिक आरोग्य"],
    "Public_Works_Department": ["सार्वजनिक बांधकाम"],
    "Revenue_and_Forest_Department": ["महसूल व वन", "महसुल व वन", "महसूल आणि वन"],
    "Rural_Development_Department": ["ग्राम विकास", "ग्रामविकास"],
    "School_Education_and_Sports_Department": ["शालेय शिक्षण"],
    "Skill_Development_and_Entrepreneurship_Department": ["कौशल्य विकास"],
    "Social_Justice_and_Special_Assistance_Department": ["सामाजिक न्याय"],
    "Soil_and_Water_Conservation_Department": ["मृद व जलसंधारण", "मृद आणि जलसंधारण", "जलसंधारण विभाग"],
    "Tourism_and_Cultural_Affairs_Department": ["पर्यटन व सांस्कृतिक", "सांस्कृतिक कार्य"],
    "Tribal_Development_Department": ["आदिवासी विकास"],
    "Urban_Development_Department": ["नगर विकास", "नगरविकास"],
    "Water_Resources_Department": ["जलसंपदा विभाग", "जलसंपदा विभागा"],
    "Water_Supply_and_Sanitation_Department": ["पाणीपुरवठा व स्वच्छता", "पाणी पुरवठा व स्वच्छता"],
    "Women_and_Child_Development_Department": ["महिला व बाल", "महिला आणि बाल"],
}


def _held(conn):
    """Folder names of the departments already in the corpus.

    gr_documents stores the DISPLAY form ("Revenue and Forest Department"), the
    dataset uses the folder form ("Revenue_and_Forest_Department"), and
    ingest_corpus._department_label maps one to the other by replacing '_' with
    ' '. Inverting that is exact, so no fuzzy matching is needed.
    """
    rows = conn.execute("SELECT DISTINCT department FROM gr_documents").fetchall()
    return {r[0].replace(" ", "_") for r in rows if r[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=DEFAULT_INDEX)
    ap.add_argument("--top", type=int, default=0, help="show only the top N (0 = all)")
    args = ap.parse_args()

    db_path = os.path.join(args.index, "corpus.db")
    if not os.path.exists(db_path):
        print(f"No corpus at {db_path}. Run scripts/ingest_corpus.py first.")
        return 1

    counts = {d: 0 for d in DEPARTMENTS}
    docs = with_block = unattributed = 0

    with corpus_db.connect(db_path, readonly=True) as conn:
        held = _held(conn)
        for (text,) in conn.execute("SELECT text FROM gr_documents WHERE text IS NOT NULL"):
            docs += 1
            block = gr_metadata.reference_block(text)
            if not block:
                continue
            with_block += 1
            hit = False
            for dept, fragments in DEPARTMENTS.items():
                if any(f in block for f in fragments):
                    counts[dept] += 1
                    hit = True
            if not hit:
                unattributed += 1

    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    if args.top:
        ranked = ranked[:args.top]

    print(f"{docs} documents, {with_block} with a reference block "
          f"({unattributed} of those name no department at all)\n")
    print(f"{'department':62s} {'docs citing it':>14s}   held?")
    print("-" * 90)
    for dept, n in ranked:
        if not n:
            continue
        pct = 100.0 * n / max(with_block, 1)
        print(f"  {dept:60s} {n:8d} ({pct:4.1f}%)   "
              f"{'HELD' if dept in held else '-- MISSING --'}")

    missing = [(d, n) for d, n in sorted(counts.items(), key=lambda kv: -kv[1])
               if d not in held and n]
    print("\n" + "=" * 90)
    print("Most-cited departments NOT in the corpus — ingest these first "
          "(PLAN Phase 6 wave A):")
    for dept, n in missing[:8]:
        print(f"  {dept:60s} {n:8d}")
    print("\nfetch:  python scripts/fetch_mahgrs.py " +
          " ".join(f"--dept {d}" for d, _ in missing[:5]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
