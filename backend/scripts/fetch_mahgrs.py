"""
fetch_mahgrs.py — download a real GR corpus from orgpedia/mahGRs.

orgpedia has already crawled, OCR'd and translated every Maharashtra GR into two
text files per resolution: '<id>.pdf.mr.txt' (original Marathi) and
'<id>.pdf.en.txt' (English translation), organised by department. We index the
ORIGINAL Marathi (the authoritative text — see the project's "translate at the
edges, not the middle" decision); the English translation is fetched too (opt-in)
only as a convenience for building the eval gold set.

The order id encodes the date: the first 8 digits are YYYYMMDD.

Usage:
    python scripts/fetch_mahgrs.py                       # 150 HTE GRs -> data/grs_text/
    python scripts/fetch_mahgrs.py --count 300
    python scripts/fetch_mahgrs.py --dept Finance_Department --translations
    python scripts/fetch_mahgrs.py --recent              # newest instead of oldest
"""

import argparse
import json
import os
import time
import urllib.request

API = "https://api.github.com/repos/orgpedia/mahGRs/contents/GRs/{dept}?per_page=100&page={page}"
RAW = "https://raw.githubusercontent.com/orgpedia/mahGRs/main/GRs/{dept}/{name}"


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "mahagr-assist"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def list_mr_files(dept, want, recent):
    """Page through the department folder collecting '*.mr.txt' names until we
    have `want` of them (or run out)."""
    names, page = [], 1
    while len(names) < want * (3 if recent else 1):
        try:
            entries = json.loads(_get(API.format(dept=dept, page=page)))
        except Exception as e:
            print(f"  listing stopped at page {page}: {e}")
            break
        if not entries:
            break
        names += [e["name"] for e in entries if e["name"].endswith(".mr.txt")]
        page += 1
        time.sleep(0.3)
    names = sorted(set(names))
    return names[-want:] if recent else names[:want]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dept", default="Higher_and_Technical_Education_Department")
    ap.add_argument("--count", type=int, default=150)
    ap.add_argument("--out", default="data/grs_text")
    ap.add_argument("--translations", action="store_true", help="also fetch .en.txt")
    ap.add_argument("--recent", action="store_true", help="newest GRs instead of oldest")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Listing {args.dept} ...")
    mr_files = list_mr_files(args.dept, args.count, args.recent)
    if not mr_files:
        print("Nothing found — check the department name against the repo's GRs/ folder.")
        return
    print(f"Downloading {len(mr_files)} GRs into {args.out}/ ...")

    ok = 0
    for i, name in enumerate(mr_files, 1):
        targets = [name] + ([name.replace(".mr.txt", ".en.txt")] if args.translations else [])
        for t in targets:
            dest = os.path.join(args.out, t)
            if os.path.exists(dest):
                continue
            try:
                data = _get(RAW.format(dept=args.dept, name=t))
                with open(dest, "wb") as f:
                    f.write(data)
            except Exception as e:
                print(f"  [{i}] failed {t}: {e}")
        ok += 1
        if i % 25 == 0:
            print(f"  {i}/{len(mr_files)} ...")
        time.sleep(0.15)

    print(f"Done. {ok} GRs in {args.out}/  (department tag: {args.dept})")
    print(f"Next: python scripts/ingest_text.py {args.out} index")


if __name__ == "__main__":
    main()
