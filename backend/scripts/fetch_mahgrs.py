"""
fetch_mahgrs.py — download the real GR corpus from orgpedia/mahGRs, one folder
per department, resumably.

orgpedia has already crawled, OCR'd and translated every Maharashtra GR into two
text files per resolution: '<id>.pdf.mr.txt' (original Marathi) and
'<id>.pdf.en.txt' (English translation), organised by department. We index the
ORIGINAL Marathi (the authoritative text — see the project's "translate at the
edges, not the middle" decision); the English translation is fetched too (opt-in)
only as a convenience for building the eval gold set.

The order id encodes the date: the first 8 digits are YYYYMMDD.

WHY THE GIT TREES API, NOT THE CONTENTS API
-------------------------------------------
The contents API pages at 100 entries, so a 4,700-GR department (9,400 files
counting translations) costs ~94 requests just to LIST it — and unauthenticated
GitHub allows 60 requests/hour. Six departments would exhaust the quota before a
single byte of GR text was downloaded. The git trees API returns a whole
directory recursively in ONE request, so listing all six departments costs 8
requests total. It also hands back each blob's `size`, which is what makes the
"how much am I about to download" estimate honest instead of a guess.
(The repo-root tree IS truncated at ~53k entries — that's why we descend to the
GRs/ tree and then to each department's own tree, none of which truncate.)

RESUMABLE, BY DESIGN
--------------------
At 18k files a download WILL be interrupted (a laptop sleeps, a network drops).
Every file is skipped if it already exists on disk, so re-running the command is
a cheap no-op for everything already fetched — the same idempotence contract
scripts/ingest_corpus.py relies on. Downloads run on a small thread pool because
the job is latency-bound (18k round-trips), not bandwidth-bound.

Usage:
    python scripts/fetch_mahgrs.py --list                  # departments + GR counts
    python scripts/fetch_mahgrs.py --cluster education     # the 6-department corpus
    python scripts/fetch_mahgrs.py --dept Finance_Department --count 300
    python scripts/fetch_mahgrs.py --cluster education --dry-run
"""

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO = "orgpedia/mahGRs"
TREE = "https://api.github.com/repos/{repo}/git/trees/{sha}"
RAW = "https://raw.githubusercontent.com/{repo}/main/GRs/{dept}/{name}"

# The education & skilling cluster (PLAN.md Phase 2). The problem statement
# targets Higher & Technical Education; the rest are the departments an HTE desk
# officer actually cross-references — school education feeds admissions, medical
# education and skill development share the same admission/fee machinery, and
# social justice / tribal development own the scholarship GRs.
CLUSTERS = {
    "education": [
        "Higher_and_Technical_Education_Department",
        "School_Education_and_Sports_Department",
        "Medical_Education_and_Drugs_Department",
        "Skill_Development_and_Entrepreneurship_Department",
        "Social_Justice_and_Special_Assistance_Department",
        "Tribal_Development_Department",
    ],
    # PLAN Phase 6, wave A: the departments the education cluster CITES most but
    # does not hold. NOT a guess — scripts/cited_departments.py counted the
    # department named in every GR's 'वाचा' block across all 18,080 documents:
    #
    #   Finance                   5,624 docs cite it (32.4%)   1,046 GRs to ingest
    #   General Administration    1,670 (9.6%)                 6,345
    #   Industries, Energy, Labour 1,602 (9.2%)                2,885
    #   Public Health               942 (5.4%)                 7,076
    #   Public Works                682 (3.9%)                 3,809
    #   Planning                    526 (3.0%)                 2,233
    #
    # Finance is the whole argument for measuring instead of guessing: a THIRD of
    # the corpus cites it, and it is one of the smallest departments in the
    # dataset — 1,046 GRs, ~12 MB, minutes of GPU. Every one of those references
    # is currently a `dangling` edge purely because we never downloaded it.
    # (The departments an earlier note guessed — Revenue & Forest, Law &
    # Judiciary — measured 0.5% and 0.3%. They are in `--all`, not here.)
    "cited": [
        "Finance_Department",
        "General_Administration_Department",
        "Industries,_Energy_and_Labour_Department",
        "Public_Health_Department",
        "Public_Works_Department",
        "Planning_Department",
    ],
}

# Default corpus root. Root '/' has ~3 GB free on this machine, so the corpus,
# the index and the DB all live on the big partition (see HANDOFF §5).
DEFAULT_OUT = os.environ.get("MAHAGR_CORPUS", "/mnt/win/mahagr/corpus")


def _get(url, retries=3):
    """GET with a small retry — GitHub occasionally 5xxs under a burst of
    parallel raw requests, and losing one file out of 18k to a transient blip
    would silently leave a hole in the corpus."""
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mahagr-assist"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):        # rate limited — back off hard
                time.sleep(5 * (attempt + 1))
            elif e.code == 404:             # genuinely absent; no point retrying
                raise
            else:
                time.sleep(1 + attempt)
            last = e
        except Exception as e:
            time.sleep(1 + attempt)
            last = e
    raise last


def _tree(sha):
    return json.loads(_get(TREE.format(repo=REPO, sha=sha)))


def department_shas():
    """{department_name: tree_sha} for every department folder in GRs/."""
    root = _tree("main")
    grs = next(e for e in root["tree"] if e["path"] == "GRs" and e["type"] == "tree")
    return {e["path"]: e["sha"] for e in _tree(grs["sha"])["tree"] if e["type"] == "tree"}


_LIST_CACHE = {}


def list_department(sha, translations=False):
    """Memoized wrapper — see _list_department.

    Unauthenticated GitHub allows **60 API requests per hour** and each listing
    costs one, so a caller that lists the same department twice (e.g. --all,
    which sorts by size and then downloads) can silently exhaust the quota and
    fail halfway through. Caching makes a repeat listing free.
    """
    key = (sha, translations)
    if key not in _LIST_CACHE:
        _LIST_CACHE[key] = _list_department(sha, translations)
    return _LIST_CACHE[key]


def _list_department(sha, translations=False):
    """[(filename, size_bytes), ...] for one department, oldest order id first.

    Sorting by name is sorting by DATE — the order id starts with YYYYMMDD — so
    --count takes a chronologically coherent slice rather than an arbitrary one.
    """
    tree = _tree(sha)
    if tree.get("truncated"):
        print("  ! department tree was truncated by GitHub — file list is incomplete",
              file=sys.stderr)
    wanted = (".mr.txt", ".en.txt") if translations else (".mr.txt",)
    return sorted((e["path"], e.get("size", 0)) for e in tree["tree"]
                  if e["type"] == "blob" and e["path"].endswith(wanted))


def download_department(dept, files, out_root, workers=8):
    """Fetch `files` into out_root/dept/, skipping anything already on disk.
    Returns (downloaded, skipped, failed)."""
    dest_dir = os.path.join(out_root, dept)
    os.makedirs(dest_dir, exist_ok=True)

    todo = [(n, os.path.join(dest_dir, n)) for n, _ in files
            if not os.path.exists(os.path.join(dest_dir, n))]
    skipped = len(files) - len(todo)
    counter, failures = {"n": 0}, []
    lock = threading.Lock()

    def fetch(item):
        name, dest = item
        try:
            # URL-ENCODE the path components. Real orgpedia filenames contain
            # SPACES — e.g. Revenue_and_Forest's '202510171720567619 .pdf.mr.txt'
            # (note the space before '.pdf') — and urllib refuses a URL with an
            # unencoded space outright: "URL can't contain control characters".
            # That is a DETERMINISTIC failure, not a flaky one, so _get's retry
            # loop could never help: the file simply never downloads, and a
            # 99,420-of-99,421 corpus looks like a rounding error rather than a
            # bug. Department names are quoted too — several contain ',' and '-'.
            data = _get(RAW.format(repo=REPO,
                                   dept=urllib.parse.quote(dept),
                                   name=urllib.parse.quote(name)))
        except Exception as e:
            with lock:
                failures.append((name, str(e)))
            return
        # Write-then-rename: an interrupted run must never leave a half-written
        # file that the next (resumable) run would skip as "already present".
        with open(dest + ".part", "wb") as f:
            f.write(data)
        os.replace(dest + ".part", dest)
        with lock:
            counter["n"] += 1
            if counter["n"] % 250 == 0:
                print(f"    {counter['n']}/{len(todo)} ...", flush=True)

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(fetch, todo))

    return counter["n"], skipped, failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dept", action="append", default=[],
                    help="department folder name (repeatable)")
    ap.add_argument("--cluster", choices=sorted(CLUSTERS),
                    help="a named group of departments (see CLUSTERS)")
    ap.add_argument("--all", action="store_true",
                    help="every department in the dataset (33, ~99,421 GRs, ~1.1 GB)")
    ap.add_argument("--count", type=int, default=0,
                    help="max GRs per department (0 = all)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--translations", action="store_true", help="also fetch .en.txt")
    ap.add_argument("--recent", action="store_true", help="newest GRs instead of oldest")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--list", action="store_true", help="list departments and exit")
    ap.add_argument("--dry-run", action="store_true", help="show the plan, download nothing")
    args = ap.parse_args()

    print(f"Listing departments in {REPO} ...")
    shas = department_shas()

    if args.list:
        for dept in sorted(shas):
            files = list_department(shas[dept])
            mb = sum(s for _, s in files) / 1e6
            print(f"  {dept:60s} {len(files):6d} GRs  {mb:7.1f} MB")
        return

    # --all is sorted by GR count, SMALLEST FIRST. The download is resumable, so
    # the only thing ordering changes is what you have if it stops early — and
    # finishing 20 small departments beats being 60% through one large one.
    if args.all:
        depts = sorted(shas, key=lambda d: len(list_department(shas[d])))
    else:
        depts = args.dept or CLUSTERS.get(args.cluster or "", [])
    if not depts:
        print("Nothing to do — pass --dept, --cluster or --all (or --list to browse).")
        return

    unknown = [d for d in depts if d not in shas]
    if unknown:
        print(f"Unknown department(s): {unknown}\nRun --list to see valid names.")
        return

    plan, total_files, total_bytes = [], 0, 0
    for dept in depts:
        files = list_department(shas[dept], args.translations)
        if args.count:
            per_doc = 2 if args.translations else 1
            files = files[-args.count * per_doc:] if args.recent else files[:args.count * per_doc]
        plan.append((dept, files))
        total_files += len(files)
        total_bytes += sum(s for _, s in files)
        print(f"  {dept:60s} {len(files):6d} files  {sum(s for _, s in files)/1e6:7.1f} MB")

    print(f"\nTotal: {total_files} files, {total_bytes/1e6:.1f} MB -> {args.out}/")
    if args.dry_run:
        print("(dry run — nothing downloaded)")
        return

    started = time.time()
    grand = {"got": 0, "skip": 0, "fail": 0}
    for dept, files in plan:
        print(f"\n{dept} ...", flush=True)
        got, skipped, failures = download_department(dept, files, args.out, args.workers)
        grand["got"] += got
        grand["skip"] += skipped
        grand["fail"] += len(failures)
        print(f"  downloaded {got}, already present {skipped}, failed {len(failures)}")
        for name, err in failures[:5]:
            print(f"    ! {name}: {err}")

    print("\n" + "=" * 60)
    print(f"Downloaded {grand['got']}, skipped {grand['skip']}, failed {grand['fail']} "
          f"in {time.time() - started:.0f}s")
    if grand["fail"]:
        print("Re-run the same command to retry the failures (it is resumable).")
    print(f"Next: python scripts/ingest_corpus.py --corpus {args.out}")


if __name__ == "__main__":
    main()
