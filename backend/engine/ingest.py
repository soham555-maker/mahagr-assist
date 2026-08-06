import os
import re
from collections import defaultdict

# pyrefly: ignore [missing-import]
import fitz  # PyMuPDF
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer

from engine import config, table_extract


def _ocr_page(page):
    """OCR one PDF page that has no text layer (a scanned Government Resolution).

    Rendered to an image at config.OCR_DPI and read by Tesseract with the
    Marathi + Hindi + English models (config.OCR_LANGS). Returns '' — never
    raises — if pytesseract or the system Tesseract/language data isn't
    installed, so a machine without OCR set up still ingests its text-layer
    PDFs fine; it just can't read scans. The one-time warning names what to
    install.
    """
    try:
        # pyrefly: ignore [missing-import]
        import io
        # pyrefly: ignore [missing-import]
        import pytesseract
        # pyrefly: ignore [missing-import]
        from PIL import Image
    except ImportError:
        if not getattr(_ocr_page, "_warned", False):
            print("  [ocr] pytesseract/Pillow not installed — scanned pages stay empty. "
                  "Install: pip install pytesseract pillow  + system tesseract with mar/hin data.")
            _ocr_page._warned = True
        return ""

    try:
        pix = page.get_pixmap(dpi=config.OCR_DPI)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img, lang=config.OCR_LANGS)
    except Exception as e:  # missing tesseract binary or a lang file lands here
        if not getattr(_ocr_page, "_warned", False):
            print(f"  [ocr] OCR unavailable ({e}) — scanned pages stay empty.")
            _ocr_page._warned = True
        return ""


def _resolve_caption(page_text, label):
    """
    Recover the full, well-spaced caption for a table from PyMuPDF page text.

    table_extract hands us only the anchor token ("Table4:"); the descriptive
    caption ("Comparison of publicly available word vectors...") lives in the page
    prose, where PyMuPDF has reconstructed the spacing that pdfplumber dropped. We
    locate the caption line by its table number, then pull the wrapped continuation
    lines (rejoining line-break hyphens), stopping when the prose ends or the table
    grid begins. Falls back to the bare label if no caption line is found.
    """
    m = re.search(r"Table\s*(\d+)", label or "")
    if not m:
        return (label or "").strip()
    num = m.group(1)
    lines = [ln.strip() for ln in page_text.split("\n")]

    # Prefer the colon form ("Table 4:") -- that's the caption itself; a period
    # form ("Table 4.") or bare mention is usually a body-text reference to it.
    starts = [i for i, ln in enumerate(lines) if re.match(rf"^Table\s*{num}\b", ln)]
    start = next((i for i in starts if re.match(rf"^Table\s*{num}\s*:", lines[i])),
                 starts[0] if starts else None)
    if start is None:
        return (label or "").strip()

    collected = [lines[start]]
    for ln in lines[start + 1:start + 6]:
        if not ln:
            break
        if re.match(r"^(Table|Figure)\s*\d", ln):   # the next caption -> stop
            break
        words = ln.split()
        numeric = sum(1 for w in words if re.search(r"\d", w))
        if words and numeric / len(words) > 0.5:    # a data row -> stop
            break
        collected.append(ln)
        if sum(len(c) for c in collected) > 500:
            break

    text = ""
    for c in collected:
        if text.endswith("-"):          # rejoin a word split by a line-break hyphen
            text = text[:-1] + c
        elif text:
            text += " " + c
        else:
            text = c
    return " ".join(text.split())[:500]


class IngestionPipeline:
    def __init__(self, model_name=config.EMBED_MODEL, model=None):
        """
        Initializes the ingestion pipeline with the specified embedding model.

        Defaults to config.EMBED_MODEL (bge-m3, multilingual) — the single
        source of truth shared with retrieval and the vector store, so query
        and document embeddings are always produced by the same model.

        model, if given, is reused instead of loading a new SentenceTransformer
        (model_name is then ignored) -- lets a caller that already has the model
        loaded (e.g. the FastAPI backend, which loads it once for retrieval)
        ingest uploads without a second multi-hundred-MB model load.
        """
        self.model = model or SentenceTransformer(model_name, device=config.EMBED_DEVICE)

    def extract_pages_from_pdf(self, pdf_path, force_ocr=False):
        """
        Extracts text page by page.
        Returns a list of (page_number, text) tuples, 1-indexed.
        Raises RuntimeError if the file can't be opened at all.

        SCANNED-DOCUMENT FALLBACK: a Government Resolution that is a scanned
        image has no text layer, so page.get_text() comes back empty. For any
        such page we fall back to OCR (Marathi + Hindi + English) so scanned
        GRs are ingested just like born-digital ones. Pages that already carry
        a text layer never pay the OCR cost.

        force_ocr=True OCRs EVERY page even when a text layer exists — needed for
        government PDFs whose embedded font has a broken ToUnicode map, so the
        text layer extracts as garbled Devanagari ("निर्णय" -> "चनणचय"). OCR of
        the rendered page recovers correct Marathi that the text layer cannot.
        """
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            raise RuntimeError(f"Could not open {pdf_path}: {e}") from e

        pages = []
        for page_num, page in enumerate(doc, start=1):
            text = "" if force_ocr else page.get_text("text")
            if force_ocr or not text.strip():    # forced, or no text layer (scanned)
                text = _ocr_page(page)
            pages.append((page_num, text))
        doc.close()
        return pages

    def table_sentences_by_page(self, pdf_path, page_texts=None):
        """
        Extract every table in the PDF (see table_extract) and turn it into
        natural-language sentences, grouped by the page it appears on.

        Returns {page_number: [(caption, [sentence, ...]), ...]} -- one tuple per
        table: its resolved caption plus its rows rendered as sentences. Embedding
        models are trained on prose, not grids, so a row like ["BERT", "94.5"]
        embeds badly; row_to_sentence turns it into "... BERT is 94.5 ..." first.

        page_texts ({page_number: text}) is the PyMuPDF page text, used to recover
        each table's full caption (see _resolve_caption). When omitted, the bare
        "Table N" label is used instead.

        Table extraction is best-effort: if pdfplumber fails on a file we return
        nothing and let ingestion proceed on the text alone.
        """
        by_page = defaultdict(list)
        try:
            tables = table_extract.extract_tables(pdf_path)
        except Exception as e:
            print(f"Table extraction skipped for {pdf_path}: {e}")
            return by_page

        for page_number, label, rows in tables:
            if not rows:
                continue
            header = rows[0]
            sentences = [
                s for s in (table_extract.row_to_sentence(header, row) for row in rows[1:]) if s
            ]
            if not sentences:
                continue
            page_text = (page_texts or {}).get(page_number, "")
            caption = _resolve_caption(page_text, label)
            by_page[page_number].append((caption, sentences))

        return by_page

    def _assemble_table_chunk(self, caption, sentences, page_number):
        """One table chunk: caption on top, then row sentences, one per line,
        attributed to the single page the table sits on."""
        lines = ([caption] if caption else []) + sentences
        return {
            'text': "\n".join(lines).strip(),
            'page_start': page_number,
            'page_end': page_number,
        }

    def _chunk_one_table(self, caption, sentences, page_number, chunk_size):
        """
        Turn one table into one or more chunks. Rows are packed whole (never split
        mid-sentence) until adding the next would exceed chunk_size words; then a
        new chunk starts. The caption is repeated at the top of every chunk, so a
        table that spills across chunks keeps its heading on each piece.
        """
        cap_len = len((caption or "").split())
        budget = max(1, chunk_size - cap_len)  # words left for row sentences per chunk

        chunks, current, current_len = [], [], 0
        for sent in sentences:
            slen = len(sent.split())
            if current and current_len + slen > budget:
                chunks.append(self._assemble_table_chunk(caption, current, page_number))
                current, current_len = [], 0
            current.append(sent)
            current_len += slen
        if current:
            chunks.append(self._assemble_table_chunk(caption, current, page_number))
        return chunks

    def build_table_chunks(self, pdf_path, page_texts, chunk_size=250):
        """
        Build standalone chunks for every table in the PDF, each caption-first and
        chunked independently of the surrounding prose. Keeping tables out of the
        250-word prose windows stops a table's content from being diluted by
        unrelated page text, so a query about the table (its caption, its data)
        matches a focused chunk instead of a mostly-prose one.
        Returns [{'text', 'page_start', 'page_end'}, ...].
        """
        by_page = self.table_sentences_by_page(pdf_path, page_texts)
        chunks = []
        for page_number in sorted(by_page):
            for caption, sentences in by_page[page_number]:
                chunks.extend(
                    self._chunk_one_table(caption, sentences, page_number, chunk_size)
                )
        return chunks

    def append_table_sentences(self, pages, pdf_path):
        """
        Append each page's table sentences to that page's text. Used only by the
        dump viewer to show tables inline with prose; the real pipeline chunks
        tables separately (see build_table_chunks).
        """
        page_texts = {page_number: text for page_number, text in pages}
        by_page = self.table_sentences_by_page(pdf_path, page_texts)
        if not by_page:
            return pages

        merged = []
        for page_number, text in pages:
            tables = by_page.get(page_number)
            if tables:
                blocks = [
                    "\n".join(([cap] if cap else []) + sents) for cap, sents in tables
                ]
                text = (text.rstrip() + "\n" + " ".join(blocks)).strip()
            merged.append((page_number, text))
        return merged

    def chunk_pages(self, pages, chunk_size=250, overlap=50):
        """
        Splits (page_number, text) pairs into overlapping word chunks.
        Each chunk records the page range it was drawn from, since a chunk
        can straddle a page boundary once pages are flattened into one word stream.
        Returns a list of dicts: {'text', 'page_start', 'page_end'}.
        """
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        tagged_words = []
        for page_num, text in pages:
            for word in re.split(r'\s+', text.strip()):
                if word:
                    tagged_words.append((word, page_num))

        chunks = []
        i = 0
        while i < len(tagged_words):
            window = tagged_words[i:i + chunk_size]
            chunks.append({
                'text': " ".join(word for word, _ in window),
                'page_start': window[0][1],
                'page_end': window[-1][1],
            })
            i += chunk_size - overlap

        return chunks

    def process_pdf(self, pdf_path, chunk_size=250, overlap=50, source_type='corpus',
                    include_tables=True, extra_metadata=None, pages=None):
        """
        Extracts text, chunks it, and embeds each chunk.
        Returns a list of dicts: [{'text', 'embedding', 'metadata'}, ...]
        metadata carries source_file, source_type, chunk_index, page_start, page_end —
        everything Day 3 (FAISS tagging) and Day 12 (plagiarism source citation) need.

        pages, if given, is the already-extracted [(page_no, text), ...] and is
        used instead of re-reading the PDF — so a caller that has to read the
        text first (e.g. ingest_grs.py, to parse GR metadata) doesn't pay the
        extraction (and, for scans, the OCR) cost twice.

        extra_metadata (a dict or None) is merged into every chunk's metadata, so
        Day 3 can thread per-paper tags like paper_id/title/topic_group straight
        onto each chunk. The six core fields above always win on a key clash, so a
        caller can never accidentally overwrite the invariant metadata.

        Embeddings are L2-normalized (unit length) at encode time, so a plain inner
        product against them IS cosine similarity — this is what lets the FAISS
        IndexFlatIP index (Day 3) and the plagiarism score (Day 9-10) both read as
        exact cosine without re-normalizing anywhere downstream.

        Prose and tables are chunked SEPARATELY and tagged content_type 'text' or
        'table'. Prose flows through 250-word overlapping windows; each table becomes
        its own caption-first chunk(s). Keeping tables out of the prose windows means
        a table chunk isn't diluted by unrelated page text, so table content (its
        caption, its data) retrieves as a focused unit. content_type lets retrieval
        (Days 4-5) search or weight the two modalities independently.
        """
        if pages is None:
            pages = self.extract_pages_from_pdf(pdf_path)

        if not any(text.strip() for _, text in pages):
            return []

        # Prose chunks -- tables are NOT merged in; they're chunked on their own below.
        chunks = self.chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
        for chunk in chunks:
            chunk['content_type'] = 'text'

        # Table chunks -- each table its own caption-first chunk(s).
        if include_tables:
            page_texts = {page_number: text for page_number, text in pages}
            for table_chunk in self.build_table_chunks(pdf_path, page_texts, chunk_size=chunk_size):
                table_chunk['content_type'] = 'table'
                chunks.append(table_chunk)

        if not chunks:
            return []

        texts = [c['text'] for c in chunks]
        embeddings = self.model.encode(texts, convert_to_tensor=False,
                                       normalize_embeddings=True)

        source_file = os.path.basename(pdf_path)
        results = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            metadata = {
                'source_file': source_file,
                'source_type': source_type,
                'content_type': chunk['content_type'],
                'chunk_index': idx,
                'page_start': chunk['page_start'],
                'page_end': chunk['page_end'],
            }
            if extra_metadata:
                # Core fields listed second so they override on any key clash:
                # the six invariant fields can never be clobbered by a caller.
                metadata = {**extra_metadata, **metadata}
            results.append({
                'text': chunk['text'],
                'embedding': embedding,
                'metadata': metadata,
            })

        return results

    @staticmethod
    def _split_pages(text):
        """Split already-extracted text into [(page_no, text), ...] on the
        '# Page N' markers the orgpedia GR .txt files carry, so page provenance
        survives into citations. Text without markers becomes a single page 1."""
        parts = re.split(r"(?m)^#\s*Page\s*(\d+)\s*$", text)
        if len(parts) == 1:
            return [(1, text)]
        pages, i = [], 1
        while i < len(parts):
            num = int(parts[i])
            body = parts[i + 1] if i + 1 < len(parts) else ""
            pages.append((num, body))
            i += 2
        return pages or [(1, text)]

    def process_text(self, text, source_file, chunk_size=250, overlap=50,
                     source_type="gr", extra_metadata=None):
        """Ingest PRE-EXTRACTED text (e.g. an orgpedia .mr.txt GR that was
        already OCR'd upstream) — same {text, embedding, metadata} chunk
        contract as process_pdf, but no PDF/OCR/pdfplumber step. Pages come from
        the '# Page N' markers; there is no separate table modality (OCR text
        has no ruled lines), so every chunk is content_type 'text'."""
        pages = self._split_pages(text)
        if not any(t.strip() for _, t in pages):
            return []
        chunks = self.chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
        for c in chunks:
            c["content_type"] = "text"
        return self._embed_chunks(chunks, source_file, source_type, extra_metadata)

    def _embed_chunks(self, chunks, pdf_path, source_type, extra_metadata):
        """Shared tail of process_pdf/process_pdf_v2: batch-embed every chunk's
        text in ONE encode() call and attach the standard metadata. A chunk may
        carry a transient 'image_bytes' (visual assets, v2 only); it is passed
        through on the result dict untouched (never embedded, never put in
        metadata) so api.py can upload it to Storage and stamp the path."""
        if not chunks:
            return []

        texts = [c['text'] for c in chunks]
        embeddings = self.model.encode(texts, convert_to_tensor=False,
                                       normalize_embeddings=True)

        source_file = os.path.basename(pdf_path)
        results = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            metadata = {
                'source_file': source_file,
                'source_type': source_type,
                'content_type': chunk['content_type'],
                'chunk_index': idx,
                'page_start': chunk['page_start'],
                'page_end': chunk['page_end'],
            }
            if extra_metadata:
                metadata = {**extra_metadata, **metadata}
            result = {'text': chunk['text'], 'embedding': embedding, 'metadata': metadata}
            if 'image_bytes' in chunk:
                result['image_bytes'] = chunk['image_bytes']
            results.append(result)
        return results

    def process_pdf_v2(self, pdf_path, chunk_size=250, overlap=50,
                       source_type='corpus', extra_metadata=None):
        """
        Upload-pipeline v2: same {text, embedding, metadata} chunk contract as
        process_pdf, but ONE layout-model pass (visual_ingest) supplies prose
        markdown AND figure/formula crops:

          * text chunks -- pymupdf4llm markdown per page (tables kept INLINE),
                           chunked by the SAME chunk_pages word-window logic so
                           page provenance is preserved. No separate pdfplumber
                           pass -- that ~4.4s is the deliberate speed win; a
                           table's data stays next to its caption in the text
                           chunk, and upload retrieval is flat (not per-modality)
                           so a distinct 'table' type would add nothing here.
          * figure/formula chunks -- caption+context text as the embedded body;
                           each carries a transient image_bytes (the PNG crop)
                           for api.py to persist to Storage.

        RESILIENCE: if the layout model fails on a PDF, fall back to v1
        (process_pdf, which DOES run pdfplumber tables) so an upload never
        regresses to an error just because a file confuses the heavier parser.
        """
        from engine import visual_ingest  # lazy: pulls in pymupdf4llm/onnxruntime only when uploads run

        try:
            doc, parsed = visual_ingest.parse_pdf(pdf_path)
            try:
                prose = visual_ingest.prose_pages(parsed)  # tables kept inline
                assets = visual_ingest.visual_assets(doc, parsed)
            finally:
                doc.close()
        except Exception as e:  # heavier parser choked -> don't lose the upload
            print(f"visual parse failed for {pdf_path} ({e}); falling back to v1")
            return self.process_pdf(pdf_path, chunk_size=chunk_size, overlap=overlap,
                                    source_type=source_type, extra_metadata=extra_metadata)

        if not any(md.strip() for _, md in prose):
            return []

        chunks = self.chunk_pages(prose, chunk_size=chunk_size, overlap=overlap)
        for chunk in chunks:
            chunk['content_type'] = 'text'

        for asset in assets:
            chunks.append({
                'text': asset['text'],
                'page_start': asset['page'],
                'page_end': asset['page'],
                'content_type': asset['content_type'],
                'image_bytes': asset['image_bytes'],
            })

        return self._embed_chunks(chunks, pdf_path, source_type, extra_metadata)

    def dump_pages(self, pdf_path, out_path=None, include_tables=True):
        """
        Write the full extracted text for every page -- with that page's table
        sentences appended -- to a file. This is exactly what gets fed to
        chunking/embedding. Does not embed, so it runs without the model loaded.

        Returns the path written. If out_path is None, writes to the current
        directory as "<pdf-stem>_extracted.txt".
        """
        pages = self.extract_pages_from_pdf(pdf_path)
        if include_tables:
            pages = self.append_table_sentences(pages, pdf_path)

        if out_path is None:
            stem = os.path.splitext(os.path.basename(pdf_path))[0]
            out_path = f"{stem}_extracted.txt"

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"Extraction of: {pdf_path}\n")
            f.write(f"Total pages: {len(pages)}\n\n")
            for page_number, text in pages:
                f.write("=" * 70 + "\n")
                f.write(f"PAGE {page_number}\n")
                f.write("=" * 70 + "\n")
                f.write((text.strip() if text.strip() else "(no extractable text)") + "\n\n")

        return out_path


if __name__ == "__main__":
    import sys

    # `python ingest.py dump <pdf> [out.txt]` -- write page-wise text + table
    # sentences to a file. Skips model loading (via __new__): no embeddings needed.
    if len(sys.argv) > 2 and sys.argv[1] == "dump":
        viewer = IngestionPipeline.__new__(IngestionPipeline)
        out = sys.argv[3] if len(sys.argv) > 3 else None
        try:
            written = viewer.dump_pages(sys.argv[2], out_path=out)
        except RuntimeError as e:
            print(f"Failed to read {sys.argv[2]}: {e}")
            sys.exit(1)
        print(f"Wrote extraction to {written}")

    elif len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        pipeline = IngestionPipeline()
        try:
            results = pipeline.process_pdf(pdf_path)
        except RuntimeError as e:
            print(f"Failed to process {pdf_path}: {e}")
            sys.exit(1)

        print(f"Extracted {len(results)} chunks.")
        if results:
            print(f"Embedding dimension: {len(results[0]['embedding'])}")
            print("\nSample chunk 0:")
            print(results[0]['text'][:200] + "...")
            print("\nSample metadata:")
            print(results[0]['metadata'])
