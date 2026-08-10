/**
 * System-design diagrams, hand-written as inline SVG.
 *
 * Deliberately not a diagram library (mermaid ~500 kB, react-flow 100-400 kB):
 * these are fixed, presentational pictures with no interaction, so a library
 * would ship a renderer to draw six static boxes and bring its own visual
 * language that would then need restyling to match the portal. They use the
 * same palette tokens as the rest of the site and scale with their container.
 *
 * Every diagram carries <title> for screen readers and a prose caption beside
 * it, so no information exists only inside a picture.
 */

const NAVY = "#0B2545";
const NAVY_700 = "#13315C";
const TEAL = "#1C7293";
const TEAL_BRIGHT = "#2CA6A4";
const ICE = "#EEF3F8";
const LINE = "#D4DEEA";
const SLATE = "#627D98";

/* ── shared primitives ──────────────────────────────────────────────────── */

function Box({
  x,
  y,
  w,
  h,
  label,
  sub,
  fill = "#ffffff",
  stroke = LINE,
  color = NAVY,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  label: string;
  sub?: string;
  fill?: string;
  stroke?: string;
  color?: string;
}) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={8} fill={fill} stroke={stroke} strokeWidth={1.5} />
      <text
        x={x + w / 2}
        y={sub ? y + h / 2 - 4 : y + h / 2 + 4}
        textAnchor="middle"
        fontSize={12.5}
        fontWeight={600}
        fill={color}
      >
        {label}
      </text>
      {sub && (
        <text x={x + w / 2} y={y + h / 2 + 13} textAnchor="middle" fontSize={10.5} fill={SLATE}>
          {sub}
        </text>
      )}
    </g>
  );
}

function Arrow({ x1, y1, x2, y2 }: { x1: number; y1: number; x2: number; y2: number }) {
  return <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={SLATE} strokeWidth={1.5} markerEnd="url(#arrowhead)" />;
}

function Defs() {
  return (
    <defs>
      <marker id="arrowhead" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
        <polygon points="0 0, 8 3, 0 6" fill={SLATE} />
      </marker>
    </defs>
  );
}

function Band({
  x,
  y,
  w,
  h,
  label,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  label: string;
}) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={10} fill={ICE} stroke={LINE} strokeDasharray="4 3" />
      <text x={x + 12} y={y + 17} fontSize={10.5} fontWeight={700} fill={TEAL} letterSpacing={0.6}>
        {label}
      </text>
    </g>
  );
}

/* ── 1. whole-system architecture ───────────────────────────────────────── */

export function ArchitectureDiagram() {
  return (
    <svg viewBox="0 0 900 430" className="h-auto w-full" role="img" aria-labelledby="arch-title">
      <title id="arch-title">
        Three-stage architecture: offline ingestion, the knowledge store, and the online query and
        answer path.
      </title>
      <Defs />

      {/* 1 — ingestion */}
      <Band x={10} y={10} w={880} h={110} label="1 · INGESTION  (offline, GPU)" />
      <Box x={30} y={45} w={130} h={56} label="GR documents" sub="PDF · DOCX · scans" />
      <Arrow x1={162} y1={73} x2={190} y2={73} />
      <Box x={192} y={45} w={120} h={56} label="OCR" sub="Tesseract mar+eng" />
      <Arrow x1={314} y1={73} x2={342} y2={73} />
      <Box x={344} y={45} w={130} h={56} label="Chunk + tables" sub="rows → sentences" />
      <Arrow x1={476} y1={73} x2={504} y2={73} />
      <Box x={506} y={45} w={130} h={56} label="Embed" sub="bge-m3 · 1024-d · fp16" />
      <Arrow x1={638} y1={73} x2={666} y2={73} />
      <Box
        x={668}
        y={45}
        w={200}
        h={56}
        label="Metadata parse"
        sub="GR no · date · dept · refs"
      />

      {/* 2 — knowledge store */}
      <Band x={10} y={135} w={880} h={120} label="2 · KNOWLEDGE STORE  (on disk, /mnt/win)" />
      <Box
        x={40}
        y={172}
        w={240}
        h={64}
        label="FAISS IndexHNSWFlat"
        sub="74,004 vectors · ~O(log n)"
        fill="#ffffff"
        stroke={TEAL}
      />
      <Box
        x={320}
        y={172}
        w={250}
        h={64}
        label="SQLite: documents + chunks"
        sub="text · metadata · FTS5 BM25"
        fill="#ffffff"
        stroke={TEAL}
      />
      <Box
        x={610}
        y={172}
        w={240}
        h={64}
        label="SQLite: gr_edges"
        sub="supersede / cites graph"
        fill="#ffffff"
        stroke={TEAL}
      />
      {/* the join between the two stores, drawn as the single line it is */}
      <line x1={282} y1={204} x2={318} y2={204} stroke={TEAL_BRIGHT} strokeWidth={2} />
      <text x={300} y={252} textAnchor="middle" fontSize={10} fontWeight={600} fill={TEAL}>
        faiss_id
      </text>

      {/* 3 — query */}
      <Band x={10} y={270} w={880} h={150} label="3 · QUERY & ANSWER  (online, per request)" />
      <Box x={30} y={310} w={120} h={56} label="Officer query" sub="English / मराठी" />
      <Arrow x1={152} y1={338} x2={180} y2={338} />
      <Box x={182} y={310} w={130} h={56} label="Hybrid search" sub="HNSW + BM25 → RRF" />
      <Arrow x1={314} y1={338} x2={342} y2={338} />
      <Box x={344} y={310} w={130} h={56} label="Group by GR" sub="chunk → citable unit" />
      <Arrow x1={476} y1={338} x2={504} y2={338} />
      <Box x={506} y={310} w={130} h={56} label="Rerank + gate" sub="cross-encoder" />
      <Arrow x1={638} y1={338} x2={666} y2={338} />
      <Box
        x={668}
        y={310}
        w={200}
        h={56}
        label="Grounded answer"
        sub="cited · or abstains"
        fill={NAVY}
        stroke={NAVY}
        color="#ffffff"
      />
      <text x={768} y={392} textAnchor="middle" fontSize={10.5} fill={SLATE}>
        LLM: qwen2.5:3b via Ollama — local
      </text>

      {/* store → query */}
      <line
        x1={160}
        y1={240}
        x2={247}
        y2={306}
        stroke={SLATE}
        strokeWidth={1.5}
        strokeDasharray="3 3"
        markerEnd="url(#arrowhead)"
      />
      <line
        x1={445}
        y1={240}
        x2={409}
        y2={306}
        stroke={SLATE}
        strokeWidth={1.5}
        strokeDasharray="3 3"
        markerEnd="url(#arrowhead)"
      />
    </svg>
  );
}

/* ── 2. ingestion pipeline detail ───────────────────────────────────────── */

export function IngestionPipelineDiagram() {
  const steps = [
    { t: "Fetch", s: "git trees API, resumable" },
    { t: "Extract", s: "text layer, OCR fallback" },
    { t: "Split", s: "prose vs pipe-tables" },
    { t: "Chunk", s: "250 words, 50 overlap" },
    { t: "Embed", s: "GPU batch 16, fp16" },
    { t: "Persist", s: "FAISS + SQLite" },
  ];
  return (
    <svg viewBox="0 0 900 150" className="h-auto w-full" role="img" aria-labelledby="ing-title">
      <title id="ing-title">
        Six-step offline ingestion pipeline, from fetching GRs to persisting vectors and text.
      </title>
      <Defs />
      {steps.map((st, i) => {
        const x = 12 + i * 148;
        return (
          <g key={st.t}>
            <Box x={x} y={40} w={130} h={58} label={st.t} sub={st.s} />
            <circle cx={x + 12} cy={52} r={9} fill={TEAL} />
            <text x={x + 12} y={56} textAnchor="middle" fontSize={10} fontWeight={700} fill="#fff">
              {i + 1}
            </text>
            {i < steps.length - 1 && <Arrow x1={x + 132} y1={69} x2={x + 146} y2={69} />}
          </g>
        );
      })}
      <text x={450} y={126} textAnchor="middle" fontSize={11} fill={SLATE}>
        Resumable and idempotent — the saved FAISS index is the authority; SQLite rows past its
        ntotal are dropped and re-ingested on restart.
      </text>
    </svg>
  );
}

/* ── 3. query / retrieval pipeline detail ───────────────────────────────── */

export function RetrievalPipelineDiagram() {
  return (
    <svg viewBox="0 0 900 340" className="h-auto w-full" role="img" aria-labelledby="ret-title">
      <title id="ret-title">
        Two-stage retrieval: approximate nearest-neighbour and BM25 candidates fused by reciprocal
        rank fusion, hydrated from SQLite, grouped per GR, reranked by a cross-encoder and gated by
        a score threshold before generation.
      </title>
      <Defs />

      <Box x={370} y={8} w={160} h={46} label="Question" sub="EN / मराठी" fill={NAVY} stroke={NAVY} color="#fff" />

      <Arrow x1={420} y1={56} x2={250} y2={82} />
      <Arrow x1={480} y1={56} x2={650} y2={82} />

      <Box x={140} y={84} w={220} h={54} label="Dense: FAISS HNSW" sub="efSearch 512 · top 60 chunks" stroke={TEAL} />
      <Box x={540} y={84} w={220} h={54} label="Sparse: SQLite FTS5 BM25" sub="Devanagari-aware tokens" stroke={TEAL} />

      <Arrow x1={250} y1={140} x2={420} y2={166} />
      <Arrow x1={650} y1={140} x2={480} y2={166} />

      <Box x={340} y={168} w={220} h={48} label="RRF fusion" sub="rank-based, scale-free" />
      <Arrow x1={450} y1={218} x2={450} y2={240} />
      <Box x={310} y={242} w={280} h={44} label="Hydrate from SQLite → group by GR" sub="chunks find the passage; the GR is the citable unit" />
      <Arrow x1={450} y1={288} x2={450} y2={306} />
      <Box
        x={280}
        y={296}
        w={340}
        h={40}
        label="Cross-encoder rerank → threshold gate"
        fill={NAVY}
        stroke={NAVY}
        color="#fff"
      />

      <text x={90} y={116} textAnchor="middle" fontSize={10.5} fill={SLATE}>
        meaning
      </text>
      <text x={820} y={116} textAnchor="middle" fontSize={10.5} fill={SLATE}>
        exact terms
      </text>
      <text x={700} y={266} fontSize={10.5} fill={TEAL} fontWeight={600}>
        filters pushed into
      </text>
      <text x={700} y={280} fontSize={10.5} fill={TEAL} fontWeight={600}>
        the search, not after
      </text>
    </svg>
  );
}

/* ── 4. supersede knowledge graph ───────────────────────────────────────── */

export function GraphDiagram() {
  return (
    <svg viewBox="0 0 900 250" className="h-auto w-full" role="img" aria-labelledby="graph-title">
      <title id="graph-title">
        A supersede chain: an older GR is replaced by a newer one, which is itself replaced. A
        dangling reference points at an order the corpus does not hold.
      </title>
      <Defs />

      <Box x={40} y={90} w={170} h={56} label="GR 2018" sub="प्रामाअ-२०१८/…/एसएम-४" />
      <Arrow x1={212} y1={118} x2={268} y2={118} />
      <text x={240} y={108} textAnchor="middle" fontSize={10} fill={TEAL} fontWeight={700}>
        supersedes
      </text>

      <Box x={270} y={90} w={170} h={56} label="GR 2021" sub="उमाशा-२०२१/…/एसएम-४" />
      <Arrow x1={442} y1={118} x2={498} y2={118} />
      <text x={470} y={108} textAnchor="middle" fontSize={10} fill={TEAL} fontWeight={700}>
        supersedes
      </text>

      <Box
        x={500}
        y={90}
        w={190}
        h={56}
        label="GR 2023 — in force"
        fill={NAVY}
        stroke={NAVY}
        color="#fff"
      />

      {/* dangling ghost */}
      <rect
        x={720}
        y={90}
        width={160}
        height={56}
        rx={8}
        fill="#ffffff"
        stroke={SLATE}
        strokeWidth={1.5}
        strokeDasharray="5 4"
      />
      <text x={800} y={114} textAnchor="middle" fontSize={12} fontWeight={600} fill={SLATE}>
        Cited order
      </text>
      <text x={800} y={129} textAnchor="middle" fontSize={10.5} fill={SLATE}>
        not in corpus
      </text>
      <line
        x1={692}
        y1={118}
        x2={716}
        y2={118}
        stroke={SLATE}
        strokeWidth={1.5}
        strokeDasharray="4 3"
        markerEnd="url(#arrowhead)"
      />

      <text x={450} y={196} textAnchor="middle" fontSize={11.5} fill={NAVY} fontWeight={600}>
        Every edge is stored three-valued: resolved · dangling · ambiguous.
      </text>
      <text x={450} y={216} textAnchor="middle" fontSize={11} fill={SLATE}>
        Danglers are kept, not dropped — &ldquo;this GR builds on an order we do not hold&rdquo; is
        information an officer needs.
      </text>
    </svg>
  );
}
