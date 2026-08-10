import Link from "next/link";
import {
  ArrowRight,
  BookOpenCheck,
  Boxes,
  CircleSlash,
  Cpu,
  Database,
  FileSearch,
  Fingerprint,
  Gauge,
  Languages,
  Layers,
  Lock,
  Network,
  Quote,
  ScanText,
  ServerCog,
  ShieldCheck,
  Table2,
  TriangleAlert,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";

import { LandingFooter, LandingHeader } from "@/components/landing/site-chrome";
import { LiveCorpusStat } from "@/components/landing/live-corpus-stat";
import {
  ArchitectureDiagram,
  GraphDiagram,
  IngestionPipelineDiagram,
  RetrievalPipelineDiagram,
} from "@/components/landing/diagrams";

/* ── small section helpers ──────────────────────────────────────────────── */

function Section({
  id,
  eyebrow,
  title,
  lead,
  children,
  tone = "light",
}: {
  id?: string;
  eyebrow: string;
  title: string;
  lead?: string;
  children: React.ReactNode;
  tone?: "light" | "ice" | "navy";
}) {
  const bg = tone === "navy" ? "bg-navy text-white" : tone === "ice" ? "bg-ice" : "bg-white";
  return (
    <section id={id} className={`${bg} scroll-mt-16 px-5 py-20 sm:py-24`}>
      <div className="mx-auto max-w-6xl">
        <p
          className={`text-xs font-semibold uppercase tracking-[0.14em] ${
            tone === "navy" ? "text-teal-bright" : "text-teal"
          }`}
        >
          {eyebrow}
        </p>
        <h2
          className={`mt-3 max-w-3xl font-serif text-3xl font-semibold tracking-tight sm:text-4xl ${
            tone === "navy" ? "text-white" : "text-navy"
          }`}
        >
          {title}
        </h2>
        {lead && (
          <p
            className={`mt-4 max-w-2xl text-[15px] leading-relaxed ${
              tone === "navy" ? "text-iceblue" : "text-slate2"
            }`}
          >
            {lead}
          </p>
        )}
        <div className="mt-12">{children}</div>
      </div>
    </section>
  );
}

function Spec({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-b border-line py-2.5 last:border-0">
      <dt className="text-[13px] text-slate2">{k}</dt>
      <dd className="text-right font-mono text-[13px] font-medium text-navy">{v}</dd>
    </div>
  );
}

/* ── page ───────────────────────────────────────────────────────────────── */

export default function LandingPage() {
  return (
    <>
      <LandingHeader />
      <main>
        {/* ── HERO ─────────────────────────────────────────────────────── */}
        <section className="relative overflow-hidden bg-navy px-5 pb-20 pt-32 sm:pt-40">
          <div
            aria-hidden
            className="pointer-events-none absolute -right-40 -top-40 h-[32rem] w-[32rem] rounded-full bg-teal/20 blur-3xl"
          />
          <div
            aria-hidden
            className="pointer-events-none absolute -bottom-52 -left-32 h-[28rem] w-[28rem] rounded-full bg-teal-bright/10 blur-3xl"
          />

          <div className="relative mx-auto max-w-6xl">
            <Badge
              variant="outline"
              className="border-teal-bright/40 bg-teal-bright/10 text-teal-bright"
            >
              VJTI AI Hackathon 2026 · Problem Statement 3
            </Badge>

            <h1 className="mt-6 max-w-4xl font-serif text-4xl font-semibold leading-[1.1] tracking-tight text-white sm:text-6xl">
              Answers an officer can actually cite.
            </h1>
            <p className="mt-3 font-serif text-xl text-teal-bright sm:text-2xl">
              शासन निर्णय सहाय्यक
            </p>

            <p className="mt-6 max-w-2xl text-base leading-relaxed text-iceblue sm:text-lg">
              A multilingual, source-grounded question-answering assistant over Maharashtra
              Government Resolutions. Ask in English or Marathi; it answers{" "}
              <strong className="font-semibold text-white">only</strong> from retrieved GRs, puts a
              citation on every claim, says &ldquo;not covered&rdquo; instead of guessing, and runs
              entirely on your own hardware.
            </p>

            <div className="mt-9 flex flex-wrap items-center gap-3">
              <Button asChild size="lg" className="bg-teal-bright text-navy hover:bg-teal-bright/90">
                <Link href="/ask">
                  Open the portal <ArrowRight size={17} className="ml-1.5" />
                </Link>
              </Button>
              <Button
                asChild
                size="lg"
                variant="outline"
                className="border-iceblue/30 bg-transparent text-white hover:bg-navy-700 hover:text-white"
              >
                <Link href="/browse">Browse the corpus</Link>
              </Button>
            </div>

            <div className="mt-10 flex flex-wrap gap-2.5">
              {[
                { icon: <Quote size={13} />, label: "Grounded — only from sources" },
                { icon: <Languages size={13} />, label: "Multilingual — English + मराठी" },
                { icon: <FileSearch size={13} />, label: "Explainable — every claim cited" },
                { icon: <Lock size={13} />, label: "Private — runs on-premise" },
              ].map((p) => (
                <span
                  key={p.label}
                  className="inline-flex items-center gap-1.5 rounded-full border border-iceblue/20 bg-white/5 px-3 py-1.5 text-[12.5px] text-iceblue"
                >
                  <span className="text-teal-bright">{p.icon}</span>
                  {p.label}
                </span>
              ))}
            </div>

            <Separator className="my-12 bg-navy-700" />
            <LiveCorpusStat />
          </div>
        </section>

        {/* ── PROBLEM ──────────────────────────────────────────────────── */}
        <Section
          id="problem"
          eyebrow="The problem"
          title="Officers must find trustworthy answers inside thousands of Government Resolutions."
          lead="Maharashtra publishes GRs, circulars, notifications and office orders continuously. Finding the one clause that governs a decision — and being sure it has not since been superseded — is slow, manual work."
        >
          <div className="grid gap-5 md:grid-cols-3">
            {[
              {
                icon: <Boxes className="text-teal" size={20} />,
                title: "Sheer volume",
                body: "Tens of thousands of GRs across ~33 departments. Locating one provision by hand takes an officer minutes to hours, and there is no way to be certain nothing newer overrides it.",
              },
              {
                icon: <ScanText className="text-teal" size={20} />,
                title: "Marathi, and scans",
                body: "Most GRs are in Marathi and many exist only as scanned image PDFs. Standard search tools index neither, and standard tokenizers split Devanagari at vowel marks.",
              },
              {
                icon: <TriangleAlert className="text-teal" size={20} />,
                title: "Generic AI invents",
                body: "A ChatGPT-style tool answers plausibly with no source. For governance that is worse than no answer at all — a confident, uncited, wrong figure is unusable and unauditable.",
              },
            ].map((c) => (
              <Card key={c.title} className="border-line shadow-sm transition-all hover:-translate-y-1 hover:shadow-md">
                <CardHeader className="pb-3">
                  <div className="mb-2.5 grid h-10 w-10 place-items-center rounded-lg bg-ice">
                    {c.icon}
                  </div>
                  <CardTitle className="text-base text-navy">{c.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-[13.5px] leading-relaxed text-slate2">{c.body}</p>
                </CardContent>
              </Card>
            ))}
          </div>

          <div className="mt-8 rounded-xl border-l-4 border-teal bg-ice px-6 py-5">
            <p className="text-[15px] leading-relaxed text-navy">
              <strong className="font-semibold">What is actually needed:</strong> answers pulled
              straight from authenticated documents, in the officer&rsquo;s own language, with the
              source shown — and no data leaving the department.
            </p>
          </div>
        </Section>

        {/* ── SOLUTION ─────────────────────────────────────────────────── */}
        <Section
          eyebrow="The approach"
          title="Retrieval-augmented generation, with refusal as a first-class outcome."
          lead="The model is never asked what it knows. It is handed a small set of passages retrieved from real GRs and constrained to answer from those alone — or to say it cannot."
          tone="ice"
        >
          <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
            {[
              {
                icon: <Quote size={19} />,
                title: "Grounded",
                body: "Answers come only from retrieved GR text. Citations are parsed back out of the reply and resolved against the blocks actually sent, so a citation to a document the model never saw is detected, not trusted.",
              },
              {
                icon: <Languages size={19} />,
                title: "Multilingual",
                body: "Ask in English, retrieve a Marathi GR. bge-m3 puts both languages in one shared vector space, so no translation step stands between the officer and the authoritative wording.",
              },
              {
                icon: <CircleSlash size={19} />,
                title: "Abstains",
                body: "Retrieval carries its own confidence. When nothing clears the score gate the prompt orders a refusal rather than leaving it to the model's judgement — measured, hedging was not enough.",
              },
              {
                icon: <Lock size={19} />,
                title: "On-premise",
                body: "Embeddings, ANN search, reranking, the knowledge graph, the audit log and the LLM all run locally. Proven, not asserted: a script blanks the API key and blocks every non-loopback socket, then answers.",
              },
            ].map((p) => (
              <Card key={p.title} className="border-line bg-white shadow-sm transition-all hover:-translate-y-1 hover:shadow-md">
                <CardHeader className="pb-3">
                  <div className="mb-2.5 grid h-10 w-10 place-items-center rounded-lg bg-navy text-teal-bright">
                    {p.icon}
                  </div>
                  <CardTitle className="text-base text-navy">{p.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-[13px] leading-relaxed text-slate2">{p.body}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </Section>

        {/* ── ARCHITECTURE ─────────────────────────────────────────────── */}
        <Section
          id="architecture"
          eyebrow="System design"
          title="Three stages, three stores, one machine."
          lead="Ingestion runs offline on the GPU. The knowledge store is two files that must agree. Query and answer runs online, with the GPU handed to the language model instead."
        >
          <Card className="border-line shadow-sm transition-all hover:shadow-md">
            <CardContent className="overflow-x-auto p-6">
              <div className="min-w-[720px]">
                <ArchitectureDiagram />
              </div>
            </CardContent>
          </Card>

          <div className="mt-8 grid gap-5 md:grid-cols-3">
            {[
              {
                icon: <Database size={18} />,
                title: "Vectors ≠ documents",
                body: "FAISS holds vectors and nothing else — no chunk text at all. SQLite holds the text, the metadata and the BM25 index. gr_chunks.faiss_id is the vector's position in the index, and that single integer is the entire join.",
              },
              {
                icon: <Layers size={18} />,
                title: "Why that split matters",
                body: "It is what keeps RAM flat as the corpus grows. The earlier design kept every chunk's text in a RAM sidecar — fine at 713 vectors, fatal at 74,000. Each half can also be swapped independently.",
              },
              {
                icon: <ServerCog size={18} />,
                title: "One GPU, two jobs",
                body: "6 GB of VRAM cannot hold the embedder, the reranker and the LLM at once. Ingestion gets the whole card; serving puts the embedder on CPU (0.1 s/query) and gives the GPU to the LLM and the cross-encoder.",
              },
            ].map((c) => (
              <div key={c.title} className="rounded-xl border border-line bg-ice/60 p-5 transition-all hover:bg-white hover:shadow-sm">
                <div className="mb-3 flex items-center gap-2.5 text-navy">
                  <span className="text-teal">{c.icon}</span>
                  <h3 className="text-[14.5px] font-semibold">{c.title}</h3>
                </div>
                <p className="text-[13px] leading-relaxed text-slate2">{c.body}</p>
              </div>
            ))}
          </div>
        </Section>

        {/* ── PIPELINES ────────────────────────────────────────────────── */}
        <Section
          id="pipelines"
          eyebrow="Pipelines"
          title="What happens to a document, and what happens to a question."
          tone="ice"
        >
          <Tabs defaultValue="ingest">
            <TabsList className="mb-6 bg-white">
              <TabsTrigger value="ingest">Ingestion (offline)</TabsTrigger>
              <TabsTrigger value="query">Query &amp; answer (online)</TabsTrigger>
            </TabsList>

            <TabsContent value="ingest" className="mt-0">
              <Card className="border-line bg-white shadow-sm">
                <CardContent className="overflow-x-auto p-6">
                  <div className="min-w-[760px]">
                    <IngestionPipelineDiagram />
                  </div>
                </CardContent>
              </Card>

              <div className="mt-6 grid gap-5 md:grid-cols-2">
                <Card className="border-line bg-white shadow-sm">
                  <CardHeader className="pb-3">
                    <CardTitle className="flex items-center gap-2 text-base text-navy">
                      <Gauge size={17} className="text-teal" /> fp16 was the whole ingestion win
                    </CardTitle>
                    <CardDescription>
                      Measured on 512 real Marathi chunks, not assumed.
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <dl>
                      <Spec k="float32, batch 8" v="9.8 chunks/s · 2.62 GB" />
                      <Spec k="float16, batch 8" v="34.6 chunks/s · 1.32 GB" />
                      <Spec k="float16, batch 16" v="36.6 chunks/s · 1.49 GB" />
                      <Spec k="fp32↔fp16 cosine agreement" v="min 0.99975" />
                    </dl>
                    <p className="mt-4 text-[13px] leading-relaxed text-slate2">
                      The instinct was &ldquo;raise the batch size&rdquo;; the measurement said the
                      card was already compute-bound at 100%, and the real lever was{" "}
                      <em>precision</em>. Batch 32 in fp32 simply OOMs. Full corpus: ~25 minutes.
                    </p>
                  </CardContent>
                </Card>

                <Card className="border-line bg-white shadow-sm">
                  <CardHeader className="pb-3">
                    <CardTitle className="flex items-center gap-2 text-base text-navy">
                      <Table2 size={17} className="text-teal" /> Tables become sentences
                    </CardTitle>
                    <CardDescription>
                      A fee grid embeds badly; a sentence embeds well.
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <p className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-slate2">
                      Before — raw table in the GR
                    </p>
                    <pre className="overflow-x-auto rounded-lg bg-navy px-3.5 py-3 font-mono text-[11.5px] leading-relaxed text-iceblue">
{`प्रवर्ग            | वार्षिक शुल्क (रु.)
खुला / Open        | 12000
इतर मागासवर्ग/OBC | 6000
अनुसूचित जाती/SC   | 0`}
                    </pre>
                    <p className="mb-2 mt-4 text-[11px] font-semibold uppercase tracking-wide text-slate2">
                      After — what actually gets embedded
                    </p>
                    <p className="rounded-lg border border-teal/30 bg-teal/5 px-3.5 py-3 text-[12.5px] leading-relaxed text-navy">
                      &ldquo;प्रवर्ग (Category) is इतर मागासवर्ग / OBC, वार्षिक शुल्क (रु.) is
                      6000.&rdquo;
                    </p>
                    <p className="mt-4 text-[13px] leading-relaxed text-slate2">
                      Numbers are preserved exactly, so &ldquo;What is the OBC diploma fee?&rdquo;
                      retrieves this row and returns ₹6000 rather than a paraphrase.
                    </p>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="query" className="mt-0">
              <Card className="border-line bg-white shadow-sm">
                <CardContent className="overflow-x-auto p-6">
                  <div className="min-w-[720px]">
                    <RetrievalPipelineDiagram />
                  </div>
                </CardContent>
              </Card>

              <div className="mt-6 grid gap-5 md:grid-cols-3">
                {[
                  {
                    t: "Why hybrid",
                    b: "Dense search finds meaning and survives paraphrase; BM25 finds exact strings like a GR number that an embedding blurs. Reciprocal rank fusion combines them on rank, so the two incomparable score scales never have to be normalised.",
                  },
                  {
                    t: "Why group by GR",
                    b: "Chunks are the right unit to search and the wrong unit to cite. Several chunks of one GR would otherwise crowd out every other document, so hits are collapsed per GR before reranking — the officer cites an order, not a passage.",
                  },
                  {
                    t: "Why rerank",
                    b: "The bi-encoder embeds the query and the passage separately, so it is fast but approximate. The cross-encoder reads the pair together and is far more precise — 15 pairs per query, which is why it is the one model that must stay on the GPU.",
                  },
                ].map((c) => (
                  <div key={c.t} className="rounded-xl border border-line bg-white p-5 transition-all hover:shadow-md hover:-translate-y-1">
                    <h3 className="mb-2.5 text-[14.5px] font-semibold text-navy">{c.t}</h3>
                    <p className="text-[13px] leading-relaxed text-slate2">{c.b}</p>
                  </div>
                ))}
              </div>
            </TabsContent>
          </Tabs>
        </Section>

        {/* ── RETRIEVAL INTERNALS ──────────────────────────────────────── */}
        <Section
          id="retrieval"
          eyebrow="Under the hood"
          title="The parameters, and how each one was chosen."
          lead="Every number here was calibrated against a 23-question gold set on the real corpus, not copied from a tutorial."
        >
          <div className="grid gap-6 lg:grid-cols-2">
            <Card className="border-line shadow-sm">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base text-navy">
                  <Network size={17} className="text-teal" /> Vector index
                </CardTitle>
              </CardHeader>
              <CardContent>
                <dl>
                  <Spec k="Index" v="FAISS IndexHNSWFlat" />
                  <Spec k="Dimensions" v="1024 (bge-m3)" />
                  <Spec k="Metric" v="cosine / inner product" />
                  <Spec k="M · efConstruction" v="32 · 200" />
                  <Spec k="efSearch" v="512" />
                  <Spec k="recall@60 vs brute force" v="0.986 @ 0.53 ms" />
                  <Spec k="Metadata filtering" v="native IDSelector" />
                </dl>
                <p className="mt-4 text-[13px] leading-relaxed text-slate2">
                  Filters are pushed <em>into</em> the search rather than applied after it. A
                  post-filter silently loses recall whenever the entire top-k belongs to an excluded
                  department — the results look fine and are quietly wrong.
                </p>
              </CardContent>
            </Card>

            <Card className="border-line shadow-sm">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base text-navy">
                  <Cpu size={17} className="text-teal" /> Models &amp; placement
                </CardTitle>
              </CardHeader>
              <CardContent>
                <dl>
                  <Spec k="Embeddings" v="BAAI/bge-m3" />
                  <Spec k="Reranker" v="bge-reranker-v2-m3" />
                  <Spec k="Generation" v="qwen2.5:3b-instruct-q8_0" />
                  <Spec k="Inference server" v="Ollama, localhost" />
                  <Spec k="Context window" v="8192 tokens" />
                  <Spec k="Embedder (serving)" v="CPU — 0.1 s/query" />
                  <Spec k="Reranker" v="GPU fp16 — 0.33 s" />
                </dl>
                <p className="mt-4 text-[13px] leading-relaxed text-slate2">
                  The reranker on CPU measured <strong className="text-navy">27.6 s</strong> per
                  query against 0.33 s on GPU — an 80× gap that was, on its own, the entire latency
                  problem. A bi-encoder and a cross-encoder have opposite cost profiles, so one
                  device switch for both models was the wrong abstraction.
                </p>
              </CardContent>
            </Card>
          </div>

          <Accordion type="single" collapsible className="mt-8">
            <AccordionItem value="a">
              <AccordionTrigger className="text-[14.5px] text-navy">
                Devanagari broke the tokenizer, twice
              </AccordionTrigger>
              <AccordionContent className="text-[13.5px] leading-relaxed text-slate2">
                A `\w`-based BM25 tokenizer splits Marathi at vowel marks — शासन becomes श and सन —
                and an ASCII-only pattern drops the language entirely. The working pattern is
                <code className="mx-1 rounded bg-ice px-1.5 py-0.5 font-mono text-[12px] text-navy">
                  [a-z0-9ऀ-ॿ]+
                </code>
                . Separately, token <em>density</em> is a property of the tokenizer, not of the
                language: Llama byte-BPEs Devanagari at ~2 tokens/char, qwen2.5 at 1.09. Applying
                Llama&rsquo;s rate to qwen over-counted every chunk by 1.7×, so the prompt builder
                sent one context block and silently dropped ten.
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="b">
              <AccordionTrigger className="text-[14.5px] text-navy">
                Why BM25 had to move into SQLite
              </AccordionTrigger>
              <AccordionContent className="text-[13.5px] leading-relaxed text-slate2">
                The usual `rank_bm25` library keeps a tokenized copy of the whole corpus in RAM — a
                dict per document. That is fine at 713 chunks and several gigabytes at 74,000. FTS5
                does the same BM25 ranking off disk, inside the database that already holds the
                text. The trap: FTS5&rsquo;s MATCH syntax treats <code>-</code>, <code>/</code>,{" "}
                <code>.</code> and <code>OR</code> as operators, so a raw question — or a GR number
                like संकीर्ण-२०२३/प्र.क्र.४५ — raises a syntax error unless it is tokenized and
                quoted first.
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="c">
              <AccordionTrigger className="text-[14.5px] text-navy">
                Prompt length is not free on a small model
              </AccordionTrigger>
              <AccordionContent className="text-[13.5px] leading-relaxed text-slate2">
                Given identical retrieved context, the full production prompt produced{" "}
                <strong className="text-navy">4 completion tokens</strong> — the literal string
                &ldquo;[1]&rdquo;. A two-line prompt produced 99 tokens and a correct, cited answer.
                Roughly 350 words of mostly-prohibitive rules collapse a 3B model into emitting only
                the token it is sure of. Six variants were measured and each traded one failure for
                another: demanding citations harder stopped it abstaining; adding a worked example
                made it emit the <em>example</em> as a cited answer. The system now ships two
                prompts — a long one for large models, a compact one selected automatically for the
                local model.
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </Section>

        {/* ── GRAPH ────────────────────────────────────────────────────── */}
        <Section
          id="graph"
          eyebrow="Knowledge graph"
          title="Which order is actually in force?"
          lead="A GR is rarely the last word. It amends, supersedes and cites others. Nodes are GRs and edges are parsed deterministically from their reference lines — this is a domain knowledge graph, not Graph RAG: the edges provide provenance and conflict warnings, they do not drive retrieval."
          tone="ice"
        >
          <Card className="border-line bg-white shadow-sm">
            <CardContent className="overflow-x-auto p-6">
              <div className="min-w-[700px]">
                <GraphDiagram />
              </div>
            </CardContent>
          </Card>

          <div className="mt-8 grid gap-6 lg:grid-cols-2">
            <Card className="border-line bg-white shadow-sm">
              <CardHeader>
                <CardTitle className="text-base text-navy">Measured on the corpus</CardTitle>
              </CardHeader>
              <CardContent>
                <dl>
                  <Spec k="Edges built" v="60,420" />
                  <Spec k="Resolved to a held document" v="5,753 (9.5%)" />
                  <Spec k="…of those, by cited date" v="1,771" />
                  <Spec k="Documents with ≥1 resolved edge" v="3,904" />
                  <Spec k="Dangling (order not held)" v="48,044" />
                  <Spec k="References carrying a date" v="43,738 (72%)" />
                </dl>
              </CardContent>
            </Card>

            <Card className="border-line bg-white shadow-sm">
              <CardHeader>
                <CardTitle className="text-base text-navy">
                  The hard part is resolution, not traversal
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3.5 text-[13.5px] leading-relaxed text-slate2">
                <p>
                  Traversing a few thousand edges is trivial. Deciding whether
                  संकीर्ण-२०२३/प्र.क्र.४५/तांशि-४ printed in one OCR&rsquo;d GR is the same order as a
                  number parsed from another is not — they differ by spacing, by Devanagari versus
                  ASCII digits, and by punctuation the scanner dropped.
                </p>
                <p>
                  The first build resolved <strong className="text-navy">2%</strong> of references.
                  The fault was upstream: a GR number was being matched as a whitespace-bounded
                  token, but real numbers <em>contain</em> spaces. Parsing reference lines the way
                  they are actually written, and using the cited date to separate documents sharing
                  a number, took it to <strong className="text-navy">9.5%</strong> — and the
                  remaining gap is honest, since only ~21% of unresolved references even name one of
                  the six departments held.
                </p>
                <p className="rounded-lg border-l-2 border-teal bg-ice px-4 py-3 text-navy">
                  The slash is deliberately preserved during normalisation. It is structural in a GR
                  number, and over-normalising would <em>fabricate</em> supersessions — far worse
                  than missing one.
                </p>
              </CardContent>
            </Card>
          </div>
        </Section>

        {/* ── SECURITY ─────────────────────────────────────────────────── */}
        <Section
          id="security"
          eyebrow="Security &amp; governance"
          title="A government portal, not an open demo."
          lead="Four roles from the SRS, least privilege, and an audit trail designed around what it deliberately does not record."
        >
          <div className="grid gap-6 lg:grid-cols-5">
            <Card className="border-line shadow-sm lg:col-span-3">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base text-navy">
                  <ShieldCheck size={17} className="text-teal" /> Role-based access
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-[38%]">Role</TableHead>
                      <TableHead>Can</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {[
                      ["Desk Officer", "Ask, browse, summarize, submit feedback"],
                      ["Legal Translator", "+ force the answer language"],
                      ["Reviewer", "+ read other officers' searches"],
                      ["IT Admin", "+ /admin, user management, audit export"],
                    ].map(([r, c]) => (
                      <TableRow key={r}>
                        <TableCell className="font-medium text-navy">{r}</TableCell>
                        <TableCell className="text-slate2">{c}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
                <p className="mt-4 text-[13px] leading-relaxed text-slate2">
                  Enforced by a FastAPI dependency on the server, never only in the UI. Sessions are
                  JWTs held in <code className="rounded bg-ice px-1 py-0.5 font-mono text-[12px]">sessionStorage</code>{" "}
                  rather than localStorage, which narrows what an XSS bug can reach.
                </p>
              </CardContent>
            </Card>

            <div className="space-y-5 lg:col-span-2">
              <Card className="border-line shadow-sm">
                <CardHeader className="pb-3">
                  <CardTitle className="flex items-center gap-2 text-base text-navy">
                    <Fingerprint size={17} className="text-teal" /> What the audit log stores
                  </CardTitle>
                </CardHeader>
                <CardContent className="text-[13px] leading-relaxed text-slate2">
                  <p className="mb-3">
                    Who asked, when, from which IP, the question, and the GR numbers cited.
                  </p>
                  <p className="rounded-lg border-l-2 border-teal bg-ice px-4 py-3 text-navy">
                    Deliberately <strong>not</strong> the answer text or document bodies. An audit
                    trail must prove who asked what — not become a second, uncontrolled copy of the
                    corpus. Knowing what not to log is the design decision.
                  </p>
                </CardContent>
              </Card>

              <Card className="border-line shadow-sm">
                <CardHeader className="pb-3">
                  <CardTitle className="flex items-center gap-2 text-base text-navy">
                    <BookOpenCheck size={17} className="text-teal" /> Isolation
                  </CardTitle>
                </CardHeader>
                <CardContent className="text-[13px] leading-relaxed text-slate2">
                  Users and the audit log live in the portal database, never in the corpus database.
                  A full corpus rebuild therefore cannot touch the audit trail — a property worth
                  being able to state out loud for a government system. Per-user token-bucket rate
                  limiting keeps one officer from exhausting a shared local model.
                </CardContent>
              </Card>
            </div>
          </div>
        </Section>

        {/* ── RESULTS ──────────────────────────────────────────────────── */}
        <Section
          id="results"
          eyebrow="Measured"
          title="Numbers from the gold set, including the ones that got worse."
          lead="A 23-question gold set (English + Marathi, exact-number questions, and out-of-corpus questions that must be refused). Scaling the corpus 90× degraded ranking and latency; groundedness and abstention held."
          tone="navy"
        >
          <Card className="border-navy-700 bg-navy-700/40">
            <CardContent className="overflow-x-auto p-0">
              <Table>
                <TableHeader>
                  <TableRow className="border-navy-700 hover:bg-transparent">
                    <TableHead className="text-iceblue">Metric</TableHead>
                    <TableHead className="text-iceblue">196-GR index</TableHead>
                    <TableHead className="text-iceblue">18,078-GR corpus</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {[
                    ["hit@1 / hit@5", "19/20 · 20/20", "12/20 · 14/20"],
                    ["Mean reciprocal rank", "0.975", "0.642"],
                    ["Answer carries a citation", "19–20/20", "15/20"],
                    ["GROUNDED (no phantom citation)", "20/20", "20/20"],
                    ["Answer correct", "19–20/20", "11/20"],
                    ["Degenerate (bare “[1]”)", "0/20", "0/20"],
                    ["Abstains when out of corpus", "2/3", "2/3"],
                  ].map(([m, a, b]) => (
                    <TableRow key={m} className="border-navy-700 hover:bg-navy-700/30">
                      <TableCell className="font-medium text-white">{m}</TableCell>
                      <TableCell className="font-mono text-iceblue">{a}</TableCell>
                      <TableCell className="font-mono text-iceblue">{b}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>

          <div className="mt-8 grid gap-5 md:grid-cols-2">
            <div className="rounded-xl border border-navy-700 bg-navy-700/30 p-5 transition-colors hover:bg-navy-700/50">
              <h3 className="mb-2.5 text-[14.5px] font-semibold text-teal-bright">
                What survived scaling — and matters most
              </h3>
              <p className="text-[13px] leading-relaxed text-iceblue">
                GROUNDED stayed 20/20. Not one answer cited a document that was not in its context,
                and the abstention machinery held unchanged. The system became{" "}
                <em>less accurate</em>, never <em>less honest</em> — which is the correct direction
                for a tool whose failure mode must never be a confident, cited, wrong figure.
              </p>
            </div>
            <div className="rounded-xl border border-navy-700 bg-navy-700/30 p-5 transition-colors hover:bg-navy-700/50">
              <h3 className="mb-2.5 text-[14.5px] font-semibold text-teal-bright">
                What regressed, stated plainly
              </h3>
              <p className="text-[13px] leading-relaxed text-iceblue">
                Ranking. At 18,000 documents there are far more plausible-but-wrong passages than at
                196, and the rerank threshold is still the one calibrated for the small corpus.
                Three of the correct-answer failures target synthetic fixtures absent from this
                index, so the fair figure is ~11/17. Recalibration is the next measured step, not a
                claim already made.
              </p>
            </div>
          </div>

          <p className="mt-6 text-[12px] leading-relaxed text-iceblue/70">
            Ranges are run-to-run variance at temperature 0.2 over 20 questions — treat a difference
            of 1 as noise. These figures predate the current prompt and table-chunking changes and
            are due to be re-measured; they are shown as last recorded rather than restated as
            current.
          </p>
        </Section>

        {/* ── SRS COMPLIANCE ───────────────────────────────────────────── */}
        <Section
          eyebrow="Requirements"
          title="Mapped to the SRS, requirement by requirement."
          tone="ice"
        >
          <div className="grid gap-4 md:grid-cols-2">
            {[
              ["3.1 Knowledge repository", "OCR for scanned Marathi/English, per-chunk metadata and 1024-d embeddings in a vector index."],
              ["3.2 Semantic search", "Dense + BM25 hybrid, relevance-ranked, with each source and its score shown."],
              ["3.3 Question answering", "RAG restricted to retrieved GRs, a citation on every claim, conflicting GRs flagged, refusal when unsupported."],
              ["3.4 Multilingual assistant", "English and Marathi in and out, cross-lingual retrieval, official terminology preserved, language switchable mid-conversation."],
              ["3.5 Officer assistance", "Explain simply, summarize, compare two GRs, recommend related orders, identify superseded and referenced GRs."],
              ["3.7 Interface & workflow", "Secure portal with login and roles, conversational chat, persisted history, view/download the referenced GR, feedback on answers."],
              ["NFR — performance", "Sub-10-second target; the reranker device split took end-to-end p50 from 38 s to 2.6 s on the small corpus."],
              ["NFR — security", "On-premise deployment, proven offline by a script that blocks every non-loopback socket."],
            ].map(([k, v]) => (
              <div key={k} className="flex gap-3.5 rounded-xl border border-line bg-white p-5">
                <div className="mt-0.5 shrink-0">
                  <span className="grid h-6 w-6 place-items-center rounded-full bg-teal/10 text-teal">
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" aria-hidden>
                      <path
                        d="M20 6L9 17l-5-5"
                        stroke="currentColor"
                        strokeWidth="3"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                  </span>
                </div>
                <div>
                  <h3 className="text-[14px] font-semibold text-navy">{k}</h3>
                  <p className="mt-1 text-[13px] leading-relaxed text-slate2">{v}</p>
                </div>
              </div>
            ))}
          </div>
        </Section>

        {/* ── STACK ────────────────────────────────────────────────────── */}
        <Section
          id="stack"
          eyebrow="Technology"
          title="Everything open-source, everything local."
        >
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {[
              ["BAAI/bge-m3", "Multilingual embeddings, 1024-d, 100+ languages in one shared space"],
              ["bge-reranker-v2-m3", "Multilingual cross-encoder, loaded in fp16 on GPU"],
              ["FAISS IndexHNSWFlat", "In-process approximate nearest neighbour, no extra server"],
              ["SQLite + FTS5", "Documents, chunks, BM25, the graph, and the audit log"],
              ["Ollama + Qwen2.5-3B", "Local generation, OpenAI-compatible endpoint, 8192 ctx"],
              ["Tesseract OCR", "Scanned Marathi and English GRs (mar + hin + eng)"],
              ["FastAPI", "Slim on-prem API — no cloud dependency anywhere in the path"],
              ["Next.js 14 + Tailwind", "Officer portal, shadcn/ui components"],
              ["PyMuPDF · pdfplumber", "PDF text and table extraction"],
            ].map(([t, d]) => (
              <div
                key={t}
                className="rounded-xl border border-line bg-white p-5 transition-colors hover:border-teal/50"
              >
                <h3 className="font-mono text-[13.5px] font-semibold text-navy">{t}</h3>
                <p className="mt-1.5 text-[13px] leading-relaxed text-slate2">{d}</p>
              </div>
            ))}
          </div>

          <div className="mt-10 rounded-2xl border border-line bg-ice px-6 py-8 sm:px-10">
            <h3 className="font-serif text-2xl font-semibold text-navy">
              Constraints shaped the design
            </h3>
            <p className="mt-3 max-w-3xl text-[14px] leading-relaxed text-slate2">
              A 6 GB laptop GPU and a nearly full disk are why this runs a 3B model beside a 568M
              cross-encoder, why the vector index lives on a separate partition, and why the
              embedder is on CPU at serving time. Each of those was a measurement, not a preference
              — and the pipeline is unchanged for the full ~100k-GR corpus, which is a matter of
              GPU-hours rather than architecture.
            </p>
          </div>
        </Section>

        {/* ── CTA ──────────────────────────────────────────────────────── */}
        <section className="bg-navy px-5 py-20">
          <div className="mx-auto max-w-3xl text-center">
            <h2 className="font-serif text-3xl font-semibold tracking-tight text-white sm:text-4xl">
              Grounded. Multilingual. Explainable. Private.
            </h2>
            <p className="mt-4 text-[15px] leading-relaxed text-iceblue">
              An AI assistant government officers can actually trust — because it shows its sources
              and admits what it does not know.
            </p>
            <div className="mt-8 flex flex-wrap justify-center gap-3">
              <Button asChild size="lg" className="bg-teal-bright text-navy hover:bg-teal-bright/90">
                <Link href="/ask">
                  Ask a question <ArrowRight size={17} className="ml-1.5" />
                </Link>
              </Button>
              <Button
                asChild
                size="lg"
                variant="outline"
                className="border-iceblue/30 bg-transparent text-white hover:bg-navy-700 hover:text-white"
              >
                <Link href="/browse">Browse 18,080 GRs</Link>
              </Button>
            </div>
          </div>
        </section>
      </main>
      <LandingFooter />
    </>
  );
}
