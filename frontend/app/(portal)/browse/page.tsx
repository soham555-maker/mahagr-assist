"use client";

import { useEffect, useState } from "react";
import { GitCompareArrows, Search, ArrowRightLeft, Link2, ShieldCheck, FileText, Download } from "lucide-react";
import {
  api,
  downloadDocument,
  type DocMeta,
  type Supersession,
  type Related,
  type AnswerResult,
} from "@/lib/api";
import { CorpusStat, EmptyHint, LangToggle, SourceCard, Spinner, useCorpusStats } from "@/components/ui";
import { GraphPanel } from "@/components/graph";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

const PAGE = 50;

export default function BrowsePage() {
  const [docs, setDocs] = useState<DocMeta[]>([]);
  const [query, setQuery] = useState("");
  const [loadErr, setLoadErr] = useState<string | null>(null);
  // Opened from a citation's "open" link (?doc=<id>) so a source in an answer
  // leads somewhere real — SRS FR 3.7.4.
  const [selected, setSelected] = useState<string | null>(null);
  useEffect(() => {
    const doc = new URLSearchParams(window.location.search).get("doc");
    if (doc) setSelected(doc);
  }, []);
  const [depts, setDepts] = useState<string[]>([]);
  const [listLoading, setListLoading] = useState(false);
  const [atEnd, setAtEnd] = useState(false);
  const stats = useCorpusStats();

  const [compareMode, setCompareMode] = useState(false);
  const [picks, setPicks] = useState<string[]>([]);
  const [language, setLanguage] = useState("auto");
  const [compareResult, setCompareResult] = useState<AnswerResult | null>(null);
  const [comparing, setComparing] = useState(false);

  // Search runs on the SERVER, not over a client-side copy of the corpus: at
  // ~18,000 GRs, fetching every document to filter it in the browser would be a
  // multi-megabyte page load before the officer has typed anything.
  // Debounced so a query isn't fired on every keystroke.
  useEffect(() => {
    const t = setTimeout(() => {
      setListLoading(true);
      api
        .documents({ q: query.trim() || undefined, department: depts.length ? depts : undefined, limit: PAGE })
        .then((d) => {
          setDocs(d);
          setLoadErr(null);
          setAtEnd(d.length < PAGE);
        })
        .catch((e) => setLoadErr((e as Error).message))
        .finally(() => setListLoading(false));
    }, 250);
    return () => clearTimeout(t);
  }, [query, depts]);

  async function loadMore() {
    setListLoading(true);
    try {
      const more = await api.documents({
        q: query.trim() || undefined,
        department: depts.length ? depts : undefined,
        limit: PAGE,
        offset: docs.length,
      });
      setDocs((d) => [...d, ...more]);
      setAtEnd(more.length < PAGE);
    } catch (e) {
      setLoadErr((e as Error).message);
    } finally {
      setListLoading(false);
    }
  }

  function rowClick(d: DocMeta) {
    if (compareMode) {
      setPicks((p) =>
        p.includes(d.doc) ? p.filter((x) => x !== d.doc) : p.length < 2 ? [...p, d.doc] : p,
      );
    } else {
      setSelected(d.doc);
      setCompareResult(null);
    }
  }

  async function runCompare() {
    if (picks.length !== 2) return;
    setComparing(true);
    setCompareResult(null);
    try {
      setCompareResult(await api.compare(picks[0], picks[1], language));
    } catch (e) {
      setCompareResult({ answer: `Compare failed: ${(e as Error).message}`, sources: [], phantom_citations: [] });
    } finally {
      setComparing(false);
    }
  }

  return (
    <main className="mx-auto grid max-w-6xl gap-5 px-4 py-6 md:grid-cols-[minmax(300px,380px)_1fr]">
      {/* ---- left: list ---- */}
      <section className="flex min-h-0 flex-col">
        <div className="mb-3 flex items-center gap-2">
          <div className="relative flex-1">
            <Search
              size={16}
              className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-slate2"
              aria-hidden
            />
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search by title or GR number…"
              aria-label="Search Government Resolutions"
              className="bg-white pl-9"
            />
          </div>
          <Button
            onClick={() => {
              setCompareMode((v) => !v);
              setPicks([]);
              setCompareResult(null);
            }}
            aria-pressed={compareMode}
            variant={compareMode ? "default" : "outline"}
            className={cn(
              compareMode
                ? "bg-teal text-white hover:bg-teal/90"
                : "border-line bg-white text-slate2 hover:text-ink",
            )}
          >
            <GitCompareArrows size={16} className="mr-1.5" /> Compare
          </Button>
        </div>

        {(stats?.departments.length || 0) > 1 && (
          <div className="mb-3 flex flex-wrap gap-1.5">
            <button
              onClick={() => setDepts([])}
              aria-pressed={depts.length === 0}
              className="cursor-pointer rounded-full focus-visible:outline-none"
            >
              <Badge
                variant="outline"
                className={cn(
                  "font-normal transition-colors",
                  depts.length === 0
                    ? "border-teal bg-ice text-teal"
                    : "border-line bg-white text-slate2 hover:text-ink",
                )}
              >
                All departments
              </Badge>
            </button>
            {stats!.departments.map((d) => {
              const on = depts.includes(d.name);
              return (
                <button
                  key={d.name}
                  onClick={() => setDepts((p) => (on ? p.filter((x) => x !== d.name) : [...p, d.name]))}
                  aria-pressed={on}
                  title={`${d.documents.toLocaleString()} GRs`}
                  className="cursor-pointer rounded-full focus-visible:outline-none"
                >
                  <Badge
                    variant="outline"
                    className={cn(
                      "font-normal transition-colors",
                      on
                        ? "border-teal bg-ice text-teal"
                        : "border-line bg-white text-slate2 hover:text-ink",
                    )}
                  >
                    {d.name.replace(/ Department$/, "")}
                  </Badge>
                </button>
              );
            })}
          </div>
        )}

        <div className="mb-2">
          <CorpusStat stats={stats} />
        </div>

        {loadErr && (
          <Alert variant="destructive" className="mb-2">
            <AlertDescription>
              Couldn’t load documents: {loadErr}. Is the backend running on port 8000?
            </AlertDescription>
          </Alert>
        )}

        <ul className="scroll-thin -mx-1 flex-1 space-y-1.5 overflow-y-auto px-1" style={{ maxHeight: "calc(100dvh - 160px)" }}>
          {docs.map((d) => {
            const picked = picks.includes(d.doc);
            const active = selected === d.doc && !compareMode;
            return (
              <li key={d.doc}>
                <button
                  onClick={() => rowClick(d)}
                  className={`w-full cursor-pointer rounded-lg border p-3 text-left transition-colors duration-200 ${
                    active || picked ? "border-teal bg-ice" : "border-line bg-white hover:border-teal"
                  }`}
                >
                  <div className="flex items-start gap-2">
                    {compareMode && (
                      <span
                        className={`mt-0.5 grid h-4 w-4 shrink-0 place-items-center rounded border ${
                          picked ? "border-teal bg-teal text-white" : "border-slate2"
                        }`}
                        aria-hidden
                      >
                        {picked && <span className="text-[10px]">✓</span>}
                      </span>
                    )}
                    <div className="min-w-0">
                      <p className="line-clamp-2 text-sm font-medium leading-snug text-ink">
                        {d.title || d.doc}
                      </p>
                      <p className="mt-1 truncate text-xs text-slate2">
                        {d.gr_number && <span className="text-teal">{d.gr_number}</span>}
                        {d.date && <span> · {d.date}</span>}
                      </p>
                    </div>
                  </div>
                </button>
              </li>
            );
          })}
          {docs.length === 0 && !loadErr && (
            <li className="p-3 text-sm text-slate2">
              {listLoading ? "Loading GRs…" : "No GRs match that search."}
            </li>
          )}
          {docs.length > 0 && !atEnd && (
            <li className="pt-1">
              <Button
                onClick={loadMore}
                disabled={listLoading}
                variant="outline"
                className="w-full border-line bg-white text-slate2 hover:border-teal hover:text-teal"
              >
                {listLoading ? "Loading…" : `Load ${PAGE} more`}
              </Button>
            </li>
          )}
        </ul>

        {compareMode && (
          <div className="mt-3 flex items-center gap-2 rounded-lg border border-line bg-white p-2">
            <LangToggle value={language} onChange={setLanguage} />
            <Button
              onClick={runCompare}
              disabled={picks.length !== 2 || comparing}
              className="ml-auto bg-navy text-white hover:bg-teal disabled:opacity-40"
            >
              <ArrowRightLeft size={15} className="mr-1.5" /> Compare&nbsp;these&nbsp;two (
              {picks.length}/2)
            </Button>
          </div>
        )}
      </section>

      {/* ---- right: detail / compare ---- */}
      <section className="min-w-0">
        {compareMode ? (
          comparing ? (
            <Panel><Spinner label="Comparing the two resolutions…" /></Panel>
          ) : compareResult ? (
            <AnswerView title="Comparison" result={compareResult} />
          ) : (
            <EmptyHint icon={<GitCompareArrows size={22} />}>
              Pick two GRs from the list, then “Compare these two”. Great with a GR and the one that supersedes it.
            </EmptyHint>
          )
        ) : selected ? (
          <DocDetail id={selected} onSelect={setSelected} />
        ) : (
          <EmptyHint>Select a Government Resolution to read it and see what it supersedes and relates to.</EmptyHint>
        )}
      </section>
    </main>
  );
}

function Panel({ children }: { children: React.ReactNode }) {
  return (
    <Card className="border-line shadow-sm">
      <CardContent className="p-5">{children}</CardContent>
    </Card>
  );
}

function AnswerView({ title, result }: { title: string; result: AnswerResult }) {
  return (
    <Panel>
      <h2 className="font-serif text-xl font-semibold text-ink">{title}</h2>
      <div className="prose-answer mt-3 text-[15px] text-ink">{result.answer}</div>
      {result.sources.length > 0 && (
        <div className="mt-4 border-t border-line pt-3">
          <p className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-slate2">
            <ShieldCheck size={14} className="text-teal" /> Sources
          </p>
          <ul className="grid gap-2">
            {result.sources.map((s) => (
              <SourceCard key={s.n} s={s} />
            ))}
          </ul>
        </div>
      )}
    </Panel>
  );
}

function DocDetail({ id, onSelect }: { id: string; onSelect?: (id: string) => void }) {
  const [text, setText] = useState<string | null>(null);
  const [meta, setMeta] = useState<{ gr_number: string | null; date: string | null; title: string | null } | null>(null);
  const [sup, setSup] = useState<Supersession | null>(null);
  const [rel, setRel] = useState<Related[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [summary, setSummary] = useState<AnswerResult | null>(null);
  const [summing, setSumming] = useState(false);

  async function summarize() {
    setSumming(true);
    setSummary(null);
    try {
      setSummary(await api.summarize(id, "auto"));
    } catch (e) {
      setSummary({ answer: `Summarize failed: ${(e as Error).message}`, sources: [], phantom_citations: [] });
    } finally {
      setSumming(false);
    }
  }

  useEffect(() => {
    setText(null); setSup(null); setRel([]); setErr(null); setSummary(null);
    api.documentText(id).then((d) => { setText(d.text); setMeta(d); }).catch((e) => setErr((e as Error).message));
    api.supersede(id).then(setSup).catch(() => {});
    api.related(id).then((r) => setRel(r.related)).catch(() => {});
  }, [id]);

  if (err)
    return (
      <Alert variant="destructive">
        <AlertDescription>{err}</AlertDescription>
      </Alert>
    );
  if (text === null) return <Panel><Spinner label="Loading document…" /></Panel>;

  return (
    <div className="space-y-4">
      <Panel>
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <h2 className="font-serif text-xl font-semibold leading-snug text-ink">{meta?.title || id}</h2>
            <p className="mt-1.5 flex flex-wrap gap-x-3 text-sm text-slate2">
              {meta?.gr_number && <span className="font-medium text-teal">{meta.gr_number}</span>}
              {meta?.date && <span>{meta.date}</span>}
            </p>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <Button
              onClick={() => downloadDocument(id).catch(() => {})}
              title="Download the full text of this GR"
              variant="outline"
              size="sm"
              className="border-line bg-white text-slate2 hover:border-teal hover:text-teal"
            >
              <Download size={15} className="mr-1.5" /> Download
            </Button>
            <Button
              onClick={summarize}
              disabled={summing}
              variant="outline"
              size="sm"
              className="border-line bg-white text-slate2 hover:border-teal hover:text-teal"
            >
              <FileText size={15} className="mr-1.5" /> Summarize
            </Button>
          </div>
        </div>
      </Panel>

      {summing ? (
        <Panel><Spinner label="Summarizing this GR…" /></Panel>
      ) : summary ? (
        <AnswerView title="Summary" result={summary} />
      ) : null}

      {/* The knowledge graph (PLAN Phase 3). Renders nothing when the graph
          has not been built, so the page degrades rather than erroring. */}
      <GraphPanel docId={id} onSelect={onSelect} />

      {sup && sup.found && (
        <Panel>
          <h3 className="flex items-center gap-1.5 text-sm font-semibold text-ink">
            <ArrowRightLeft size={15} className="text-teal" /> Supersession
          </h3>
          <div className="mt-2 space-y-2 text-sm">
            {sup.superseded_by.length > 0 ? (
              sup.superseded_by.map((s) => (
                <p key={s.gr_number} className="rounded-md bg-amber-50 px-3 py-2 text-amber-900">
                  Superseded by <span className="font-medium">{s.gr_number}</span> ({s.date}) — {s.title}
                </p>
              ))
            ) : (
              <p className="text-slate2">Not superseded by any GR in this corpus.</p>
            )}
            {sup.cites.length > 0 && (
              <div>
                <p className="text-xs font-semibold uppercase tracking-wide text-slate2">Cites / builds on</p>
                <ul className="mt-1 space-y-1">
                  {sup.cites.map((c) => (
                    <li key={c.gr_number} className="text-ink">
                      {c.gr_number}{" "}
                      <span className="text-xs text-slate2">
                        {c.in_corpus ? "· in corpus" : "· not in corpus"}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </Panel>
      )}

      {rel.length > 0 && (
        <Panel>
          <h3 className="flex items-center gap-1.5 text-sm font-semibold text-ink">
            <Link2 size={15} className="text-teal" /> Related resolutions
          </h3>
          <ul className="mt-2 space-y-1.5">
            {rel.map((r) => (
              <li key={r.doc} className="flex items-start gap-2 text-sm">
                <span className="mt-0.5 shrink-0 text-xs tabular-nums text-slate2">{Math.round(r.score * 100)}%</span>
                <span className="text-ink">
                  {r.title} {r.gr_number && <span className="text-teal">· {r.gr_number}</span>}
                </span>
              </li>
            ))}
          </ul>
        </Panel>
      )}

      <Panel>
        <h3 className="mb-2 text-sm font-semibold text-ink">Full text</h3>
        <div className="scroll-thin prose-answer max-h-[520px] overflow-y-auto rounded-lg bg-ice p-4 text-sm text-ink">
          {text}
        </div>
      </Panel>
    </div>
  );
}
