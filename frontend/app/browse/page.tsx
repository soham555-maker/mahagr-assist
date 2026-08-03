"use client";

import { useEffect, useMemo, useState } from "react";
import { GitCompareArrows, Search, ArrowRightLeft, Link2, ShieldCheck, FileText } from "lucide-react";
import {
  api,
  type DocMeta,
  type Supersession,
  type Related,
  type AnswerResult,
} from "@/lib/api";
import { EmptyHint, LangToggle, SourceCard, Spinner } from "@/components/ui";

export default function BrowsePage() {
  const [docs, setDocs] = useState<DocMeta[]>([]);
  const [query, setQuery] = useState("");
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);

  const [compareMode, setCompareMode] = useState(false);
  const [picks, setPicks] = useState<string[]>([]);
  const [language, setLanguage] = useState("auto");
  const [compareResult, setCompareResult] = useState<AnswerResult | null>(null);
  const [comparing, setComparing] = useState(false);

  useEffect(() => {
    api.documents().then(setDocs).catch((e) => setLoadErr((e as Error).message));
  }, []);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return docs;
    return docs.filter((d) =>
      [d.title, d.gr_number, d.date, d.doc].some((f) => (f || "").toLowerCase().includes(q)),
    );
  }, [docs, query]);

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
          <div className="flex flex-1 items-center gap-2 rounded-lg border border-line bg-white px-3 focus-within:border-teal">
            <Search size={16} className="text-slate2" aria-hidden />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search GRs…"
              aria-label="Search Government Resolutions"
              className="w-full bg-transparent py-2 text-sm outline-none placeholder:text-iceblue"
            />
          </div>
          <button
            onClick={() => {
              setCompareMode((v) => !v);
              setPicks([]);
              setCompareResult(null);
            }}
            aria-pressed={compareMode}
            className={`flex cursor-pointer items-center gap-1.5 rounded-lg border px-3 py-2 text-sm transition-colors duration-200 ${
              compareMode ? "border-teal bg-teal text-white" : "border-line bg-white text-slate2 hover:text-ink"
            }`}
          >
            <GitCompareArrows size={16} /> Compare
          </button>
        </div>

        {loadErr && (
          <p className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-800">
            Couldn’t load documents: {loadErr}. Is the backend running on port 8000?
          </p>
        )}

        <ul className="scroll-thin -mx-1 flex-1 space-y-1.5 overflow-y-auto px-1" style={{ maxHeight: "calc(100dvh - 160px)" }}>
          {filtered.map((d) => {
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
          {docs.length === 0 && !loadErr && <li className="p-3 text-sm text-slate2">Loading GRs…</li>}
        </ul>

        {compareMode && (
          <div className="mt-3 flex items-center gap-2 rounded-lg border border-line bg-white p-2">
            <LangToggle value={language} onChange={setLanguage} />
            <button
              onClick={runCompare}
              disabled={picks.length !== 2 || comparing}
              className="ml-auto flex cursor-pointer items-center gap-1.5 rounded-lg bg-navy px-3 py-2 text-sm text-white transition-colors duration-200 hover:bg-teal disabled:cursor-not-allowed disabled:opacity-40"
            >
              <ArrowRightLeft size={15} /> Compare&nbsp;these&nbsp;two ({picks.length}/2)
            </button>
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
          <DocDetail id={selected} />
        ) : (
          <EmptyHint>Select a Government Resolution to read it and see what it supersedes and relates to.</EmptyHint>
        )}
      </section>
    </main>
  );
}

function Panel({ children }: { children: React.ReactNode }) {
  return <div className="rounded-xl border border-line bg-white p-5">{children}</div>;
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

function DocDetail({ id }: { id: string }) {
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

  if (err) return <Panel><p className="text-sm text-red-800">{err}</p></Panel>;
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
          <button
            onClick={summarize}
            disabled={summing}
            className="flex shrink-0 cursor-pointer items-center gap-1.5 rounded-lg border border-line bg-white px-3 py-2 text-sm text-slate2 transition-colors duration-200 hover:border-teal hover:text-teal disabled:opacity-40"
          >
            <FileText size={15} /> Summarize
          </button>
        </div>
      </Panel>

      {summing ? (
        <Panel><Spinner label="Summarizing this GR…" /></Panel>
      ) : summary ? (
        <AnswerView title="Summary" result={summary} />
      ) : null}

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
