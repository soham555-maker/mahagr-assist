"use client";

import { useEffect, useRef, useState } from "react";
import { FileText, AlertTriangle, SlidersHorizontal, X, Download, ExternalLink } from "lucide-react";
import { api, downloadDocument, scopeCount, type CorpusStats, type Scope, type Source } from "@/lib/api";

export function Spinner({ label }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-slate2" role="status" aria-live="polite">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-line border-t-teal" aria-hidden />
      {label}
    </span>
  );
}

export function LangToggle({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const opts: [string, string][] = [["auto", "Auto"], ["en", "English"], ["mr", "मराठी"]];
  return (
    <div className="inline-flex rounded-lg border border-line bg-white p-0.5" role="group" aria-label="Answer language">
      {opts.map(([v, label]) => (
        <button
          key={v}
          type="button"
          onClick={() => onChange(v)}
          aria-pressed={value === v}
          className={`cursor-pointer rounded-md px-3 py-1 text-sm transition-colors duration-200 ${
            value === v ? "bg-navy text-white" : "text-slate2 hover:text-ink"
          }`}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

export function ScoreBar({ score }: { score: number }) {
  const pct = Math.max(0, Math.min(100, Math.round(score * 100)));
  return (
    <span className="inline-flex items-center gap-1.5" title={`relevance ${score.toFixed(3)}`}>
      <span className="h-1.5 w-16 overflow-hidden rounded-full bg-ice2" aria-hidden>
        <span className="block h-full rounded-full bg-teal" style={{ width: `${pct}%` }} />
      </span>
      <span className="text-xs tabular-nums text-slate2">{pct}%</span>
    </span>
  );
}

export function SourceCard({ s }: { s: Source }) {
  const [busy, setBusy] = useState(false);

  // SRS FR 3.7.4: a citation must be checkable, so it carries the actions that
  // let an officer verify it — open the GR, or take the text away.
  async function download() {
    if (!s.doc) return;
    setBusy(true);
    try {
      await downloadDocument(s.doc);
    } catch {
      /* the citation stays readable even if the download fails */
    } finally {
      setBusy(false);
    }
  }

  return (
    <li className="group rounded-lg border border-line bg-white p-3">
      <div className="flex items-start gap-2">
        <span className="mt-0.5 grid h-5 w-5 shrink-0 place-items-center rounded bg-navy text-[11px] font-semibold text-white">
          {s.n}
        </span>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium leading-snug text-ink">{s.title || s.source_file}</p>
          <p className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-slate2">
            {s.gr_number && <span className="font-medium text-teal">{s.gr_number}</span>}
            {s.date && <span>· {s.date}</span>}
            <span>· {s.pages}</span>
            <ScoreBar score={s.score} />
          </p>
        </div>
        {s.doc && (
          <div className="flex shrink-0 items-center gap-1">
            <a
              href={`/browse?doc=${encodeURIComponent(s.doc)}`}
              title="Open this GR"
              aria-label="Open this Government Resolution"
              className="rounded p-1 text-slate2 transition-colors hover:text-teal"
            >
              <ExternalLink size={14} />
            </a>
            <button
              onClick={download}
              disabled={busy}
              title="Download the full text"
              aria-label="Download this Government Resolution"
              className="rounded p-1 text-slate2 transition-colors hover:text-teal disabled:opacity-40"
            >
              <Download size={14} />
            </button>
          </div>
        )}
      </div>
    </li>
  );
}

/** A GR-domain answer is an abstention if the model says it lacks the info. */
export function isAbstention(text: string): boolean {
  const t = text.toLowerCase();
  return (
    t.includes("insufficient information") ||
    t.includes("does not appear to cover") ||
    t.includes("does not contain") ||
    text.includes("अपुरी माहिती") ||
    text.includes("पुरेशी माहिती नाही")
  );
}

export function AbstentionBanner() {
  return (
    <div className="flex items-start gap-2 rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900">
      <AlertTriangle size={18} className="mt-0.5 shrink-0" aria-hidden />
      <span>The assistant did not find this in the indexed documents and declined to answer rather than guess.</span>
    </div>
  );
}

/** Fetch the corpus shape once per mount. Shared by the Ask and Browse pages so
 *  neither has to hard-code a department list — the options come from whatever
 *  was actually ingested. */
export function useCorpusStats() {
  const [stats, setStats] = useState<CorpusStats | null>(null);
  useEffect(() => {
    api.corpusStats().then(setStats).catch(() => {});
  }, []);
  return stats;
}

/** "Searching 18,078 Government Resolutions across 6 departments." The point the
 *  whole scale phase exists to make, stated plainly rather than as a badge. */
export function CorpusStat({ stats }: { stats: CorpusStats | null }) {
  if (!stats?.documents) return null;
  const years =
    stats.date_from && stats.date_to
      ? `, ${stats.date_from.slice(0, 4)}–${stats.date_to.slice(0, 4)}`
      : "";
  return (
    <p className="text-sm text-slate2">
      Searching{" "}
      <span className="font-medium text-ink tabular-nums">{stats.documents.toLocaleString()}</span>{" "}
      Government Resolutions across{" "}
      <span className="font-medium text-ink">{stats.departments.length}</span>{" "}
      {stats.departments.length === 1 ? "department" : "departments"}
      {years}.
    </p>
  );
}

/** Department / date / language scope, collapsed behind one button.
 *
 *  Deliberately not always-on: filtering is the exception, and three permanent
 *  dropdowns above a chat box would dominate a screen whose primary action is
 *  typing a question. The button shows how many facets are active, so a
 *  narrowed search is never invisible — an officer must always be able to see
 *  that results were restricted. */
export function ScopeFilter({
  stats,
  value,
  onChange,
}: {
  stats: CorpusStats | null;
  value: Scope;
  onChange: (s: Scope) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const n = scopeCount(value);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onEsc = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onEsc);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onEsc);
    };
  }, [open]);

  const depts = stats?.departments || [];
  const selected = value.departments || [];

  function toggleDept(name: string) {
    const next = selected.includes(name) ? selected.filter((d) => d !== name) : [...selected, name];
    onChange({ ...value, departments: next.length ? next : undefined });
  }

  if (!depts.length) return null;

  return (
    <div className="relative" ref={ref}>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className={`flex cursor-pointer items-center gap-1.5 rounded-lg border px-3 py-1 text-sm transition-colors duration-200 ${
          n ? "border-teal bg-ice text-teal" : "border-line bg-white text-slate2 hover:text-ink"
        }`}
      >
        <SlidersHorizontal size={14} />
        Scope
        {n > 0 && (
          <span className="grid h-4 min-w-4 place-items-center rounded-full bg-teal px-1 text-[10px] font-semibold text-white">
            {n}
          </span>
        )}
      </button>

      {open && (
        <div className="absolute bottom-full left-0 z-20 mb-2 w-80 rounded-xl border border-line bg-white p-4 shadow-lg">
          <div className="mb-3 flex items-center justify-between">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate2">Limit the search</p>
            <button
              type="button"
              onClick={() => setOpen(false)}
              aria-label="Close"
              className="cursor-pointer text-slate2 hover:text-ink"
            >
              <X size={14} />
            </button>
          </div>

          <p className="mb-1.5 text-xs text-slate2">Department</p>
          <ul className="scroll-thin mb-3 max-h-40 space-y-0.5 overflow-y-auto">
            {depts.map((d) => (
              <li key={d.name}>
                <label className="flex cursor-pointer items-center gap-2 rounded px-1 py-1 text-sm text-ink hover:bg-ice">
                  <input
                    type="checkbox"
                    checked={selected.includes(d.name)}
                    onChange={() => toggleDept(d.name)}
                    className="accent-teal"
                  />
                  <span className="line-clamp-1 flex-1">{d.name.replace(/ Department$/, "")}</span>
                  <span className="text-xs tabular-nums text-slate2">{d.documents.toLocaleString()}</span>
                </label>
              </li>
            ))}
          </ul>

          <p className="mb-1.5 text-xs text-slate2">Issued between</p>
          <div className="mb-3 flex items-center gap-2">
            <input
              type="date"
              value={value.date_from || ""}
              min={stats?.date_from || undefined}
              max={stats?.date_to || undefined}
              onChange={(e) => onChange({ ...value, date_from: e.target.value || undefined })}
              aria-label="From date"
              className="w-full rounded-lg border border-line px-2 py-1 text-sm text-ink outline-none focus:border-teal"
            />
            <span className="text-xs text-slate2">to</span>
            <input
              type="date"
              value={value.date_to || ""}
              min={stats?.date_from || undefined}
              max={stats?.date_to || undefined}
              onChange={(e) => onChange({ ...value, date_to: e.target.value || undefined })}
              aria-label="To date"
              className="w-full rounded-lg border border-line px-2 py-1 text-sm text-ink outline-none focus:border-teal"
            />
          </div>

          <p className="mb-1.5 text-xs text-slate2">Document language</p>
          <div className="mb-3 inline-flex rounded-lg border border-line p-0.5">
            {([["", "Any"], ["mr", "मराठी"], ["en", "English"]] as [string, string][]).map(([v, label]) => (
              <button
                key={v}
                type="button"
                onClick={() => onChange({ ...value, doc_language: v || undefined })}
                aria-pressed={(value.doc_language || "") === v}
                className={`cursor-pointer rounded-md px-3 py-1 text-sm transition-colors duration-200 ${
                  (value.doc_language || "") === v ? "bg-navy text-white" : "text-slate2 hover:text-ink"
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          {n > 0 && (
            <button
              type="button"
              onClick={() => onChange({})}
              className="block w-full cursor-pointer rounded-lg border border-line py-1.5 text-sm text-slate2 transition-colors hover:border-teal hover:text-teal"
            >
              Clear all
            </button>
          )}
        </div>
      )}
    </div>
  );
}

/** A visible reminder, on the answer itself, that the corpus was narrowed —
 *  so a "nothing found" result is never mistaken for "the corpus lacks it". */
export function ScopeNotice({ scope, onClear }: { scope: Scope; onClear: () => void }) {
  const bits: string[] = [];
  if (scope.departments?.length)
    bits.push(scope.departments.map((d) => d.replace(/ Department$/, "")).join(", "));
  if (scope.date_from || scope.date_to)
    bits.push(`${scope.date_from || "start"} → ${scope.date_to || "now"}`);
  if (scope.doc_language) bits.push(scope.doc_language === "mr" ? "मराठी" : "English");
  if (!bits.length) return null;
  return (
    <p className="flex flex-wrap items-center gap-2 text-xs text-slate2">
      <span>Searched only: {bits.join(" · ")}</span>
      <button type="button" onClick={onClear} className="cursor-pointer underline hover:text-ink">
        clear
      </button>
    </p>
  );
}

export function EmptyHint({ icon, children }: { icon?: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="grid place-items-center gap-3 py-16 text-center text-slate2">
      <span className="grid h-12 w-12 place-items-center rounded-full bg-ice text-teal" aria-hidden>
        {icon || <FileText size={22} />}
      </span>
      <p className="max-w-sm text-sm">{children}</p>
    </div>
  );
}
