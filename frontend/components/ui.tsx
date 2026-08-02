import { FileText, AlertTriangle } from "lucide-react";
import type { Source } from "@/lib/api";

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
  return (
    <li className="rounded-lg border border-line bg-white p-3">
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
