"use client";

import { useEffect, useState } from "react";
import { api, type CorpusStats } from "@/lib/api";
import { cn } from "@/lib/utils";

const FALLBACK = { documents: 18080, chunks: 74004, departments: 6 };

/**
 * The corpus size, read live from /corpus/stats so the landing page can never
 * quote a number the running system does not actually hold.
 *
 * Falls back to the last measured figures if the API is down — a landing page
 * that renders "—" because a backend is asleep is worse than one that renders
 * the true-as-of-build numbers, and the distinction is shown in the caption.
 */
export function LiveCorpusStat({ className }: { className?: string }) {
  const [stats, setStats] = useState<CorpusStats | null>(null);
  const [live, setLive] = useState(false);

  useEffect(() => {
    let cancelled = false;
    api
      .corpusStats()
      .then((s) => {
        if (!cancelled) {
          setStats(s);
          setLive(true);
        }
      })
      .catch(() => {
        /* backend not running — the fallback below is used */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const documents = stats?.documents ?? FALLBACK.documents;
  const chunks = stats?.chunks ?? FALLBACK.chunks;
  // the fixture pseudo-department is not one of the six real ones
  const departments = stats
    ? stats.departments.filter((d) => d.documents > 100).length
    : FALLBACK.departments;

  const items = [
    { value: documents.toLocaleString("en-IN"), label: "Government Resolutions indexed" },
    { value: chunks.toLocaleString("en-IN"), label: "embedded passages (1024-d)" },
    { value: String(departments), label: "departments, one shared index" },
    { value: "0", label: "bytes leaving the machine" },
  ];

  return (
    <div className={className}>
      <dl className="grid grid-cols-2 gap-x-6 gap-y-7 sm:grid-cols-4">
        {items.map((it) => (
          <div key={it.label}>
            <dt className="font-serif text-3xl font-semibold tabular-nums text-white sm:text-4xl">
              {it.value}
            </dt>
            <dd className="mt-1.5 text-[12.5px] leading-snug text-iceblue">{it.label}</dd>
          </div>
        ))}
      </dl>
      <p className="mt-5 flex items-center gap-2 text-[11.5px] text-iceblue/80">
        <span
          className={cn(
            "inline-block h-1.5 w-1.5 rounded-full",
            live ? "bg-teal-bright" : "bg-iceblue/50",
          )}
          aria-hidden
        />
        {live ? "Live from the running index" : "Last measured figures — backend not reachable"}
      </p>
    </div>
  );
}
