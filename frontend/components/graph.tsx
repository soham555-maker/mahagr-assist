"use client";

import { useEffect, useState } from "react";
import { GitBranch, AlertTriangle, ArrowRight } from "lucide-react";
import { api, type GraphNode, type GraphResult } from "@/lib/api";
import { Spinner } from "@/components/ui";

/**
 * The supersede/citation graph for one GR (PLAN Phase 3).
 *
 * WHY HAND-DRAWN SVG AND NOT react-flow / cytoscape
 * -------------------------------------------------
 * Those libraries are 100-400 kB of JS and bring their own visual language,
 * which would then have to be restyled to match the portal. What we actually
 * need is small and fixed: one focal GR, its direct neighbours, and a linear
 * supersede chain — a radial layout computed with two lines of trigonometry.
 * The whole component is a few kB and inherits the existing theme tokens.
 * If the graph ever needs pan/zoom/drag over hundreds of nodes, that is the
 * point to reach for a real graph library.
 *
 * The layout is deliberately DETERMINISTIC (no force simulation): the same GR
 * always renders the same way, so a presenter can point at a node and it will
 * still be there on the next run.
 */

const W = 560;
const H = 300;
const CX = W / 2;
const CY = H / 2;
const R = 108;

type Placed = GraphNode & { x: number; y: number; role: "root" | "newer" | "older" };

// How many neighbours one arc can hold before the labels collide. A well-cited
// GR in an 18k corpus can have 16+ neighbours; drawing them all turns the
// diagram into a smear, so the rest are stated as a count instead of drawn.
const PER_ARC = 7;

function layout(res: GraphResult): { placed: Placed[]; hidden: number } {
  const superseders = new Set(
    res.edges.filter((e) => e.kind === "supersedes" && e.dst === res.root.id).map((e) => e.src),
  );
  const others = res.nodes.filter((n) => n.id !== res.root.id);

  // Nodes that supersede this GR go on top, everything else below — so
  // "something replaced this" is readable as a direction, not just a colour.
  // Newest first, so a cap keeps the links that matter.
  const byDate = (a: GraphNode, b: GraphNode) => (b.date || "").localeCompare(a.date || "");
  const above = others.filter((n) => superseders.has(n.id)).sort(byDate);
  const below = others.filter((n) => !superseders.has(n.id)).sort(byDate);
  const hidden = Math.max(0, above.length - PER_ARC) + Math.max(0, below.length - PER_ARC);

  const place = (all: GraphNode[], from: number, to: number, role: Placed["role"]) => {
    const list = all.slice(0, PER_ARC);
    return list.map((n, i) => {
      const t = list.length === 1 ? 0.5 : i / (list.length - 1);
      const a = from + (to - from) * t;
      return { ...n, x: CX + R * Math.cos(a), y: CY + R * Math.sin(a) * 0.72, role };
    });
  };

  return {
    placed: [
      { ...res.root, x: CX, y: CY, role: "root" as const },
      ...place(above, Math.PI * 1.25, Math.PI * 1.75, "newer"),
      ...place(below, Math.PI * 0.75, Math.PI * 0.25, "older"),
    ],
    hidden,
  };
}

const FILL: Record<Placed["role"], string> = {
  root: "var(--navy, #0f2b46)",
  newer: "var(--teal, #0f766e)",
  older: "#ffffff",
};

export function GraphPanel({
  docId,
  onSelect,
}: {
  docId: string;
  onSelect?: (id: string) => void;
}) {
  const [data, setData] = useState<GraphResult | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!docId) return;
    setLoading(true);
    setErr(null);
    setData(null);
    api
      .graph(docId, 1)
      .then(setData)
      .catch((e) => setErr((e as Error).message))
      .finally(() => setLoading(false));
  }, [docId]);

  if (loading) return <Spinner label="Loading the supersession graph…" />;
  if (err)
    return (
      <p className="rounded-lg border border-line bg-white p-3 text-sm text-slate2">
        Graph unavailable: {err}. Has <code>scripts/build_graph.py</code> been run?
      </p>
    );
  if (!data?.found) return null;

  const { placed, hidden } = layout(data);
  const at = (id: string) => placed.find((p) => p.id === id);
  // With a full arc the numbers overlap; the hover title still carries them.
  const showLabels = placed.length <= 9;
  const inForce = data.chain.length ? data.chain[data.chain.length - 1] : null;

  return (
    <div className="rounded-lg border border-line bg-white p-4">
      <p className="mb-3 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-slate2">
        <GitBranch size={14} className="text-teal" /> Supersession &amp; references
      </p>

      {/* The headline finding, stated in words before the picture — a presenter
          should not have to interpret a diagram to deliver the point. */}
      {inForce && (
        <div className="mb-3 flex items-start gap-2 rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900">
          <AlertTriangle size={16} className="mt-0.5 shrink-0" aria-hidden />
          <span>
            This GR appears to have been superseded. The order likely in force is{" "}
            <button
              onClick={() => onSelect?.(inForce.id)}
              className="cursor-pointer font-semibold underline underline-offset-2"
            >
              {inForce.gr_number || inForce.id}
            </button>
            {inForce.date && ` (${inForce.date})`}. Verify before relying on it.
          </span>
        </div>
      )}

      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img"
           aria-label="Supersession and citation graph">
        <defs>
          <marker id="gr-arrow" viewBox="0 0 10 10" refX="9" refY="5"
                  markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--slate2, #64748b)" />
          </marker>
        </defs>

        {data.edges.map((e, i) => {
          const a = at(e.src);
          const b = at(e.dst);
          if (!a || !b) return null;
          return (
            <line
              key={i}
              x1={a.x} y1={a.y} x2={b.x} y2={b.y}
              stroke="var(--slate2, #64748b)"
              strokeWidth={e.kind === "supersedes" ? 2 : 1}
              strokeDasharray={e.kind === "supersedes" ? undefined : "4 3"}
              markerEnd="url(#gr-arrow)"
              opacity={0.55}
            />
          );
        })}

        {placed.map((n) => (
          <g
            key={n.id}
            transform={`translate(${n.x},${n.y})`}
            onClick={() => n.role !== "root" && onSelect?.(n.id)}
            className={n.role === "root" ? "" : "cursor-pointer"}
          >
            <title>{`${n.gr_number || n.id}${n.date ? ` (${n.date})` : ""}\n${n.title}`}</title>
            <circle
              r={n.role === "root" ? 13 : 9}
              fill={FILL[n.role]}
              stroke={n.role === "older" ? "var(--slate2, #64748b)" : "none"}
              strokeWidth={1}
            />
            {(showLabels || n.role === "root") && (
              <text
                y={n.role === "root" ? 30 : 24}
                textAnchor="middle"
                className="fill-ink"
                fontSize="10"
              >
                {(n.gr_number || n.id).slice(0, 22)}
              </text>
            )}
            {n.date && (showLabels || n.role === "root") && (
              <text y={n.role === "root" ? 42 : 35} textAnchor="middle"
                    className="fill-slate2" fontSize="9">
                {n.date.slice(0, 4)}
              </text>
            )}
            {!showLabels && n.role !== "root" && n.date && (
              <text y={n.role === "newer" ? -14 : 22} textAnchor="middle"
                    className="fill-slate2" fontSize="9">
                {n.date.slice(0, 4)}
              </text>
            )}
          </g>
        ))}
      </svg>

      <ul className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate2">
        <li className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-full bg-navy" /> this GR
        </li>
        <li className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-full bg-teal" /> supersedes it
        </li>
        <li className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-full border border-slate2 bg-white" /> referenced
        </li>
        <li>— solid = supersedes · dashed = cites</li>
        {!showLabels && <li>hover a node for its GR number</li>}
        {hidden > 0 && <li>+{hidden} more linked GRs not drawn</li>}
      </ul>

      {data.chain.length > 1 && (
        <div className="mt-3 border-t border-line pt-3">
          <p className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-slate2">
            Supersession chain
          </p>
          <p className="flex flex-wrap items-center gap-1.5 text-sm text-ink">
            <span className="text-slate2">{data.root.gr_number || data.root.id}</span>
            {data.chain.map((c) => (
              <span key={c.id} className="flex items-center gap-1.5">
                <ArrowRight size={13} className="text-slate2" />
                <button onClick={() => onSelect?.(c.id)} className="cursor-pointer underline underline-offset-2">
                  {c.gr_number || c.id}
                </button>
              </span>
            ))}
          </p>
        </div>
      )}

      {/* Ghost references: shown, not silently dropped. The gap IS the finding. */}
      {data.dangling.length > 0 && (
        <div className="mt-3 border-t border-line pt-3">
          <p className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-slate2">
            Referenced but not in the corpus ({data.dangling.length})
          </p>
          <ul className="flex flex-wrap gap-1.5">
            {data.dangling.slice(0, 8).map((d, i) => (
              <li
                key={i}
                title={d.resolution === "ambiguous"
                  ? "Matched more than one GR — not linked, to avoid a wrong link"
                  : "This GR is referenced but is not in our corpus"}
                className="rounded-full border border-dashed border-line px-2.5 py-0.5 text-xs text-slate2"
              >
                {d.gr_number.slice(0, 34)}
                {/* The cited date is what makes a missing order findable. */}
                {d.date && <span className="ml-1 text-slate2/70">· {d.date}</span>}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
