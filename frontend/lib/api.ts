// Typed client for the MahaGR FastAPI backend (backend/app/api.py).
const BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type Source = {
  n: number;
  /** The GR's document id — what /documents/{id}/text and /graph/{id} accept,
   *  so a citation is actionable (open/download) rather than a dead label. */
  doc: string | null;
  title: string;
  gr_number: string | null;
  date: string | null;
  department: string | null;
  language: string | null;
  pages: string;
  content_type: string;
  score: number;
  source_file: string | null;
};

export type AnswerResult = {
  answer: string;
  sources: Source[];
  phantom_citations: number[];
  warnings?: string[];
  low_confidence?: boolean;
  model?: string;
  conversation_id?: string;
  message_id?: string;
};

export type ConvMeta = { id: string; title: string; created_at: number; messages: number };
export type StoredMsg = { id: string; role: "user" | "assistant"; content: string; sources: Source[]; warnings: string[] };

export type DocMeta = {
  doc: string;
  gr_number: string | null;
  date: string | null;
  department: string | null;
  language: string | null;
  title: string | null;
};

export type Supersession = {
  found: boolean;
  gr_number: string;
  declares_supersession: boolean;
  cites: { gr_number: string; in_corpus: string | null }[];
  superseded_by: { doc: string; gr_number: string; date: string; title: string }[];
};

export type Related = { doc: string; score: number; gr_number: string | null; date: string | null; title: string };

export type GraphNode = {
  id: string;
  gr_number: string | null;
  date: string | null;
  title: string;
  department: string | null;
};
export type GraphEdge = { src: string; dst: string; kind: "supersedes" | "cites" };
/** A reference to a GR the corpus does not hold. Surfaced rather than hidden:
 *  "this order builds on something we don't have" is information, not an error. */
export type GraphDangling = {
  src_id: string;
  gr_number: string;
  /** The date printed beside the number in the citing GR — what an officer
   *  needs in order to go and find the missing order on the GR portal. */
  date: string | null;
  kind: string;
  resolution: "dangling" | "ambiguous";
};
export type GraphResult = {
  found: boolean;
  root: GraphNode;
  nodes: GraphNode[];
  edges: GraphEdge[];
  /** Transitive supersede chain, oldest link first. The LAST entry is the
   *  order actually in force. */
  chain: GraphNode[];
  dangling: GraphDangling[];
};

/** What the corpus actually holds — drives the "searching N GRs" line and the
 *  department filter options, so the UI never hard-codes a department list. */
export type CorpusStats = {
  documents: number;
  chunks: number;
  departments: { name: string; documents: number }[];
  date_from: string | null;
  date_to: string | null;
};

/** Restricts a search to part of the corpus. Every field is optional; an empty
 *  object means "search everything". */
export type Scope = {
  departments?: string[];
  date_from?: string;
  date_to?: string;
  doc_language?: string;
};

export function scopeCount(s: Scope): number {
  return (
    (s.departments?.length ? 1 : 0) +
    (s.date_from || s.date_to ? 1 : 0) +
    (s.doc_language ? 1 : 0)
  );
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const token = typeof window !== "undefined" ? sessionStorage.getItem("mahagr_token") : null;
  const headers: Record<string, string> = { 
    "Content-Type": "application/json", 
    ...(token ? { "Authorization": `Bearer ${token}` } : {}),
    ...(init?.headers as any || {}) 
  };

  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers,
  });
  if (res.status === 401) {
    if (typeof window !== "undefined" && window.location.pathname !== "/login") {
      sessionStorage.removeItem("mahagr_token");
      window.location.href = "/login";
    }
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail || detail;
    } catch {}
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

function qs(params: Record<string, string | number | string[] | undefined>): string {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === "") continue;
    if (Array.isArray(v)) v.forEach((item) => p.append(k, item));
    else p.set(k, String(v));
  }
  const s = p.toString();
  return s ? `?${s}` : "";
}

export const api = {
  login: (data: URLSearchParams) =>
    req<{ access_token: string; token_type: string; role: string; username: string }>("/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: data.toString(),
    }),
  health: () =>
    req<{
      status: string;
      indexed_vectors: number;
      vector_backend: string;
      documents: number | null;
      departments: number | null;
      embedding_model: string;
    }>("/health"),
  corpusStats: () => req<CorpusStats>("/corpus/stats"),
  ask: (question: string, language: string, conversation_id?: string | null, scope?: Scope) =>
    req<AnswerResult>("/ask", {
      method: "POST",
      body: JSON.stringify({ question, language, conversation_id, ...(scope || {}) }),
    }),
  conversations: () => req<ConvMeta[]>("/conversations"),
  conversation: (id: string) => req<{ conversation_id: string; messages: StoredMsg[] }>(`/conversations/${encodeURIComponent(id)}`),
  deleteConversation: (id: string) => req<{ deleted: string }>(`/conversations/${encodeURIComponent(id)}`, { method: "DELETE" }),
  feedback: (conversation_id: string, message_id: string, rating: "up" | "down") =>
    req<{ feedback_id: string }>("/feedback", { method: "POST", body: JSON.stringify({ conversation_id, message_id, rating }) }),
  summarize: (doc_id: string, language: string) =>
    req<AnswerResult>("/summarize", { method: "POST", body: JSON.stringify({ doc_id, language }) }),
  explain: (question: string, language: string) =>
    req<AnswerResult>("/explain", { method: "POST", body: JSON.stringify({ question, language }) }),
  compare: (doc_a: string, doc_b: string, language: string) =>
    req<AnswerResult>("/compare", { method: "POST", body: JSON.stringify({ doc_a, doc_b, language }) }),
  // Paginated: the corpus is ~18,000 GRs, so the list is a page at a time.
  documents: (opts?: { q?: string; department?: string[]; limit?: number; offset?: number }) =>
    req<DocMeta[]>(`/documents${qs({ ...opts })}`),
  documentText: (id: string) =>
    req<{
      doc_id: string;
      gr_number: string | null;
      date: string | null;
      department?: string | null;
      title: string | null;
      text: string;
    }>(`/documents/${encodeURIComponent(id)}/text`),
  supersede: (id: string) => req<Supersession>(`/supersede/${encodeURIComponent(id)}`),
  // The supersede/citation neighbourhood. Requires the graph to have been built
  // on the backend: `python scripts/build_graph.py`.
  graph: (id: string, depth = 1) =>
    req<GraphResult>(`/graph/${encodeURIComponent(id)}${qs({ depth })}`),
  related: (id: string) => req<{ doc_id: string; related: Related[] }>(`/related/${encodeURIComponent(id)}`),
};

/**
 * Fetch a GR's full text and save it as a .txt file (SRS FR 3.7.4 — "view and
 * download referenced Government documents").
 *
 * Done client-side from the existing /documents/{id}/text endpoint rather than
 * adding a download route: the text is already served, and this keeps the
 * on-prem story simple — no file streaming, no temp files on the server.
 * The original PDFs are not in the corpus (orgpedia ships pre-OCR'd text), so
 * .txt is the honest artifact to hand over, not a re-rendered PDF.
 */
export async function downloadDocument(id: string) {
  const d = await api.documentText(id);
  const header = [
    d.gr_number && `GR Number: ${d.gr_number}`,
    d.date && `Date: ${d.date}`,
    d.department && `Department: ${d.department}`,
    d.title && `Subject: ${d.title}`,
    `Source: MahaGR Assist (document id ${d.doc_id})`,
  ]
    .filter(Boolean)
    .join("\n");
  const blob = new Blob([`${header}\n${"-".repeat(60)}\n\n${d.text}`], {
    type: "text/plain;charset=utf-8",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${d.gr_number?.replace(/[^\w.-]+/g, "_") || d.doc_id}.txt`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export { BASE as API_BASE };
