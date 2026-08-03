// Typed client for the MahaGR FastAPI backend (backend/app/api.py).
const BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type Source = {
  n: number;
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
};

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

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail || detail;
    } catch {}
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => req<{ status: string; indexed_vectors: number; embedding_model: string }>("/health"),
  ask: (question: string, language: string, history?: { role: string; content: string }[]) =>
    req<AnswerResult>("/ask", { method: "POST", body: JSON.stringify({ question, language, history }) }),
  summarize: (doc_id: string, language: string) =>
    req<AnswerResult>("/summarize", { method: "POST", body: JSON.stringify({ doc_id, language }) }),
  explain: (question: string, language: string) =>
    req<AnswerResult>("/explain", { method: "POST", body: JSON.stringify({ question, language }) }),
  compare: (doc_a: string, doc_b: string, language: string) =>
    req<AnswerResult>("/compare", { method: "POST", body: JSON.stringify({ doc_a, doc_b, language }) }),
  documents: () => req<DocMeta[]>("/documents"),
  documentText: (id: string) =>
    req<{ doc_id: string; gr_number: string | null; date: string | null; title: string | null; text: string }>(
      `/documents/${encodeURIComponent(id)}/text`,
    ),
  supersede: (id: string) => req<Supersession>(`/supersede/${encodeURIComponent(id)}`),
  related: (id: string) => req<{ doc_id: string; related: Related[] }>(`/related/${encodeURIComponent(id)}`),
};

export { BASE as API_BASE };
