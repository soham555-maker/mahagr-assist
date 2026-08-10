"use client";

import { useEffect, useRef, useState } from "react";
import { Send, ShieldCheck, Plus, Trash2, ThumbsUp, ThumbsDown, TriangleAlert } from "lucide-react";
import { api, scopeCount, type Scope, type Source, type ConvMeta } from "@/lib/api";
import {
  AbstentionBanner,
  CorpusStat,
  LangToggle,
  ScopeFilter,
  ScopeNotice,
  SourceCard,
  Spinner,
  isAbstention,
  useCorpusStats,
} from "@/components/ui";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";

type Msg = {
  role: "user" | "assistant";
  content: string;
  id?: string;
  sources?: Source[];
  warnings?: string[];
  abstain?: boolean;
  error?: boolean;
  feedback?: "up" | "down";
  /** The scope in force when this answer was produced — kept per message, not
   *  per page, so scrolling back shows what each answer actually searched. */
  scope?: Scope;
};

const SAMPLES = [
  "What is the procedure to approve new colleges under a university?",
  "ग्रंथालय संचालनालयाच्या आधुनिकीकरणाबाबतचा निर्णय काय आहे?",
];

export default function ChatPage() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [language, setLanguage] = useState("auto");
  const [explainMode, setExplainMode] = useState(false);
  const [loading, setLoading] = useState(false);
  const [conversations, setConversations] = useState<ConvMeta[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [scope, setScope] = useState<Scope>({});
  const stats = useCorpusStats();
  const endRef = useRef<HTMLDivElement>(null);

  const refreshConversations = () => api.conversations().then(setConversations).catch(() => {});
  useEffect(() => {
    api.conversations().then(setConversations).catch(() => {});
  }, []);

  function newChat() {
    setMessages([]);
    setConversationId(null);
  }

  async function loadConversation(id: string) {
    try {
      const res = await api.conversation(id);
      setMessages(
        res.messages.map((m) => ({
          role: m.role,
          content: m.content,
          id: m.id,
          sources: m.sources,
          warnings: m.warnings,
          abstain: m.role === "assistant" && isAbstention(m.content),
        })),
      );
      setConversationId(id);
    } catch {
      /* ignore */
    }
  }

  async function removeConversation(id: string, e: React.MouseEvent) {
    e.stopPropagation();
    await api.deleteConversation(id).catch(() => {});
    if (id === conversationId) newChat();
    refreshConversations();
  }

  function vote(idx: number, rating: "up" | "down") {
    const m = messages[idx];
    if (!m.id || !conversationId) return;
    setMessages((arr) => arr.map((x, i) => (i === idx ? { ...x, feedback: rating } : x)));
    api.feedback(conversationId, m.id, rating).catch(() => {});
  }

  async function send(question: string) {
    if (!question.trim() || loading) return;
    setMessages((m) => [...m, { role: "user", content: question }]);
    setInput("");
    setLoading(true);
    setTimeout(() => endRef.current?.scrollIntoView({ behavior: "smooth" }), 50);
    try {
      const res = explainMode
        ? await api.explain(question, language)
        : await api.ask(question, language, conversationId, scope);
      if (!explainMode && res.conversation_id) {
        setConversationId(res.conversation_id);
        refreshConversations();
      }
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: res.answer,
          id: res.message_id,
          sources: res.sources,
          warnings: res.warnings,
          abstain: isAbstention(res.answer),
          scope: explainMode ? undefined : scope,
        },
      ]);
    } catch (e) {
      setMessages((m) => [
        ...m,
        { role: "assistant", content: `Could not reach the assistant: ${(e as Error).message}`, error: true },
      ]);
    } finally {
      setLoading(false);
      setTimeout(() => endRef.current?.scrollIntoView({ behavior: "smooth" }), 50);
    }
  }

  return (
    <main className="mx-auto grid max-w-6xl gap-5 px-4 py-4 md:grid-cols-[240px_1fr]">
      {/* ---- sidebar: conversation history ---- */}
      <aside className="hidden md:flex md:flex-col">
        <Button
          onClick={newChat}
          variant="outline"
          className="mb-3 w-full border-line bg-white text-ink hover:border-teal hover:text-teal"
        >
          <Plus size={15} className="mr-1.5" /> New chat
        </Button>
        <ul className="scroll-thin space-y-1 overflow-y-auto" style={{ maxHeight: "calc(100dvh - 130px)" }}>
          {conversations.map((c) => (
            <li key={c.id}>
              <div
                onClick={() => loadConversation(c.id)}
                className={`group flex cursor-pointer items-center gap-2 rounded-lg border px-3 py-2 text-sm transition-colors duration-200 ${
                  c.id === conversationId ? "border-teal bg-ice" : "border-transparent hover:bg-ice"
                }`}
              >
                <span className="line-clamp-1 flex-1 text-ink">{c.title}</span>
                <button
                  onClick={(e) => removeConversation(c.id, e)}
                  aria-label="Delete conversation"
                  className="text-slate2 opacity-0 transition-opacity hover:text-red-600 group-hover:opacity-100"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            </li>
          ))}
          {conversations.length === 0 && (
            <li className="px-3 py-2 text-xs text-slate2">No past conversations yet.</li>
          )}
        </ul>
      </aside>

      {/* ---- chat column ---- */}
      <div className="flex min-h-[calc(100dvh-80px)] flex-col">
        <div className="flex flex-1 flex-col">
          {messages.length === 0 ? (
            <div className="flex flex-1 flex-col justify-center py-10">
              <h1 className="font-serif text-3xl font-semibold text-ink">Ask about Government Resolutions</h1>
              <p className="mt-2 max-w-xl text-slate2">
                Answers come only from the indexed GRs, with a citation on every claim. Ask in English or Marathi.
              </p>
              <div className="mt-3">
                <CorpusStat stats={stats} />
              </div>
              <div className="mt-6 flex flex-wrap gap-2">
                {SAMPLES.map((s) => (
                  <Button
                    key={s}
                    onClick={() => send(s)}
                    variant="outline"
                    className="h-auto whitespace-normal rounded-full border-line bg-white px-4 py-2 text-left font-normal text-ink hover:border-teal hover:text-teal"
                  >
                    {s}
                  </Button>
                ))}
              </div>
            </div>
          ) : (
            <div className="scroll-thin flex-1 space-y-5 py-6">
              {messages.map((m, i) =>
                m.role === "user" ? (
                  <div key={i} className="flex justify-end">
                    <p className="max-w-[80%] rounded-2xl rounded-br-sm bg-navy px-4 py-2.5 text-white">{m.content}</p>
                  </div>
                ) : (
                  <div key={i} className="max-w-[92%]">
                    {m.error ? (
                      <Alert variant="destructive">
                        <AlertDescription>{m.content}</AlertDescription>
                      </Alert>
                    ) : (
                      <div className="rounded-2xl rounded-bl-sm border border-line bg-white p-4">
                        {m.abstain && (
                          <div className="mb-3">
                            <AbstentionBanner />
                          </div>
                        )}
                        {m.warnings && m.warnings.length > 0 && (
                          <Alert className="mb-3 border-amber-300 bg-amber-50 text-amber-900 [&>svg]:text-amber-600">
                            <TriangleAlert className="h-4 w-4" />
                            <AlertTitle>Possible conflict — check before relying on this</AlertTitle>
                            <AlertDescription>
                              <ul className="mt-1 list-disc space-y-0.5 pl-4">
                                {m.warnings.map((w, wi) => (
                                  <li key={wi}>{w}</li>
                                ))}
                              </ul>
                            </AlertDescription>
                          </Alert>
                        )}
                        <div className="prose-answer text-[15px] text-ink">{m.content}</div>
                        {m.scope && scopeCount(m.scope) > 0 && (
                          <div className="mt-3 border-t border-line pt-2">
                            <ScopeNotice scope={m.scope} onClear={() => setScope({})} />
                          </div>
                        )}
                        {m.sources && m.sources.length > 0 && (
                          <div className="mt-4 border-t border-line pt-3">
                            <p className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-slate2">
                              <ShieldCheck size={14} className="text-teal" /> Sources
                            </p>
                            <ul className="grid gap-2">
                              {m.sources.map((s) => (
                                <SourceCard key={s.n} s={s} />
                              ))}
                            </ul>
                          </div>
                        )}
                        {m.id && (
                          <div className="mt-3 flex items-center gap-2 text-slate2">
                            <span className="text-xs">Helpful?</span>
                            <Button
                              onClick={() => vote(i, "up")}
                              aria-label="Helpful"
                              aria-pressed={m.feedback === "up"}
                              variant="ghost"
                              size="icon"
                              className={cn("h-7 w-7 hover:text-teal", m.feedback === "up" && "text-teal")}
                            >
                              <ThumbsUp size={14} />
                            </Button>
                            <Button
                              onClick={() => vote(i, "down")}
                              aria-label="Not helpful"
                              aria-pressed={m.feedback === "down"}
                              variant="ghost"
                              size="icon"
                              className={cn(
                                "h-7 w-7 hover:text-destructive",
                                m.feedback === "down" && "text-destructive",
                              )}
                            >
                              <ThumbsDown size={14} />
                            </Button>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ),
              )}
              {loading && (
                <div className="max-w-[92%] rounded-2xl rounded-bl-sm border border-line bg-white p-4">
                  <Spinner label="Retrieving and grounding an answer…" />
                </div>
              )}
              <div ref={endRef} />
            </div>
          )}
        </div>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            send(input);
          }}
          className="sticky bottom-0 border-t border-line bg-[#f7fafc] py-3"
        >
          <div className="mb-2 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <LangToggle value={language} onChange={setLanguage} />
              {!explainMode && <ScopeFilter stats={stats} value={scope} onChange={setScope} />}
              <div
                className="flex items-center gap-2"
                title="Answer in simple, plain language"
              >
                <Switch
                  id="explain-mode"
                  checked={explainMode}
                  onCheckedChange={setExplainMode}
                  className="data-[state=checked]:bg-teal"
                />
                <Label htmlFor="explain-mode" className="cursor-pointer text-xs font-normal text-slate2">
                  Explain simply
                </Label>
              </div>
            </div>
            {messages.length > 0 && (
              <Button
                type="button"
                onClick={newChat}
                variant="ghost"
                size="sm"
                className="h-7 text-xs text-slate2 hover:text-ink"
              >
                New chat
              </Button>
            )}
          </div>
          <div className="flex items-end gap-2 rounded-xl border border-line bg-white p-2 focus-within:border-teal">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send(input);
                }
              }}
              rows={1}
              placeholder="Ask in English or Marathi…"
              aria-label="Your question"
              className="max-h-40 flex-1 resize-none bg-transparent px-2 py-1.5 text-ink outline-none placeholder:text-iceblue"
            />
            <Button
              type="submit"
              disabled={loading || !input.trim()}
              aria-label="Send"
              size="icon"
              className="h-9 w-9 shrink-0 bg-teal text-white hover:bg-navy disabled:opacity-40"
            >
              <Send size={16} />
            </Button>
          </div>
        </form>
      </div>
    </main>
  );
}
