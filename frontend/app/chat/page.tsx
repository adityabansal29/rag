"use client";

import { useEffect, useRef, useState } from "react";
import { Send, ChevronDown, ChevronUp, Plus, BookOpen } from "lucide-react";
import { sendChat, type Source } from "@/lib/api";
import { cn } from "@/lib/utils";

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
}

function TypingDots() {
  return (
    <div className="flex items-center gap-1.5 px-4 py-3.5">
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          className="w-2 h-2 rounded-full bg-muted-foreground animate-bounce"
          style={{ animationDelay: `${i * 0.15}s`, animationDuration: "0.8s" }}
        />
      ))}
    </div>
  );
}

function SourceCard({ source, rank }: { source: Source; rank: number }) {
  const [open, setOpen] = useState(false);
  const typeColors: Record<string, string> = {
    text:  "bg-cyan-700/15 text-cyan-700",
    image: "bg-olive-300/15 text-olive-300",
    table: "bg-mist-400/15 text-mist-400",
  };
  return (
    <div className="rounded-xl border border-border bg-card overflow-hidden text-xs">
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between px-3 py-2.5 hover:bg-muted/30 transition-colors text-left gap-3"
      >
        <div className="flex items-center gap-2 min-w-0">
          <span className="shrink-0 w-5 h-5 rounded-full gradient-brand flex items-center justify-center text-white font-bold text-[9px]">
            {rank}
          </span>
          <span className="truncate text-muted-foreground font-medium">{source.source || "unknown"}</span>
          {source.page && <span className="shrink-0 text-muted-foreground">p.{source.page}</span>}
          <span className={cn("shrink-0 px-1.5 py-0.5 rounded-md font-medium", typeColors[source.type || "text"] ?? "bg-muted text-muted-foreground")}>
            {source.type || "text"}
          </span>
          <span className="shrink-0 text-olive-300 font-semibold">{Number(source.score).toFixed(3)}</span>
        </div>
        {open ? <ChevronUp size={12} className="shrink-0 text-muted-foreground" /> : <ChevronDown size={12} className="shrink-0 text-muted-foreground" />}
      </button>
      {open && (
        <div className="px-3 pb-3 pt-1 border-t border-border">
          <p className="text-muted-foreground leading-relaxed whitespace-pre-wrap">{source.content}</p>
        </div>
      )}
    </div>
  );
}

function AssistantMessage({ msg }: { msg: Message }) {
  const [showSources, setShowSources] = useState(false);
  const hasSources = msg.sources && msg.sources.length > 0;
  return (
    <div className="flex flex-col gap-2.5 max-w-[80%]">
      <div className="rounded-2xl rounded-tl-sm bg-card border border-border px-4 py-3.5 text-sm leading-relaxed whitespace-pre-wrap">
        {msg.content}
      </div>
      {hasSources && (
        <div className="space-y-2">
          <button
            onClick={() => setShowSources((o) => !o)}
            className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors"
          >
            <BookOpen size={11} />
            {showSources ? "Hide" : "Show"} {msg.sources!.length} source{msg.sources!.length !== 1 ? "s" : ""}
            {showSources ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
          </button>
          {showSources && (
            <div className="space-y-1.5">
              {msg.sources!.map((s, i) => <SourceCard key={i} source={s} rank={i + 1} />)}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [threadId, setThreadId] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    let tid = sessionStorage.getItem("rag_thread_id");
    if (!tid) {
      tid = crypto.randomUUID();
      sessionStorage.setItem("rag_thread_id", tid);
    }
    setThreadId(tid);
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const send = async () => {
    const text = input.trim();
    if (!text || loading) return;
    setInput("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
    setMessages((m) => [...m, { role: "user", content: text }]);
    setLoading(true);
    try {
      const { answer, sources } = await sendChat(text, threadId);
      setMessages((m) => [...m, { role: "assistant", content: answer, sources }]);
    } catch {
      setMessages((m) => [...m, { role: "assistant", content: "Something went wrong. Please try again." }]);
    } finally {
      setLoading(false);
    }
  };

  const newChat = () => {
    const tid = crypto.randomUUID();
    sessionStorage.setItem("rag_thread_id", tid);
    setThreadId(tid);
    setMessages([]);
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  };

  return (
    <div className="flex flex-col h-screen">
      {/* Header */}
      <div className="shrink-0 border-b border-border px-6 py-4 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold tracking-tight">Chat</h1>
          {threadId && (
            <p className="text-xs text-muted-foreground font-mono mt-0.5">
              session · {threadId.slice(0, 8)}
            </p>
          )}
        </div>
        <button
          onClick={newChat}
          className="flex items-center gap-1.5 text-sm font-medium px-3 py-1.5 rounded-lg bg-muted hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors"
        >
          <Plus size={14} />
          New chat
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-8 space-y-6">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full gap-4 text-center">
            <div className="w-16 h-16 rounded-2xl gradient-brand flex items-center justify-center shadow-lg">
              <Send size={22} className="text-white" />
            </div>
            <div>
              <p className="font-semibold text-lg">Ask anything</p>
              <p className="text-sm text-muted-foreground mt-1">
                Questions are answered from your indexed documents.
              </p>
            </div>
            <div className="grid grid-cols-1 gap-2 mt-2 w-full max-w-sm">
              {[
                "Summarize the key findings",
                "What are the main topics covered?",
                "List the most important conclusions",
              ].map((prompt) => (
                <button
                  key={prompt}
                  onClick={() => { setInput(prompt); textareaRef.current?.focus(); }}
                  className="text-left px-4 py-2.5 rounded-xl border border-border bg-card hover:bg-muted/40 text-sm text-muted-foreground hover:text-foreground transition-colors"
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={cn("flex", msg.role === "user" ? "justify-end" : "justify-start")}>
            {msg.role === "user" ? (
              <div className="rounded-2xl rounded-tr-sm px-4 py-3.5 text-sm max-w-[75%] leading-relaxed whitespace-pre-wrap gradient-brand text-white font-medium shadow-sm">
                {msg.content}
              </div>
            ) : (
              <AssistantMessage msg={msg} />
            )}
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="rounded-2xl rounded-tl-sm bg-card border border-border">
              <TypingDots />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="shrink-0 border-t border-border px-6 py-4 bg-background/80 backdrop-blur-sm">
        <div className="flex gap-3 items-end max-w-3xl mx-auto">
          <div className="flex-1 rounded-2xl border border-border bg-card focus-within:border-cyan-700/60 focus-within:ring-2 focus-within:ring-cyan-700/20 transition-all">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="Ask a question about your documents…"
              rows={1}
              className="w-full resize-none bg-transparent px-4 py-3 text-sm focus:outline-none placeholder:text-muted-foreground leading-relaxed"
              style={{ minHeight: "48px", maxHeight: "160px" }}
              onInput={(e) => {
                const t = e.currentTarget;
                t.style.height = "auto";
                t.style.height = Math.min(t.scrollHeight, 160) + "px";
              }}
            />
          </div>
          <button
            onClick={send}
            disabled={!input.trim() || loading}
            className="shrink-0 h-12 w-12 rounded-2xl gradient-primary-btn text-white flex items-center justify-center shadow-sm"
          >
            <Send size={16} />
          </button>
        </div>
        <p className="text-center text-[11px] text-muted-foreground/60 mt-2">Enter to send · Shift+Enter for newline</p>
      </div>
    </div>
  );
}
