"use client";

import { FormEvent, useMemo, useState } from "react";
import { Bot, CheckCircle2, Clock3, FileText, Loader2, Route, Send, ShieldAlert } from "lucide-react";
import { demoAnswerResponse } from "../lib/demo-response.mjs";
import { buildFallbackAnswerResponse } from "../lib/fallback-response.mjs";
import { normalizeApiBase, reviewLabel, reviewTone } from "../lib/format.mjs";

type SourceItem = {
  source_id: string;
  title: string;
  path: string;
  score: number | null;
  snippet: string | null;
};

type TraceStep = {
  step: string;
  detail: string;
  latency_ms: number | null;
};

type AnswerResponse = {
  question: string;
  session_id: string;
  answer: string;
  mode: string;
  review_status: string | null;
  sources: SourceItem[];
  trace: TraceStep[];
};

const API_BASE = normalizeApiBase(process.env.NEXT_PUBLIC_API_BASE_URL);

export default function Page() {
  const [question, setQuestion] = useState(demoAnswerResponse.question);
  const [sessionId] = useState("web-demo");
  const [answer, setAnswer] = useState<AnswerResponse>(demoAnswerResponse);
  const [activePanel, setActivePanel] = useState<"sources" | "trace">("sources");
  const [isLoading, setIsLoading] = useState(false);
  const [apiState, setApiState] = useState<"demo" | "live" | "fallback">("demo");

  const reviewToneName = reviewTone(answer.review_status);
  const answerLines = useMemo(() => answer.answer.split("\n").filter(Boolean), [answer.answer]);

  async function submitQuestion(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed) return;
    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/v1/answer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed, session_id: sessionId }),
      });
      if (!response.ok) {
        throw new Error(`API returned ${response.status}`);
      }
      const payload = (await response.json()) as AnswerResponse;
      setAnswer(payload);
      setApiState("live");
    } catch (error) {
      const message = error instanceof Error ? error.message : "API request failed";
      setAnswer(buildFallbackAnswerResponse(trimmed, sessionId, message));
      setApiState("fallback");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="shell">
      <section className="workspace" aria-label="Agent console">
        <div className="topbar">
          <div>
            <p className="eyebrow">Phase6 Capstone</p>
            <h1>Enterprise Knowledge Agent</h1>
          </div>
          <div className={`status status-${reviewToneName}`}>
            {reviewToneName === "ok" ? <CheckCircle2 size={18} /> : <ShieldAlert size={18} />}
            <span>{reviewLabel(answer.review_status)}</span>
          </div>
        </div>

        <div className="content-grid">
          <section className="chat-surface" aria-label="Chat">
            <form className="question-bar" onSubmit={submitQuestion}>
              <label htmlFor="question">Question</label>
              <div className="input-row">
                <input
                  id="question"
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  placeholder="问一个企业知识库问题"
                />
                <button type="submit" disabled={isLoading} title="Send question">
                  {isLoading ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
                  <span>{isLoading ? "Running" : "Ask"}</span>
                </button>
              </div>
            </form>

            <div className="answer-block">
              <div className="answer-meta">
                <span className="mode-pill"><Bot size={15} />{answer.mode}</span>
                <span className="mode-pill"><Clock3 size={15} />{apiState}</span>
              </div>
              <div className="answer-text">
                {answerLines.map((line) => (
                  <p key={line}>{line}</p>
                ))}
              </div>
            </div>
          </section>

          <aside className="inspect-panel" aria-label="Inspection">
            <div className="tabs" role="tablist" aria-label="Inspection panels">
              <button
                className={activePanel === "sources" ? "active" : ""}
                onClick={() => setActivePanel("sources")}
                type="button"
              >
                <FileText size={17} />
                Sources
              </button>
              <button
                className={activePanel === "trace" ? "active" : ""}
                onClick={() => setActivePanel("trace")}
                type="button"
              >
                <Route size={17} />
                Trace
              </button>
            </div>

            {activePanel === "sources" ? (
              <div className="source-list">
                {answer.sources.length === 0 ? (
                  <p className="empty-state">No retrieved sources for this response.</p>
                ) : (
                  answer.sources.map((source, index) => (
                    <article className="source-item" key={source.source_id}>
                      <div className="source-head">
                        <span>{index + 1}</span>
                        <strong>{source.title}</strong>
                        <em>{source.score?.toFixed(3) ?? "n/a"}</em>
                      </div>
                      <p>{source.snippet}</p>
                      <code>{source.path}</code>
                    </article>
                  ))
                )}
              </div>
            ) : (
              <ol className="trace-list">
                {answer.trace.map((step, index) => (
                  <li key={`${step.step}-${index}`}>
                    <span>{index + 1}</span>
                    <div>
                      <strong>{step.step}</strong>
                      <p>{step.detail}</p>
                      <em>{step.latency_ms == null ? "sync" : `${step.latency_ms} ms`}</em>
                    </div>
                  </li>
                ))}
              </ol>
            )}
          </aside>
        </div>
      </section>
    </main>
  );
}
