"""LangGraph support bot with hybrid retrieval and confidence-based escalation.

Graph (read top to bottom):

    START -> retrieve -> grade -> [answer | escalate] -> END

Nodes:
  retrieve  - run hybrid BM25+FAISS over the KB, attach similarity scores
  grade     - compute a numeric confidence and, in borderline cases, ask the
              LLM to judge whether the retrieved context actually answers the
              question. This is the key anti-hallucination gate.
  answer    - LLM synthesizes a grounded answer that cites sources by id
  escalate  - produce a human-handoff message including the user's query and
              the top retrieval candidates so the human starts with context

Confidence policy:
  top_sim >= HIGH_CONF             -> answer
  LOW_CONF <= top_sim < HIGH_CONF  -> LLM-as-judge ("does this context answer
                                     the user's question? yes/no") decides
  top_sim <  LOW_CONF              -> escalate

These thresholds are tuned against the eval set; see README for the trade-offs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Literal, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from retriever import HybridRetriever, build_retriever

HIGH_CONF = 0.55
LOW_CONF = 0.35

ANSWER_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o-mini"

SYSTEM_ANSWER = """You are a CloudSync Pro support agent. Answer ONLY from the provided context.
- Be concise (2-5 sentences) and direct.
- If the context contains a step-by-step procedure, preserve the steps.
- Cite sources inline as [source_id] using the ids shown in the context.
- If the context does not actually answer the question, reply with exactly: INSUFFICIENT_CONTEXT
Do not invent URLs, prices, version numbers, or policies that aren't in the context."""

SYSTEM_JUDGE = """You are a strict grader. Given a user question and a candidate context,
answer YES only if the context directly and specifically answers the question.
Answer NO if the context is merely topically related, partial, or about a different feature.
Respond with exactly one token: YES or NO."""


class BotState(TypedDict, total=False):
    query: str
    docs: List[Document]
    scored: List[tuple]  # (Document, float)
    top_sim: float
    decision: Literal["answer", "escalate"]
    judge_used: bool
    answer: str
    escalation_reason: str


@dataclass
class Bot:
    retriever: HybridRetriever
    answer_llm: ChatOpenAI
    judge_llm: ChatOpenAI
    graph: object

    def ask(self, query: str) -> dict:
        return self.graph.invoke({"query": query})


def _format_context(scored: List[tuple]) -> str:
    lines = []
    for doc, sim in scored:
        lines.append(
            f"[{doc.metadata['id']}] ({doc.metadata['source']} - sim={sim:.2f}) {doc.metadata['title']}\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(lines)


def build_bot(retriever: Optional[HybridRetriever] = None) -> Bot:
    retriever = retriever or build_retriever()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it before running: export OPENAI_API_KEY=sk-..."
        )

    answer_llm = ChatOpenAI(model=ANSWER_MODEL, temperature=0)
    judge_llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0)

    def retrieve_node(state: BotState) -> BotState:
        docs = retriever.retrieve(state["query"])
        scored = retriever.score_query(state["query"], docs)
        scored.sort(key=lambda x: x[1], reverse=True)
        top_sim = scored[0][1] if scored else 0.0
        return {"docs": [d for d, _ in scored], "scored": scored, "top_sim": top_sim}

    def grade_node(state: BotState) -> BotState:
        top = state["top_sim"]
        if top >= HIGH_CONF:
            return {"decision": "answer", "judge_used": False}
        if top < LOW_CONF:
            return {
                "decision": "escalate",
                "judge_used": False,
                "escalation_reason": f"Low retrieval similarity (top={top:.2f} < {LOW_CONF}).",
            }
        # Borderline -> LLM-as-judge on the single best candidate.
        best_doc, _ = state["scored"][0]
        verdict = judge_llm.invoke(
            [
                SystemMessage(content=SYSTEM_JUDGE),
                HumanMessage(
                    content=f"QUESTION:\n{state['query']}\n\nCONTEXT:\n{best_doc.page_content}"
                ),
            ]
        ).content.strip().upper()
        if verdict.startswith("YES"):
            return {"decision": "answer", "judge_used": True}
        return {
            "decision": "escalate",
            "judge_used": True,
            "escalation_reason": f"Borderline similarity ({top:.2f}); judge said NO.",
        }

    def answer_node(state: BotState) -> BotState:
        context = _format_context(state["scored"])
        resp = answer_llm.invoke(
            [
                SystemMessage(content=SYSTEM_ANSWER),
                HumanMessage(content=f"QUESTION:\n{state['query']}\n\nCONTEXT:\n{context}"),
            ]
        ).content.strip()
        # LLM second-line defense: if it admits it can't answer, escalate.
        if "INSUFFICIENT_CONTEXT" in resp.upper():
            return {
                "decision": "escalate",
                "answer": "",
                "escalation_reason": "Answer LLM flagged context as insufficient.",
            }
        return {"answer": resp}

    def escalate_node(state: BotState) -> BotState:
        top_title = state["scored"][0][0].metadata["title"] if state.get("scored") else "(no candidates)"
        msg = (
            "I'm not confident I can answer this accurately, so I'm escalating to a human agent.\n"
            f"Your question: {state['query']}\n"
            f"Closest match in our knowledge base: {top_title}\n"
            "A support agent will reach out shortly."
        )
        return {"answer": msg}

    def route(state: BotState) -> str:
        return state["decision"]

    workflow = StateGraph(BotState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("answer", answer_node)
    workflow.add_node("escalate", escalate_node)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges("grade", route, {"answer": "answer", "escalate": "escalate"})
    # The answer node may itself flip to escalate; route off its output too.
    workflow.add_conditional_edges(
        "answer",
        lambda s: "escalate" if s.get("decision") == "escalate" else END,
        {"escalate": "escalate", END: END},
    )
    workflow.add_edge("escalate", END)

    return Bot(retriever=retriever, answer_llm=answer_llm, judge_llm=judge_llm, graph=workflow.compile())


if __name__ == "__main__":
    bot = build_bot()
    out = bot.ask("how do I turn on 2fa?")
    print(out["answer"])
    print(f"\n[decision={out['decision']}, top_sim={out['top_sim']:.2f}, judge_used={out.get('judge_used')}]")
