"""Wrapper to run PRISM agent pipeline for evaluation with variant support.

Variants work by:
- full_system: Normal graph execution
- no_internal_eval: Sets skip_internal_eval=True in state (already supported)
- no_personalization: Bypasses personalization with a plain response
- no_rag: Forces course_content_found=False so it always goes to web or skips
- no_web_search: Patches web_search_node to no-op
- no_query_refinement: Patches query_refinement to pass-through
- baseline: Plain GPT-4o with no agents or RAG
"""

import sys
import time
import logging
import random
import traceback
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import OPENAI_API_KEY, OPENAI_MODEL

logger = logging.getLogger(__name__)


@dataclass
class RunTrace:
    """Captures full trace of a PRISM pipeline run."""
    eval_id: str = ""
    variant: str = ""
    profile: str = ""
    course_id: str = ""
    question: str = ""
    final_answer: str = ""
    route_taken: str = ""
    tools_used: list = field(default_factory=list)
    retrieval_context: list = field(default_factory=list)
    web_context: list = field(default_factory=list)
    citations: list = field(default_factory=list)
    internal_eval_scores: dict = field(default_factory=dict)
    is_vague: bool = False
    is_relevant: bool = True
    needs_follow_up: bool = False
    follow_up_question: str = ""
    source_type: str = ""
    response_history: list = field(default_factory=list)
    latency_ms: float = 0.0
    error: str = ""

    def to_dict(self):
        return asdict(self)


def _retry_with_backoff(fn, max_retries=3, base_delay=5):
    """Retry a function with exponential backoff on rate limit errors."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = "rate limit" in err_str or "429" in err_str or "too many requests" in err_str
            if not is_rate_limit or attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
            logger.warning(f"Rate limit hit, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)


class PRISMEvalWrapper:
    """Runs PRISM pipeline with variant controls."""

    def __init__(self):
        self._graphs = {}

    def _get_graph(self, variant: str):
        """Get or create a graph for this variant."""
        if variant not in self._graphs:
            self._graphs[variant] = self._build_variant_graph(variant)
        return self._graphs[variant]

    def _build_variant_graph(self, variant: str):
        """Build a LangGraph with variant-specific modifications."""
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.memory import MemorySaver
        from core.state import AgentState
        from core.nodes.query_refinement import query_refinement_node
        from core.nodes.relevance import relevance_node
        from core.nodes.course_rag import course_rag_node
        from core.nodes.web_search import web_search_node
        from core.nodes.personalization import personalization_node
        from core.nodes.evaluation import evaluation_node
        from core.nodes.refinement import refinement_node
        from core.graph import (
            route_after_query_refinement,
            route_after_relevance,
            route_after_course_rag,
            route_after_evaluation,
        )

        # Define pass-through / no-op node functions for variants
        def query_refinement_passthrough(state: AgentState) -> dict:
            """Skip query refinement — treat all queries as clear."""
            return {"is_vague": False, "follow_up_questions": [], "refined_query": state.get("query")}

        def course_rag_disabled(state: AgentState) -> dict:
            """Skip course RAG — force web search fallback."""
            return {"course_content_found": False, "course_context": None, "course_citations": [], "retrieved_chunks": []}

        def web_search_disabled(state: AgentState) -> dict:
            """Skip web search — return empty results."""
            return {"web_search_results": "No web search performed (disabled for evaluation).", "web_search_citations": []}

        def personalization_passthrough(state: AgentState) -> dict:
            """Skip personalization — pass context directly as response."""
            context = state.get("course_context") or state.get("web_search_results") or ""
            query = state.get("refined_query") or state.get("query", "")
            # Generate a plain response without personalization
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            try:
                def _call_personalization():
                    return client.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=[
                            {"role": "system", "content": "Answer the question using the provided context. Do not adapt for any specific audience level."},
                            {"role": "user", "content": f"Context:\n{context[:3000]}\n\nQuestion: {query}"},
                        ],
                        temperature=0.7,
                        max_tokens=2000,
                    )
                resp = _retry_with_backoff(_call_personalization)
                answer = resp.choices[0].message.content
            except Exception:
                answer = context[:2000]

            citations = state.get("course_citations", []) + state.get("web_search_citations", [])
            return {"final_response": answer, "response_citations": citations}

        # Select node functions based on variant
        qr_node = query_refinement_passthrough if variant == "no_query_refinement" else query_refinement_node
        rag_node = course_rag_disabled if variant == "no_rag" else course_rag_node
        ws_node = web_search_disabled if variant == "no_web_search" else web_search_node
        pers_node = personalization_passthrough if variant == "no_personalization" else personalization_node

        workflow = StateGraph(AgentState)

        workflow.add_node("query_refinement", qr_node)
        workflow.add_node("relevance", relevance_node)
        workflow.add_node("course_rag", rag_node)
        workflow.add_node("web_search", ws_node)
        workflow.add_node("personalization", pers_node)
        workflow.add_node("evaluation", evaluation_node)
        workflow.add_node("refinement", refinement_node)

        workflow.set_entry_point("query_refinement")

        workflow.add_conditional_edges("query_refinement", route_after_query_refinement,
                                        {"relevance": "relevance", "end": END})
        workflow.add_conditional_edges("relevance", route_after_relevance,
                                        {"course_rag": "course_rag", "end": END})
        workflow.add_conditional_edges("course_rag", route_after_course_rag,
                                        {"personalization": "personalization", "web_search": "web_search"})
        workflow.add_edge("web_search", "personalization")
        workflow.add_edge("personalization", "evaluation")
        workflow.add_conditional_edges("evaluation", route_after_evaluation,
                                        {"refinement": "refinement", "end": END})
        workflow.add_edge("refinement", "evaluation")

        checkpointer = MemorySaver()
        return workflow.compile(checkpointer=checkpointer)

    def run(
        self,
        question: str,
        course_id: str,
        user_context: dict,
        variant: str,
        eval_id: str,
        profile: str,
    ) -> RunTrace:
        trace = RunTrace(
            eval_id=eval_id,
            variant=variant,
            profile=profile,
            course_id=course_id,
            question=question,
        )

        start = time.time()
        try:
            if variant == "baseline":
                return self._run_baseline(trace, question, course_id)

            graph = self._get_graph(variant)

            from core.state import create_initial_state
            thread_id = f"eval_{eval_id}_{variant}_{profile}"
            config = {"configurable": {"thread_id": thread_id}}

            skip_eval = variant == "no_internal_eval"
            initial_state = create_initial_state(
                query=question,
                course_name=course_id,
                user_context=user_context,
                skip_internal_eval=skip_eval,
            )

            final_state = _retry_with_backoff(
                lambda: graph.invoke(initial_state, config=config)
            )

            # Extract results from final state
            trace.final_answer = final_state.get("final_response", "") or ""
            trace.is_vague = final_state.get("is_vague", False)
            trace.is_relevant = final_state.get("is_relevant", True)
            trace.needs_follow_up = final_state.get("is_vague", False) and bool(final_state.get("follow_up_questions"))
            trace.follow_up_question = (
                final_state.get("follow_up_questions", [""])[0]
                if final_state.get("follow_up_questions") else ""
            )
            trace.citations = final_state.get("response_citations", [])
            trace.internal_eval_scores = final_state.get("evaluation_scores", {}) or {}
            trace.response_history = final_state.get("response_history", [])

            # Determine route and tools
            trace.route_taken = self._determine_route(final_state)
            trace.tools_used = self._extract_tools(final_state)

            # Extract contexts for metric computation
            if final_state.get("course_context"):
                trace.retrieval_context = [final_state["course_context"]]
            if final_state.get("retrieved_chunks"):
                chunks = final_state["retrieved_chunks"]
                if isinstance(chunks, list):
                    trace.retrieval_context = [
                        c.get("chunk_text", c.get("text", str(c))) if isinstance(c, dict) else str(c)
                        for c in chunks
                    ]
            if final_state.get("web_search_results"):
                trace.web_context = [final_state["web_search_results"]]

            trace.source_type = "web" if not final_state.get("course_content_found", False) else "course"

        except Exception as e:
            trace.error = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Error running {variant} for {eval_id}: {e}")
            traceback.print_exc()

        trace.latency_ms = (time.time() - start) * 1000
        return trace

    def _run_baseline(self, trace: RunTrace, question: str, course_id: str) -> RunTrace:
        """Plain GPT-4o with no RAG, no agents."""
        from openai import OpenAI

        start = time.time()
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            def _call_baseline():
                return client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a helpful teaching assistant. Answer the student's "
                                "question about their course material clearly and accurately. "
                                f"The course is: {course_id}."
                            ),
                        },
                        {"role": "user", "content": question},
                    ],
                    temperature=0.7,
                    max_tokens=2000,
                )
            response = _retry_with_backoff(_call_baseline)
            trace.final_answer = response.choices[0].message.content
            trace.route_taken = "baseline_direct"
            trace.tools_used = []
            trace.source_type = "baseline"
        except Exception as e:
            trace.error = f"{type(e).__name__}: {str(e)}"

        trace.latency_ms = (time.time() - start) * 1000
        return trace

    def _determine_route(self, state: dict) -> str:
        if state.get("is_vague") and state.get("follow_up_questions"):
            return "clarify"
        if not state.get("is_relevant", True):
            return "refuse"
        if not state.get("course_content_found", False):
            return "web"
        return "course"

    def _extract_tools(self, state: dict) -> list:
        tools = []
        if state.get("course_content_found", False):
            tools.append("vector_retrieval")
        if state.get("web_search_results") and not state.get("course_content_found", False):
            tools.append("web_search")
        return tools
