"""
PRISM runtime wrapper for evaluation: runs variants and returns RunTrace per query.

Variants:
- llm_only: OpenAI with question only (no retrieval, no tools, no personalization).
- retriever_only: retrieval returns top chunks; answer is trivial list of chunks.
- no_personalization: full graph with user_context["eval_no_personalization"]=True.
- no_internal_eval: full graph with skip_internal_eval=True.
- full_system: full graph as-is.
"""

import logging
import time
import uuid
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def _run_llm_only(question: str, course_name: str) -> Dict[str, Any]:
    """Answer with LLM only (no retrieval, no tools)."""
    try:
        from openai import OpenAI
        from config.settings import OPENAI_API_KEY, OPENAI_MODEL
        client = OpenAI(api_key=OPENAI_API_KEY)
        start = time.perf_counter()
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful teaching assistant. Answer the student's question concisely based on general knowledge. Do not use external tools or retrieval."},
                {"role": "user", "content": question},
            ],
            temperature=0.5,
            max_tokens=1500,
        )
        latency_ms = int((time.perf_counter() - start) * 1000)
        text = (resp.choices[0].message.content or "").strip()
        return {
            "final_answer_text": text,
            "route_taken": "course",
            "tools_used": [],
            "retrieval_context": [],
            "web_context": [],
            "citations": [],
            "internal_eval_scores": None,
            "latency_ms": latency_ms,
            "token_usage": getattr(resp, "usage", None) and {"total_tokens": getattr(resp.usage, "total_tokens", None)},
            "error": None,
        }
    except Exception as e:
        logger.exception(f"llm_only failed: {e}")
        return {
            "final_answer_text": "",
            "route_taken": "course",
            "tools_used": [],
            "retrieval_context": [],
            "web_context": [],
            "citations": [],
            "internal_eval_scores": None,
            "latency_ms": None,
            "token_usage": None,
            "error": str(e),
        }


def _run_retriever_only(question: str, course_name: str, top_k: int = 15) -> Dict[str, Any]:
    """Retrieve top chunks and return them as a trivial answer (no generated answer)."""
    try:
        from retrieval.retriever import CourseRetriever
        retriever = CourseRetriever()
        start = time.perf_counter()
        chunks = retriever.retrieve(query=question, course_name=course_name, top_k=top_k)
        latency_ms = int((time.perf_counter() - start) * 1000)
        retrieval_context = []
        parts = ["Retrieved chunks:\n"]
        for i, c in enumerate(chunks, 1):
            loc = c.get("page_number") or c.get("timestamp") or "—"
            retrieval_context.append({
                "chunk_id": c.get("id"),
                "doc_id": c.get("document_name"),
                "location": loc,
                "text": (c.get("content") or "")[:2000],
            })
            parts.append(f"{i}. {c.get('document_name')} (Page/Timestamp: {loc}): { (c.get('content') or '')[:300]}...")
        final_answer_text = "\n".join(parts)
        return {
            "final_answer_text": final_answer_text,
            "route_taken": "course",
            "tools_used": ["vector_retrieval"],
            "retrieval_context": retrieval_context,
            "web_context": [],
            "citations": [],
            "internal_eval_scores": None,
            "latency_ms": latency_ms,
            "token_usage": None,
            "error": None,
        }
    except Exception as e:
        logger.exception(f"retriever_only failed: {e}")
        return {
            "final_answer_text": "",
            "route_taken": "course",
            "tools_used": [],
            "retrieval_context": [],
            "web_context": [],
            "citations": [],
            "internal_eval_scores": None,
            "latency_ms": None,
            "token_usage": None,
            "error": str(e),
        }


def _state_to_route_taken(state: Dict[str, Any]) -> str:
    if state.get("is_vague"):
        return "clarify"
    if not state.get("is_relevant", True):
        return "refuse"
    if state.get("course_content_found"):
        return "course"
    return "web"


def _state_to_tools_used(state: Dict[str, Any]) -> List[str]:
    tools = []
    if state.get("course_content_found") or state.get("retrieved_chunks"):
        tools.append("vector_retrieval")
    if state.get("web_search_results") and "not available" not in (state.get("web_search_results") or "").lower():
        tools.append("web_search")
    return tools


def _state_to_retrieval_context(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    chunks = state.get("retrieved_chunks") or []
    out = []
    for c in chunks:
        loc = c.get("page_number") or c.get("timestamp") or "—"
        out.append({
            "chunk_id": c.get("id"),
            "doc_id": c.get("document_name"),
            "location": loc,
            "text": (c.get("content") or "")[:2000],
        })
    return out


def _state_to_web_context(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    citations = state.get("web_search_citations") or []
    out = []
    for c in citations if isinstance(c, dict) else []:
        url = c.get("url") or c.get("link") or ""
        domain = c.get("domain") or ""
        if not domain and url:
            try:
                from urllib.parse import urlparse
                domain = urlparse(url).netloc or ""
            except Exception:
                pass
        snippet = c.get("snippet") or c.get("content") or c.get("title") or ""
        out.append({"url": url, "domain": domain, "snippet": snippet, "text": snippet[:2000]})
    # If citations are not dicts, use raw web_search_results as one blob
    if not out and state.get("web_search_results"):
        out.append({"url": "", "domain": "", "snippet": state["web_search_results"][:2000], "text": state["web_search_results"][:2000]})
    return out


def run_prism(
    course_id: str,
    student_profile: str,
    question: str,
    variant_config: str,
) -> Dict[str, Any]:
    """
    Run PRISM (or ablation) for one query and return a RunTrace.
    
    Args:
        course_id: e.g. INFO4100
        student_profile: undergrad | masters | phd
        question: student question
        variant_config: llm_only | retriever_only | no_personalization | no_internal_eval | full_system
        
    Returns:
        RunTrace dict (final_answer_text, route_taken, tools_used, retrieval_context, web_context, citations, internal_eval_scores, latency_ms, token_usage, error)
    """
    from eval_runner.config import get_course_name, PROFILE_TO_USER_CONTEXT

    course_name = get_course_name(course_id)
    user_context = dict(PROFILE_TO_USER_CONTEXT.get(student_profile, PROFILE_TO_USER_CONTEXT["undergrad"]))

    if variant_config == "llm_only":
        return _run_llm_only(question, course_name)

    if variant_config == "retriever_only":
        return _run_retriever_only(question, course_name)

    # Run full graph (no_personalization, no_internal_eval, full_system)
    try:
        from core.graph import create_agent_graph
        from core.state import create_initial_state
        from langchain_core.messages import HumanMessage

        user_context["eval_no_personalization"] = variant_config == "no_personalization"
        skip_internal_eval = variant_config == "no_internal_eval"

        graph = create_agent_graph()
        initial_state = create_initial_state(
            query=question,
            course_name=course_name,
            user_context=user_context,
            conversation_history=None,
            skip_internal_eval=skip_internal_eval,
        )
        config = {"configurable": {"thread_id": f"eval_{uuid.uuid4().hex}"}}
        start = time.perf_counter()
        final_state = graph.invoke(initial_state, config=config)
        latency_ms = int((time.perf_counter() - start) * 1000)

        # Follow-up (vague) path
        if final_state.get("is_vague") and final_state.get("follow_up_questions"):
            q = (final_state["follow_up_questions"] or [""])[0]
            return {
                "final_answer_text": f"I need more information. {q}" if q else "Could you please provide more details?",
                "route_taken": "clarify",
                "tools_used": [],
                "retrieval_context": [],
                "web_context": [],
                "citations": [],
                "internal_eval_scores": None,
                "latency_ms": latency_ms,
                "token_usage": None,
                "error": None,
            }

        # Not relevant (refuse) path
        if not final_state.get("is_relevant", True):
            return {
                "final_answer_text": (final_state.get("final_response") or "This question is not relevant to the course.").strip(),
                "route_taken": "refuse",
                "tools_used": [],
                "retrieval_context": [],
                "web_context": [],
                "citations": [],
                "internal_eval_scores": None,
                "latency_ms": latency_ms,
                "token_usage": None,
                "error": None,
            }

        # Normal answer path
        return {
            "final_answer_text": (final_state.get("final_response") or "").strip(),
            "route_taken": _state_to_route_taken(final_state),
            "tools_used": _state_to_tools_used(final_state),
            "retrieval_context": _state_to_retrieval_context(final_state),
            "web_context": _state_to_web_context(final_state),
            "citations": final_state.get("response_citations") or [],
            "internal_eval_scores": final_state.get("evaluation_scores"),
            "latency_ms": latency_ms,
            "token_usage": None,
            "error": None,
        }
    except Exception as e:
        logger.exception(f"run_prism ({variant_config}) failed: {e}")
        return {
            "final_answer_text": "",
            "route_taken": "course",
            "tools_used": [],
            "retrieval_context": [],
            "web_context": [],
            "citations": [],
            "internal_eval_scores": None,
            "latency_ms": None,
            "token_usage": None,
            "error": str(e),
        }
