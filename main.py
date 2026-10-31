from typing import Dict, List

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from token_utils import count_tokens, estimate_cost
from cache_utils import get_cached_response, save_cached_response

from langchain_agent import (
    build_system_prompt,
    generate_answer,
    load_history,
    summarize_history,
    dpath,
)
from agent_verifier import verified_retrieve
from ollama_diagnostics import ContextMonitor, set_monitor, reset_monitor
from redact_pii import redact_pii


from fastapi import Request
import time
from typing import Dict, List, Literal
import ollama
from fastapi.responses import StreamingResponse
 
import asyncio
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from guardrails_config import validate_input, validate_output
import os

from langfuse import observe, get_client, propagate_attributes

import re

# define private requests vs public contact information



db_path = dpath / "coverage.db"
db_path.parent.mkdir(parents=True, exist_ok=True)
# SQL FUNCTION


def init_db():
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                session_id INTEGER,
                role TEXT,
                content TEXT,
                timestamp DATETIME
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS token_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                estimated_cost REAL NOT NULL
            )
            """
        )
        conn.commit()


init_db()

langfuse = get_client()


def save_message(session_id, role, content):
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO conversations (
                session_id,
                role,
                content,
                timestamp
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                session_id,
                role,
                content,
                datetime.now(timezone.utc).isoformat()
            )
        )
        conn.commit()

def save_token_usage(
    session_id,
    input_tokens,
    output_tokens,
    estimated_cost
):
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO token_usage (
                session_id,
                timestamp,
                input_tokens,
                output_tokens,
                estimated_cost
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                session_id,
                datetime.now(timezone.utc).isoformat(),
                input_tokens,
                output_tokens,
                estimated_cost
            )
        )
        conn.commit()

# PYDANTIC CLASSES

class ChatRequest(BaseModel):
    session_id: int
    member_id: int
    message: str


class SessionTurn(BaseModel):
    role: str
    message: str


class ChatResponse(BaseModel):
    session_id: int
    member_id: int
    answer: str
    structure: str
    chunk_ids: List[str]
    tool_result: dict | None
    tool_calls: List[dict]
    history: List[SessionTurn]
    diagnostics: dict | None = None
    context_warning: dict | None = None


class SessionState(BaseModel):
    session_id: int
    member_id: int
    history: List[SessionTurn]


# In-memory session store keyed by session_id.
# For future persistence, this can be replaced with a SQLite-backed store.
session_store: Dict[int, SessionState] = {}


def get_session(session_id: int, member_id: int) -> SessionState:
    session = session_store.get(session_id)
    if session is None:
        session = SessionState(session_id=session_id, member_id=member_id, history=[])
        session_store[session_id] = session
    elif session.member_id != member_id:
        raise HTTPException(
            status_code=403,
            detail="Session does not belong to this member."
        )
    return session


def get_member_id(request: Request):
    return request.headers.get("X-Member-ID", "unknown")

PRIVATE_TOOLS = {"get_claim_status"}


@observe(as_type="generation", name="llm-generate-answer")
def call_llm_with_tracing(question: str, context: str, instructions: str):
    start_time = time.perf_counter()

    full_answer = generate_answer(
        question,
        context,
        instructions
    )

    latency_ms = (time.perf_counter() - start_time) * 1000

    input_tokens = (
        count_tokens(question)
        + count_tokens(context)
        + count_tokens(instructions)
    )

    output_tokens = count_tokens(full_answer)

    langfuse.update_current_generation(
        input={
            "question": question,
            "context": context,
            "instructions": instructions,
        },
        output=full_answer,
        usage_details={
            "input": input_tokens,
            "output": output_tokens,
        },
        metadata={"latency_ms": round(latency_ms, 2)},
    )

    return full_answer, input_tokens, output_tokens, latency_ms
 
 
limiter = Limiter(key_func=get_member_id)
 
 
app = FastAPI()
 
 
app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler
)
 
# Add middelware for timeout + error logging
 
@app.on_event("startup")
async def warm_ollama():
    print("[STARTUP] Warming Ollama qwen3:8b...")

    for attempt in range(5):
        try:
            await asyncio.to_thread(
                ollama.chat,
                model="qwen3:8b",
                messages=[
                    {
                        "role": "user",
                        "content": "Reply only with OK."
                    }
                ],
                think=False,
                options={
                    "num_predict": 2,
                    "temperature": 0,
                },
                keep_alive="30m",
            )

            print("[STARTUP] Ollama warm-up complete")
            return

        except Exception as error:
            print(
                f"[STARTUP] Ollama warm-up attempt "
                f"{attempt + 1}/5 failed: {error}"
            )

            if attempt < 4:
                await asyncio.sleep(5)

    print("[STARTUP] Ollama warm-up failed; backend continuing anyway")


@app.middleware("http")
async def log_request_time(request: Request, call_next):
    start_time = time.perf_counter()
    status_code = 500
 
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
 
    finally:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
 
        print(
            f"{request.method} {request.url.path} "
            f"| status={status_code} "
            f"| time={elapsed_ms:.2f} ms"
        )
 
 
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
 
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/system-status")
def system_status():
    """Public, minimal health details; never expose evaluation traces or prompts."""
    monitor = ContextMonitor()
    monitor.refresh_windows()
    return {
        "status": "ok",
        "ollama_reachable": monitor.ollama_reachable,
        "model": "qwen3:8b",
        "context_length": monitor.window("qwen3:8b"),
    }
 
 
@app.on_event("shutdown")
def shutdown_flush_langfuse():
    langfuse.flush()

 
@app.post("/chat")
@limiter.limit("5/minute")
@observe(
    name="chat",
    capture_input=False,
    capture_output=False
)
async def chat(request: Request, chat_request: ChatRequest):
    overall_start = time.perf_counter()
    timings = {}
    eval_diagnostics = (
        os.getenv("EVAL_DIAGNOSTICS", "").lower() == "true"
        and request.headers.get("X-Eval-Diagnostics") == "true"
    )
    print(f"\n--- [DEBUG] Starting chat request for session: {chat_request.session_id} ---")

    session = get_session(
        chat_request.session_id,
        chat_request.member_id
    )

    with propagate_attributes(
        session_id=str(chat_request.session_id),
        user_id=str(chat_request.member_id),
    ):
        t_start = time.perf_counter()
        allowed, reason = validate_input(chat_request.message)
        timings["input_guardrail_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
        print(f"[TIMING] Guardrail validation took: {timings['input_guardrail_ms']:.2f} ms")
    
        if not allowed:
            print(f"INPUT GUARDRAIL | blocked=True | reason={reason}")
            langfuse.update_current_span(
                metadata={
                    "guardrail_blocked": "true",
                    "block_reason": str(reason),
                }
            )
            message = (
                "I can't process that request because it violates "
                "the chatbot's safety or privacy guardrails."
            )
            response = ChatResponse(
                session_id=session.session_id,
                member_id=session.member_id,
                answer=message,
                structure="",
                chunk_ids=[],
                tool_result=None,
                tool_calls=[],
                history=session.history,
                diagnostics={
                    "guardrails": {"input_allowed": False, "input_reason": reason},
                    "timings_ms": timings,
                } if eval_diagnostics else None,
            )
            event = f"data: {response.model_dump_json()}\n\n"
            return StreamingResponse(iter([event]), media_type="text/event-stream")
    
        safe_user_message = redact_pii(chat_request.message)

        langfuse.update_current_span(
            input=safe_user_message
        )

        contains_pii = safe_user_message != chat_request.message

        private_request = contains_pii or bool(
            re.search(
                r"\b(?:[CM]\d+|claims?|members?)\b",
                chat_request.message,
                re.IGNORECASE,
            )
        )
    
        session.history.append(SessionTurn(role="user", message=safe_user_message))
        save_message(session_id=session.session_id, role="user", content=safe_user_message)

        context_monitor = ContextMonitor()
        await asyncio.to_thread(context_monitor.refresh_windows)
        monitor_token = set_monitor(context_monitor)
        try:
            # 2. History Summarization & Loading
            t_start = time.perf_counter()
            summary = summarize_history(chat_request.session_id, token_limit=2000)
            history = load_history(chat_request.session_id, limit=10)
            timings["history_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
            print(f"[TIMING] History load/summarize took: {timings['history_ms']:.2f} ms")
    
            # 3. Cache Check
            t_start = time.perf_counter()
            cached = None
            timings["cache_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
            print(f"[TIMING] Cache check took: {timings['cache_ms']:.2f} ms")

            if cached:
                print("CACHE HIT")
                langfuse.update_current_span(metadata={"cache_status": "hit"})
                safe_answer = cached["answer"]
                structure = cached["structure"]
                chunk_ids = cached["chunk_ids"]
                tool_result = cached["tool_result"]
                pipeline = cached.get("pipeline", {})
                tool_calls = pipeline.get("tool_calls", [])
                raw_answer = cached.get("raw_answer", safe_answer)
                input_tokens = output_tokens = 0
                latency_ms = 0
            else:
                instructions = (
                    "You are an insurance assistant for plans, benefits, "
                    "coverage, costs, and claims. Follow the supplied "
                    "retrieval or clarification context. Do not invent facts. "
                    "Historical claim matches do not verify policy coverage; "
                    "historical cost estimates are not guaranteed costs."
                    "Allow follow-ups about the insurance conversation. "
                    "For unrelated requests, briefly explain that you help "
                    "with insurance questions and conversation follow-ups exclusively. "
                )
                print("CACHE MISS")
                langfuse.update_current_span(metadata={"cache_status": "miss"})

                # 4. Retrieve directly, with one retry and a total timeout.
                t_start = time.perf_counter()

                try:
                    (
                        context,
                        structure,
                        chunk_ids,
                        tool_result,
                        tool_calls,
                        retrieval,
                    ) = await asyncio.wait_for(
                        verified_retrieve(
                            safe_user_message,
                            history=history,
                            instructions=instructions,
                        ),
                        timeout=120,
                    )
                except asyncio.TimeoutError:
                    # The call may have loaded a different model since request startup.
                    await asyncio.to_thread(context_monitor.refresh_windows)
                    raise HTTPException(
                        status_code=504,
                        detail={
                            "code": "RETRIEVAL_TIMEOUT",
                            "message": "The retrieval workflow timed out.",
                            "context_warning": context_monitor.warning(),
                        },
                    )
                except Exception as error:
                    print(f"RETRIEVAL ERROR | {error}")
                    context = (
                        "Information retrieval failed. Tell the user you "
                        "cannot access the information right now."
                    )
                    structure = "fallback"
                    chunk_ids = []
                    tool_result = None
                    tool_calls = []
                    retrieval = []

                timings["retrieval_ms"] = round(
                    (time.perf_counter() - t_start) * 1000, 2
                )



                prompt = build_system_prompt(safe_user_message, context, instructions)

                print("[DEBUG] BEFORE LLM")
                t_start = time.perf_counter()
                (
                    full_answer,
                    input_tokens,
                    output_tokens,
                    latency_ms,
                ) = call_llm_with_tracing(
                    safe_user_message,
                    context,
                    instructions
                )
                print("[DEBUG] AFTER LLM")

                raw_answer = full_answer
                t_start = time.perf_counter()
                safe_answer = validate_output(raw_answer)
                timings["output_guardrail_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
                if not isinstance(safe_answer, str):
                    safe_answer = str(safe_answer or "")

                used_private_tool = any(
                    call.get("name") in PRIVATE_TOOLS
                    for call in tool_calls
                )
                cacheable = False
                pipeline = {
                    "route": structure,
                    "instructions": instructions,
                    "context": context,
                    "prompt": prompt,
                    "retrieval": retrieval,
                    "tool_calls": tool_calls,
                }

                if cacheable:
                    save_cached_response(
                        chat_request.message,
                        {
                            "answer": safe_answer,
                            "structure": structure,
                            "chunk_ids": chunk_ids,
                            "tool_result": tool_result,
                            "raw_answer": raw_answer,
                            "pipeline": pipeline,
                        }
                    )
                    print("CACHE SET")

                save_token_usage(
                    session.session_id,
                    input_tokens,
                    output_tokens,
                    estimate_cost(input_tokens, output_tokens),
                )
    
            if not isinstance(safe_answer, str):
                safe_answer = str(safe_answer or "")
    
            langfuse.update_current_span(
                output=safe_answer
            )
    
            assistant_turn = SessionTurn(role="assistant", message=safe_answer)
            session.history.append(assistant_turn)
            save_message(
                session_id=session.session_id,
                role="assistant",
                content=redact_pii(
                    safe_answer,
                    allow_public_contacts=True,
                ),
            )
    
            timings["total_ms"] = round((time.perf_counter() - overall_start) * 1000, 2)
            diagnostics = {
                "model": {"name": "qwen3:8b", "temperature": 0, "num_predict": 200},
                "cache": "hit" if cached else "miss",
                "history": {"loaded": history, "summary": summary},
                "guardrails": {
                    "input_allowed": True,
                    "input_redacted": contains_pii,
                    "output_changed": raw_answer != safe_answer,
                    "raw_answer": raw_answer,
                },
                "tokens": {"input": input_tokens, "output": output_tokens},
                "timings_ms": {**timings, "llm_ms": round(latency_ms, 2)},
                **pipeline,
            }

            if not context_monitor.window("qwen3:8b"):
                await asyncio.to_thread(context_monitor.refresh_windows)

            response = ChatResponse(
                session_id=session.session_id,
                member_id=session.member_id,
                answer=safe_answer,
                structure=structure or "",
                chunk_ids=chunk_ids or [],
                tool_result=(
                    tool_result
                    if isinstance(tool_result, dict)
                    else {"value": tool_result} if tool_result is not None else None
                ),
                tool_calls=tool_calls,
                history=session.history,
                diagnostics=diagnostics if eval_diagnostics else None,
                context_warning=context_monitor.warning(),
            )
    
            def stream():
                yield f"data: {response.model_dump_json()}\n\n"
    
            print(f"[TIMING] Total request completed in: {timings['total_ms']:.2f} ms\n")
            return StreamingResponse(stream(), media_type="text/event-stream")
    
        except HTTPException as he:
            raise he
        except Exception as error:
            print(f"[ERROR] Chat endpoint exception after {(time.perf_counter() - overall_start)*1000:.2f} ms:", error)
            raise HTTPException(status_code=500, detail=str(error))
        finally:
            reset_monitor(monitor_token)
 
 
 
 
@app.get("/history/{session_id}", response_model=SessionState)
def history(session_id: int):
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )
    print("Session History:", session.history)
 
    return session
