from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_agent import retrieve, load_history
import sys
from pathlib import Path
import asyncio
import traceback
import os

root = Path("/app")

async def retrieve_with_retry(question, retries=1, timeout=120):
    for attempt in range(retries + 1):
        try:
            return await asyncio.wait_for(
                retrieve(question),
                timeout=timeout
            )

        except asyncio.TimeoutError:
            print(
                f"Retrieval timeout "
                f"(attempt {attempt + 1}/{retries + 1})"
            )

        except Exception as e:
            print(
                f"Retrieval error "
                f"(attempt {attempt + 1}/{retries + 1}): {e}"
            )
            traceback.print_exception(type(e), e, e.__traceback__)

        if attempt < retries:
            await asyncio.sleep(1)

    # fallback path
    return (
        "I'm having trouble accessing that right now. "
        "Please contact member support.",
        "fallback",
        [],
        None
    )

# REMOVED: Duplicate MultiServerMCPClient instantiation and unused load_tools() 
# Tools are natively handled by retrieve() imported from langchain_agent.

class AgentState(TypedDict):
    session_id: int
    question: str
    route: str
    instructions: str
    context: str
    structure: str
    chunk_ids: list
    tool_result: dict | None
    answer: str

def get_question_with_history(state: AgentState):

    question = state["question"]
    session_id = state["session_id"]

    # REMOVED: summarize_history() to prevent DB locking and LLM delays.
    # main.py already summarizes the history before entering the graph.

    history = load_history(
        session_id,
        limit=10
    )

    return f"""
Conversation history:
{history}

Current question:
{question}
"""

# CONVERTED TO ASYNC: Prevents the FastAPI event loop from freezing
def router(state: AgentState):
    question = state["question"].lower()

    claims_keywords = [
        "claim",
        "claim id",
        "claim status",
        "denied",
        "approved",
        "processed",
        "date filed",
        "claim amount",
    ]

    if any(keyword in question for keyword in claims_keywords):
        route = "claims"
    else:
        route = "coverage"

    print("ROUTE:", route)

    return {"route": route}


# -------------------------
# AGENT 2: COVERAGE SPECIALIST
# -------------------------

async def coverage_specialist(state: AgentState):

    question = state["question"]

    instructions = """
You are the Coverage Specialist.

Answer questions about:
- plan coverage
- benefits
- premiums
- deductibles
- copays
- coinsurance
- covered services
- out-of-pocket costs

Use retrieved information and tools.
Do not invent information.
"""

    context, structure, chunk_ids, tool_result = (
        await retrieve_with_retry(question)
    )

    return {
        "instructions": instructions,
        "context": context,
        "structure": structure,
        "chunk_ids": chunk_ids,
        "tool_result": tool_result
    }


# -------------------------
# AGENT 3: CLAIMS SPECIALIST
# -------------------------

async def claims_specialist(state: AgentState):

    question = state["question"]

    instructions = """
You are the Claims Specialist.

Answer questions about:
- claim status
- claim amounts
- denied claims
- processed claims
- claim payments
- claim-specific issues

Use retrieved information and tools.
Do not invent information.
"""

    context, structure, chunk_ids, tool_result = (
        await retrieve_with_retry(question)
    )

    return {
        "instructions": instructions,
        "context": context,
        "structure": structure,
        "chunk_ids": chunk_ids,
        "tool_result": tool_result
    }

def choose_specialist(state: AgentState) -> Literal["coverage_specialist", "claims_specialist"]:

    if state["route"] == "coverage":
        return "coverage_specialist"

    return "claims_specialist"


# -------------------------
# BUILD GRAPH
# -------------------------

builder = StateGraph(AgentState)

builder.add_node("router", router)
builder.add_node("coverage_specialist", coverage_specialist)
builder.add_node("claims_specialist", claims_specialist)

builder.add_edge(START, "router")

builder.add_conditional_edges(
    "router",
    choose_specialist,
    {
        "coverage_specialist": "coverage_specialist",
        "claims_specialist": "claims_specialist",
    }
)

builder.add_edge("coverage_specialist", END)
builder.add_edge("claims_specialist", END)

# This is what FastAPI will import
graph = builder.compile()

async def main():
    result = await graph.ainvoke({
        "session_id": 1,
        "question": "What is the status of claim C1003?"
    })
    print(result)

if __name__ == "__main__":
    asyncio.run(main())