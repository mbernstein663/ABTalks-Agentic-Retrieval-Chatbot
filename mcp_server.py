import sqlite3
from pathlib import Path
import sys
from typing import Any
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("Coverage Chatbot")

db_path = Path(__file__).resolve().parent / "data" / "coverage.db"


@mcp.tool(
    name="check_coverage",
    description=(
        "Check whether the available insurance data explicitly establishes "
        "coverage for a procedure under a specific plan."
    )
)
def check_coverage(plan_id: str, procedure: str) -> dict[str, Any]:

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT claim_id, status
            FROM claims
            WHERE plan_id = ?
              AND LOWER(procedure) = LOWER(?)
            """,
            (plan_id, procedure)
        )

        claims = cursor.fetchall()

    return {
        "plan_id": plan_id,
        "procedure": procedure,
        "covered": None,
        "coverage_known": False,
        "historical_claims": [
            {"claim_id": claim_id, "status": status}
            for claim_id, status in claims
        ],
        "reason": (
            "The available data does not define service-specific coverage. "
            "Historical claim status does not prove coverage or exclusion."
        )
    }


@mcp.tool(
    name="get_claim_status",
    description="Check status of a current claim specified by claim ID e.g. C1003"
)
def get_claim_status(claim_id: str) -> dict[str, Any]:

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT status
            FROM claims
            WHERE claim_id = ?
            """,
            (claim_id,)
        )

        row = cursor.fetchone()

    if row is None:
        return {
            "claim_id": claim_id,
            "status": None
        }

    return {
        "claim_id": claim_id,
        "status": row[0]
    }


if __name__ == "__main__":
    print("STARTING MCP SERVER", file=sys.stderr)
    mcp.run()