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
        "Check whether a matching historical claim exists for a procedure "
        "under a plan. It does not guarantee policy coverage "
        "or require the historical claim to be approved."
    ),
)
def check_coverage(plan_id: str, procedure: str) -> dict[str, Any]:
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            """
            SELECT COUNT(*) > 0
            FROM claims
            WHERE plan_id = ?
              AND procedure = ?
            """,
            (plan_id, procedure),
        )
        exists = bool(cursor.fetchone()[0])

    return {
        "plan_id": plan_id,
        "procedure": procedure,
        "covered": exists,
    }


@mcp.tool(
    name="get_claim_status",
    description=(
        "Get a claim's status, claim amount, and filing date "
        "using its claim ID, such as C1003."
    ),
)
def get_claim_status(claim_id: str) -> dict[str, Any]:
    claim_id = claim_id.strip().upper()

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT claim_id, status, procedure, claim_amount, date_filed
            FROM claims
            WHERE UPPER(claim_id) = ?
            """,
            (claim_id,),
        ).fetchone()

    if row is None:
        return {
            "claim_id": claim_id,
            "status": None,
            "procedure": None,
            "claim_amount": None,
            "date_filed": None,
            "found": False,
        }

    return dict(row)




@mcp.tool(
    name="get_plan_details",
    description=(
        "Get plan details, including premium, deductible, and copay "
        "percentage. The plan_id argument accepts a plan ID such as "
        "P102 or a plan name such as Silver HMO."
    ),
)
def get_plan_details(plan_id: str) -> dict[str, Any]:
    normalized_plan = plan_id.replace(" ", "").lower()

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            """
            SELECT
                plan_id,
                plan_name,
                monthly_premium,
                annual_deductible,
                copay_pct,
                coverage_type,
                network_tier
            FROM plans
            WHERE LOWER(plan_id) = ?
               OR REPLACE(LOWER(plan_name), ' ', '') = ?
            """,
            (plan_id.lower(), normalized_plan),
        )
        info = cursor.fetchone()

    if info is None:
        raise ValueError(f"No plan found for identifier: {plan_id}")

    return {
        "plan_id": info[0],
        "plan_name": info[1],
        "monthly_premium": info[2],
        "annual_deductible": info[3],
        "copay_pct": info[4],
        "coverage_type": info[5],
        "network_tier": info[6],
    }




@mcp.tool(
    name="estimate_out_of_pocket_cost",
    description=(
        "Estimate out-of-pocket cost using a historical claim amount "
        "and the plan's copay percentage. Requires a procedure and "
        "plan ID. This is a historical estimate, not a coverage guarantee "
        "or a calculation of remaining deductible."
    ),
)
def estimate_out_of_pocket_cost(
    procedure: str,
    plan_id: str,
) -> dict[str, Any]:
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            """
            SELECT
                claims.claim_amount * plans.copay_pct / 100.0
                    AS estimated_out_of_pocket
            FROM claims
            JOIN plans ON claims.plan_id = plans.plan_id
            WHERE claims.plan_id = ?
              AND claims.procedure = ?
            """,
            (plan_id, procedure),
        )
        row = cursor.fetchone()

    return {
        "procedure": procedure,
        "plan_id": plan_id,
        "estimated_cost": row[0] if row else None,
    }





if __name__ == "__main__":
    print("STARTING MCP SERVER", file=sys.stderr)
    mcp.run()