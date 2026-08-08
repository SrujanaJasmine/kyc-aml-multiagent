"""
Shared state schemas for the Multi-Agent RAG system
(KYC onboarding / AML screening / Credit Assessment / Explanation / Audit).

Pattern:
  Supervisor (orchestrator) node reads `assessment_input` (a batch of one or more
  items, each tagged with its own assessment type) and uses `Send` to fan each
  item out to the correct specialist worker (KYC / AML / CDA / QUERY).
  Each worker writes into `completed_agents`, which uses the `operator.add`
  reducer so results from parallel Sends merge back into the parent GraphState
  instead of overwriting each other.
  All workers converge on the Explanation agent, then the Audit agent.
"""

import operator
from typing import Annotated, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Graph (Orchestrator) State — shared across the whole run
# ---------------------------------------------------------------------------
class GraphState(BaseModel):
    """Top-level state for the supervisor graph."""

    assessment_type: str = Field(
        description="Overall command flag for the run, e.g. 'KYC', 'AML', 'CDA', "
        "or 'BATCH' if assessment_input contains a mix of types."
    )
    assessment_input: list[dict] = Field(
        description="Batch of assessment items to process. Each dict should "
        "contain its own 'type' key ('KYC' | 'AML' | 'CDA' | 'QUERY') plus "
        "whatever ID/question that specific agent needs."
    )
    completed_agents: Annotated[list[dict], operator.add] = Field(
        default_factory=list,
        description="Accumulated structured outputs from each specialist "
        "worker agent (KYC/AML/CDA/QUERY). The operator.add reducer lets results "
        "from multiple parallel Sends merge instead of clobbering each other.",
    )
    audit_log: Annotated[list[dict], operator.add] = Field(
        default_factory=list,
        description="Accumulated log entries every agent sends to the Audit agent.",
    )
    final_report: str = Field(
        default="",
        description="Final plain-English compliance report produced by the Explanation agent.",
    )


# ---------------------------------------------------------------------------
# Worker State — what an individual specialist agent receives/returns
# ---------------------------------------------------------------------------
class WorkerState(BaseModel):
    """State passed to a single KYC / AML / CDA / QUERY worker via Send."""

    type: Literal["KYC", "AML", "CDA", "QUERY"] = Field(
        description="Which specialist this item is routed to."
    )
    input: dict = Field(
        description="The specific input payload required for this assessment type."
    )
    completed_agents: Annotated[list[dict], operator.add] = Field(
        default_factory=list,
        description="This worker's own output. Merges back into GraphState.completed_agents.",
    )
