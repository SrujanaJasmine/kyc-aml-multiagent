"""
state.py
========
Pydantic schemas for the graph's shared state and for the state passed to a single
worker. The `operator.add` reducers are what let parallel workers merge their
results instead of overwriting each other.
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
        "contain its own 'type' key ('KYC' | 'AML' | 'CDA') plus "
        "whatever ID/question that specific agent needs."
    )
    completed_agents: Annotated[list[dict], operator.add] = Field(
        default_factory=list,
        description="Accumulated structured outputs from each specialist "
        "worker agent (KYC/AML/CDA). The operator.add reducer lets results "
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
    """State passed to a single KYC / AML / CDA worker via Send."""

    type: Literal["KYC", "AML", "CDA"] = Field(
        description="Which specialist this item is routed to."
    )
    input: dict = Field(
        description="The specific input payload required for this assessment type."
    )
    completed_agents: Annotated[list[dict], operator.add] = Field(
        default_factory=list,
        description="This worker's own output. Merges back into GraphState.completed_agents.",
    )