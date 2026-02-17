#!/usr/bin/env python3
"""AI-driven agentic workflow for oncology clinical trial orchestration.

CLINICAL CONTEXT
================
Modern oncology clinical trials involve dozens of concurrent activities:
treatment planning, literature review, regulatory compliance checking,
adverse event monitoring, and data management.  AI agents can assist human
clinicians by automating routine tasks, surfacing relevant literature, and
flagging compliance issues — all while maintaining a tamper-evident audit
trail required by FDA 21 CFR Part 11.  This example demonstrates the
unified agent interface for orchestrating multiple clinical trial agents
using a backend-agnostic abstraction layer.

USE CASES COVERED
=================
1. Treatment planning agent that queries digital twin simulations and
   recommends optimal treatment protocols based on predicted outcomes.
2. Literature review agent that searches clinical databases and summarises
   recent evidence relevant to the patient's tumor profile.
3. Compliance checking agent that validates trial protocols against
   regulatory requirements and flags potential non-conformances.
4. Multi-agent orchestration with shared tool registry, parallel execution,
   and coordinated decision-making with human-in-the-loop review.
5. Audit trail generation with hash-chained integrity verification and
   PHI detection for all agent inputs and outputs.

FRAMEWORK REQUIREMENTS
======================
Required:
    - numpy >= 1.24.0
    - scipy >= 1.11.0

Optional:
    - crewai >= 0.30.0  (CrewAI agent framework)
      https://github.com/joaomdmoura/crewAI
    - langgraph >= 0.1.0  (LangGraph state machine framework)
      https://github.com/langchain-ai/langgraph
    - autogen >= 0.2.0  (AutoGen multi-agent framework)
      https://github.com/microsoft/autogen
    - anthropic >= 0.25.0  (Anthropic Claude API client)
      https://docs.anthropic.com/

HARDWARE REQUIREMENTS
=====================
    - CPU: Any modern x86-64 or ARM64 processor.
    - RAM: >= 4 GB for agent orchestration.
    - GPU: Not required (LLM calls are API-based).

REFERENCES
==========
    - Topol, "High-performance medicine: the convergence of human and
      artificial intelligence", Nature Medicine, 2019.
      https://doi.org/10.1038/s41591-018-0300-7
    - FDA 21 CFR Part 11 — Electronic Records; Electronic Signatures.
      https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-11
    - Wu et al., "AutoGen: Enabling Next-Gen LLM Applications via Multi-
      Agent Conversation", 2023.  https://arxiv.org/abs/2308.08155
    - CrewAI Documentation.  https://docs.crewai.com/

DISCLAIMER
==========
RESEARCH USE ONLY.  This software is provided for research and educational
purposes only.  It has NOT been validated for clinical use, is NOT approved
by the FDA or any other regulatory body, and MUST NOT be used to make
clinical decisions or direct patient care.  All agent outputs must be
reviewed by qualified clinical professionals before any action is taken.

LICENSE: MIT
VERSION: 0.5.0
LAST UPDATED: 2026-02-17
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------
try:
    from crewai import Agent as CrewAgent  # type: ignore[import-untyped]

    HAS_CREWAI = True
except ImportError:
    CrewAgent = None  # type: ignore[assignment,misc]
    HAS_CREWAI = False

try:
    from langgraph.graph import StateGraph  # type: ignore[import-untyped]

    HAS_LANGGRAPH = True
except ImportError:
    StateGraph = None  # type: ignore[assignment,misc]
    HAS_LANGGRAPH = False

try:
    from autogen import AssistantAgent  # type: ignore[import-untyped]

    HAS_AUTOGEN = True
except ImportError:
    AssistantAgent = None  # type: ignore[assignment,misc]
    HAS_AUTOGEN = False

try:
    import anthropic  # type: ignore[import-untyped]

    HAS_ANTHROPIC = True
except ImportError:
    anthropic = None  # type: ignore[assignment]
    HAS_ANTHROPIC = False

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unification.agentic_generative_ai.unified_agent_interface import (
    AgentBackend,
    AgentConfig,
    AgentResponse,
    AgentRole,
    AgentState,
    AuditEventType,
    AuditTrailManager,
    PHIDetector,
    Tool,
    ToolCategory,
    ToolParameter,
    ToolRegistry,
    UnifiedAgent,
    create_standard_oncology_tools,
)

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Section 1 — Configuration: Enums + Dataclasses
# ============================================================================


class WorkflowPhase(str, Enum):
    """Phases of the agentic clinical workflow."""

    INITIALIZATION = "initialization"
    TREATMENT_PLANNING = "treatment_planning"
    LITERATURE_REVIEW = "literature_review"
    COMPLIANCE_CHECK = "compliance_check"
    SYNTHESIS = "synthesis"
    HUMAN_REVIEW = "human_review"
    FINALIZATION = "finalization"


class AgentType(str, Enum):
    """Types of clinical trial agents in the workflow."""

    TREATMENT_PLANNER = "treatment_planner"
    LITERATURE_REVIEWER = "literature_reviewer"
    COMPLIANCE_CHECKER = "compliance_checker"
    WORKFLOW_COORDINATOR = "workflow_coordinator"


class DecisionConfidence(str, Enum):
    """Confidence level for agent decisions."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    REQUIRES_REVIEW = "requires_review"


class ReviewStatus(str, Enum):
    """Status of human review for agent outputs."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"


@dataclass
class ClinicalContext:
    """Clinical context for the agentic workflow.

    Attributes:
        trial_id: Clinical trial identifier.
        patient_id: De-identified patient identifier.
        tumor_type: Primary tumor type.
        tumor_location: Anatomical tumor location.
        stage: Cancer stage (e.g., "IIIA").
        biomarkers: Relevant biomarker values.
        prior_treatments: List of prior treatments.
        comorbidities: List of comorbidities.
        performance_status: ECOG performance status (0-4).
    """

    trial_id: str = "AGENTIC_TRIAL_001"
    patient_id: str = "PT-AGENT-001"
    tumor_type: str = "non_small_cell_lung_cancer"
    tumor_location: str = "lung"
    stage: str = "IIIA"
    biomarkers: dict[str, float] = field(
        default_factory=lambda: {
            "pdl1_expression": 0.60,
            "egfr_mutation": 0.0,
            "alk_rearrangement": 0.0,
            "kras_g12c": 1.0,
            "tmb_score": 12.5,
        }
    )
    prior_treatments: list[str] = field(default_factory=list)
    comorbidities: list[str] = field(default_factory=lambda: ["hypertension", "type_2_diabetes"])
    performance_status: int = 1


@dataclass
class AgentDecision:
    """A structured decision from an agent.

    Attributes:
        agent_type: Type of agent that made the decision.
        decision_id: Unique decision identifier.
        content: Decision content text.
        confidence: Confidence level.
        evidence: Supporting evidence references.
        tools_used: Tools invoked during reasoning.
        requires_human_review: Whether human review is mandatory.
        review_status: Current review status.
        timestamp: Decision timestamp.
    """

    agent_type: AgentType = AgentType.TREATMENT_PLANNER
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    confidence: DecisionConfidence = DecisionConfidence.MEDIUM
    evidence: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    requires_human_review: bool = True
    review_status: ReviewStatus = ReviewStatus.PENDING
    timestamp: float = field(default_factory=time.time)


@dataclass
class WorkflowState:
    """Current state of the agentic workflow.

    Attributes:
        workflow_id: Unique workflow identifier.
        phase: Current workflow phase.
        clinical_context: Clinical context for the session.
        decisions: All agent decisions made so far.
        audit_events_count: Number of audit events recorded.
        start_time: Workflow start timestamp.
        completed: Whether the workflow has completed.
    """

    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    phase: WorkflowPhase = WorkflowPhase.INITIALIZATION
    clinical_context: ClinicalContext = field(default_factory=ClinicalContext)
    decisions: list[AgentDecision] = field(default_factory=list)
    audit_events_count: int = 0
    start_time: float = field(default_factory=time.time)
    completed: bool = False


@dataclass
class WorkflowReport:
    """Final report from the agentic clinical workflow.

    Attributes:
        workflow_id: Unique workflow identifier.
        clinical_context: Clinical context used.
        decisions: All agent decisions.
        synthesis: Combined synthesis text.
        total_tools_invoked: Total tool invocations across all agents.
        total_audit_events: Total audit events recorded.
        audit_integrity: Whether audit hash chain passed verification.
        total_duration_s: Total workflow duration.
        phases_completed: List of completed phases.
    """

    workflow_id: str = ""
    clinical_context: ClinicalContext = field(default_factory=ClinicalContext)
    decisions: list[AgentDecision] = field(default_factory=list)
    synthesis: str = ""
    total_tools_invoked: int = 0
    total_audit_events: int = 0
    audit_integrity: bool = False
    total_duration_s: float = 0.0
    phases_completed: list[str] = field(default_factory=list)


# ============================================================================
# Section 2 — Core Implementation: Clinical Trial Agent Factories
# ============================================================================


def _create_treatment_planning_tools() -> list[Tool]:
    """Create tools specific to treatment planning agents."""
    return [
        Tool(
            name="query_digital_twin",
            description="Query a patient digital twin for treatment simulation results",
            category=ToolCategory.SIMULATION,
            parameters=[
                ToolParameter(name="patient_id", param_type="string", description="De-identified patient ID"),
                ToolParameter(name="treatment_type", param_type="string", description="Treatment modality"),
            ],
        ),
        Tool(
            name="compare_protocols",
            description="Compare multiple treatment protocols based on digital twin predictions",
            category=ToolCategory.ANALYTICS,
            parameters=[
                ToolParameter(name="protocol_ids", param_type="array", description="List of protocol identifiers"),
            ],
        ),
        Tool(
            name="assess_toxicity_risk",
            description="Assess predicted toxicity risk for a given treatment plan",
            category=ToolCategory.SAFETY,
            parameters=[
                ToolParameter(name="treatment_plan", param_type="object", description="Treatment plan specification"),
                ToolParameter(name="patient_profile", param_type="object", description="Patient clinical profile"),
            ],
            requires_human_approval=True,
        ),
    ]


def _create_literature_review_tools() -> list[Tool]:
    """Create tools specific to literature review agents."""
    return [
        Tool(
            name="search_pubmed",
            description="Search PubMed for oncology literature",
            category=ToolCategory.LITERATURE,
            parameters=[
                ToolParameter(name="query", param_type="string", description="Search query"),
                ToolParameter(name="max_results", param_type="integer", description="Maximum results", default=10),
                ToolParameter(name="date_from", param_type="string", description="Start date filter", required=False),
            ],
        ),
        Tool(
            name="search_clinical_trials",
            description="Search ClinicalTrials.gov for active trials",
            category=ToolCategory.LITERATURE,
            parameters=[
                ToolParameter(name="condition", param_type="string", description="Disease condition"),
                ToolParameter(name="intervention", param_type="string", description="Treatment intervention"),
            ],
        ),
        Tool(
            name="summarize_evidence",
            description="Summarize a collection of literature evidence for clinical decision support",
            category=ToolCategory.ANALYTICS,
            parameters=[
                ToolParameter(name="article_ids", param_type="array", description="List of PubMed IDs"),
                ToolParameter(name="focus_topic", param_type="string", description="Topic to focus summary on"),
            ],
        ),
    ]


def _create_compliance_tools() -> list[Tool]:
    """Create tools specific to compliance checking agents."""
    return [
        Tool(
            name="check_protocol_compliance",
            description="Check treatment protocol against regulatory requirements",
            category=ToolCategory.REGULATORY,
            parameters=[
                ToolParameter(name="protocol", param_type="object", description="Treatment protocol"),
                ToolParameter(
                    name="framework",
                    param_type="string",
                    description="Regulatory framework",
                    enum_values=["FDA", "EMA", "ICH_GCP"],
                ),
            ],
        ),
        Tool(
            name="validate_consent_status",
            description="Validate patient consent status for trial procedures",
            category=ToolCategory.REGULATORY,
            parameters=[
                ToolParameter(name="patient_id", param_type="string", description="Patient ID"),
                ToolParameter(name="procedure", param_type="string", description="Proposed procedure"),
            ],
        ),
        Tool(
            name="generate_compliance_report",
            description="Generate a regulatory compliance report for the trial protocol",
            category=ToolCategory.DOCUMENTATION,
            parameters=[
                ToolParameter(name="trial_id", param_type="string", description="Trial identifier"),
                ToolParameter(name="scope", param_type="string", description="Report scope"),
            ],
        ),
    ]


def create_agent(
    agent_type: AgentType,
    backend: AgentBackend = AgentBackend.CUSTOM,
) -> UnifiedAgent:
    """Factory function to create a clinical trial agent.

    Args:
        agent_type: Type of clinical agent to create.
        backend: Agent framework backend.

    Returns:
        Configured and initialized UnifiedAgent.
    """
    configs: dict[AgentType, tuple[str, AgentRole, list[Tool]]] = {
        AgentType.TREATMENT_PLANNER: (
            "TreatmentPlanningAgent",
            AgentRole.ONCOLOGIST,
            _create_treatment_planning_tools(),
        ),
        AgentType.LITERATURE_REVIEWER: (
            "LiteratureReviewAgent",
            AgentRole.LITERATURE_REVIEWER,
            _create_literature_review_tools(),
        ),
        AgentType.COMPLIANCE_CHECKER: (
            "ComplianceCheckAgent",
            AgentRole.REGULATORY_SPECIALIST,
            _create_compliance_tools(),
        ),
    }

    name, role, tools = configs[agent_type]

    config = AgentConfig(
        name=name,
        role=role,
        backend=backend,
        tools=tools,
        phi_detection_enabled=True,
        audit_enabled=True,
        max_iterations=10,
        timeout_s=120.0,
    )

    agent = UnifiedAgent(config)
    agent.initialize()
    logger.info("Created agent: %s (role=%s, tools=%d)", name, role.value, len(tools))
    return agent


# ============================================================================
# Section 3 — Pipeline Orchestration: AgenticWorkflowOrchestrator
# ============================================================================


class AgenticWorkflowOrchestrator:
    """Orchestrates multi-agent clinical trial workflows.

    Manages the lifecycle of clinical trial agents, coordinates their
    execution across workflow phases, and compiles unified reports with
    full audit trails.

    Args:
        clinical_context: Clinical context for the workflow session.
    """

    def __init__(self, clinical_context: ClinicalContext) -> None:
        self._state = WorkflowState(clinical_context=clinical_context)
        self._agents: dict[AgentType, UnifiedAgent] = {}
        self._audit = AuditTrailManager()
        self._tool_registry = ToolRegistry()
        self._phi_detector = PHIDetector()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, backend: AgentBackend = AgentBackend.CUSTOM) -> None:
        """Initialize all agents and the shared tool registry.

        Args:
            backend: Agent framework backend to use.
        """
        self._state.phase = WorkflowPhase.INITIALIZATION
        logger.info("Initializing agentic workflow: %s", self._state.workflow_id)

        # Register standard tools in the shared registry
        for tool in create_standard_oncology_tools():
            self._tool_registry.register(tool)

        # Create specialized agents
        agent_types = [AgentType.TREATMENT_PLANNER, AgentType.LITERATURE_REVIEWER, AgentType.COMPLIANCE_CHECKER]

        for agent_type in agent_types:
            agent = create_agent(agent_type, backend=backend)
            self._agents[agent_type] = agent

        self._audit.record(
            AuditEventType.AGENT_STARTED,
            agent_id=self._state.workflow_id,
            details={"agents": [at.value for at in agent_types], "backend": backend.value},
        )

        logger.info("Workflow initialized with %d agents", len(self._agents))

    # ------------------------------------------------------------------
    # Treatment Planning Phase
    # ------------------------------------------------------------------

    def run_treatment_planning(self) -> AgentDecision:
        """Execute the treatment planning phase.

        Returns:
            AgentDecision from the treatment planning agent.
        """
        self._state.phase = WorkflowPhase.TREATMENT_PLANNING
        logger.info("Phase: Treatment Planning")

        agent = self._agents[AgentType.TREATMENT_PLANNER]
        ctx = self._state.clinical_context

        prompt = (
            f"Evaluate treatment options for a patient with {ctx.tumor_type} "
            f"(stage {ctx.stage}) located in the {ctx.tumor_location}. "
            f"Key biomarkers: PD-L1={ctx.biomarkers.get('pdl1_expression', 0):.0%}, "
            f"TMB={ctx.biomarkers.get('tmb_score', 0):.1f}. "
            f"ECOG PS={ctx.performance_status}. "
            f"Comorbidities: {', '.join(ctx.comorbidities)}. "
            f"Please query the digital twin and compare available protocols."
        )

        response = agent.execute(prompt, context={"clinical_context": ctx.__dict__})

        decision = AgentDecision(
            agent_type=AgentType.TREATMENT_PLANNER,
            content=response.content,
            confidence=DecisionConfidence.MEDIUM,
            tools_used=[tc.get("tool_name", "") for tc in response.tool_calls],
            evidence=[
                "Digital twin simulation results",
                "Protocol comparison analysis",
            ],
            requires_human_review=True,
        )

        self._state.decisions.append(decision)
        logger.info(
            "Treatment planning decision: confidence=%s, tools=%d", decision.confidence.value, len(decision.tools_used)
        )
        return decision

    # ------------------------------------------------------------------
    # Literature Review Phase
    # ------------------------------------------------------------------

    def run_literature_review(self) -> AgentDecision:
        """Execute the literature review phase.

        Returns:
            AgentDecision from the literature review agent.
        """
        self._state.phase = WorkflowPhase.LITERATURE_REVIEW
        logger.info("Phase: Literature Review")

        agent = self._agents[AgentType.LITERATURE_REVIEWER]
        ctx = self._state.clinical_context

        biomarker_keys = ", ".join(ctx.biomarkers.keys())
        prompt = (
            f"Search for recent clinical evidence on {ctx.tumor_type} treatment "
            f"in patients with the following biomarker profile: {biomarker_keys}. "
            f"Focus on immunotherapy combinations and targeted therapy options. "
            f"Summarize the top evidence for clinical decision support."
        )

        response = agent.execute(prompt)

        decision = AgentDecision(
            agent_type=AgentType.LITERATURE_REVIEWER,
            content=response.content,
            confidence=DecisionConfidence.MEDIUM,
            tools_used=[tc.get("tool_name", "") for tc in response.tool_calls],
            evidence=[
                "PubMed literature search",
                "ClinicalTrials.gov active trials",
                "Evidence summary",
            ],
            requires_human_review=False,
        )

        self._state.decisions.append(decision)
        logger.info("Literature review decision: tools=%d", len(decision.tools_used))
        return decision

    # ------------------------------------------------------------------
    # Compliance Check Phase
    # ------------------------------------------------------------------

    def run_compliance_check(self) -> AgentDecision:
        """Execute the regulatory compliance checking phase.

        Returns:
            AgentDecision from the compliance checking agent.
        """
        self._state.phase = WorkflowPhase.COMPLIANCE_CHECK
        logger.info("Phase: Compliance Check")

        agent = self._agents[AgentType.COMPLIANCE_CHECKER]
        ctx = self._state.clinical_context

        prompt = (
            f"Validate the clinical trial protocol for trial {ctx.trial_id} "
            f"against FDA and ICH-GCP regulatory requirements. "
            f"Check consent status for patient {ctx.patient_id} and "
            f"generate a compliance report covering all proposed procedures."
        )

        response = agent.execute(prompt)

        decision = AgentDecision(
            agent_type=AgentType.COMPLIANCE_CHECKER,
            content=response.content,
            confidence=DecisionConfidence.HIGH,
            tools_used=[tc.get("tool_name", "") for tc in response.tool_calls],
            evidence=[
                "FDA regulatory requirements check",
                "ICH-GCP compliance verification",
                "Patient consent validation",
            ],
            requires_human_review=True,
        )

        self._state.decisions.append(decision)
        logger.info("Compliance check decision: confidence=%s", decision.confidence.value)
        return decision

    # ------------------------------------------------------------------
    # Synthesis Phase
    # ------------------------------------------------------------------

    def run_synthesis(self) -> str:
        """Synthesize decisions from all agents into a unified recommendation.

        Returns:
            Synthesis text combining all agent outputs.
        """
        self._state.phase = WorkflowPhase.SYNTHESIS
        logger.info("Phase: Synthesis")

        lines = [
            "CLINICAL WORKFLOW SYNTHESIS",
            f"Trial: {self._state.clinical_context.trial_id}",
            f"Patient: {self._state.clinical_context.patient_id}",
            f"Tumor: {self._state.clinical_context.tumor_type} (Stage {self._state.clinical_context.stage})",
            "",
        ]

        for decision in self._state.decisions:
            lines.append(f"--- {decision.agent_type.value.upper()} ---")
            lines.append(f"Confidence: {decision.confidence.value}")
            lines.append(f"Decision: {decision.content}")
            lines.append(f"Evidence: {', '.join(decision.evidence)}")
            lines.append(f"Review Required: {decision.requires_human_review}")
            lines.append("")

        lines.append("--- WORKFLOW NOTES ---")
        lines.append("All agent outputs must be reviewed by qualified clinical staff.")
        lines.append("This synthesis is generated for decision support only.")

        synthesis = "\n".join(lines)
        logger.info("Synthesis generated: %d decisions combined", len(self._state.decisions))
        return synthesis

    # ------------------------------------------------------------------
    # Human Review Simulation
    # ------------------------------------------------------------------

    def simulate_human_review(self) -> None:
        """Simulate a human review step for decisions requiring approval.

        In production, this would involve a real human reviewer interface.
        """
        self._state.phase = WorkflowPhase.HUMAN_REVIEW
        logger.info("Phase: Human Review (simulated)")

        for decision in self._state.decisions:
            if decision.requires_human_review:
                decision.review_status = ReviewStatus.APPROVED
                self._audit.record(
                    AuditEventType.HUMAN_REVIEW_COMPLETED,
                    agent_id=self._state.workflow_id,
                    details={
                        "decision_id": decision.decision_id,
                        "agent_type": decision.agent_type.value,
                        "review_status": decision.review_status.value,
                    },
                    user_identity="dr_reviewer_simulated",
                )
                logger.info(
                    "Decision %s (%s): %s",
                    decision.decision_id[:8],
                    decision.agent_type.value,
                    decision.review_status.value,
                )

    # ------------------------------------------------------------------
    # Full Workflow Execution
    # ------------------------------------------------------------------

    def run_full_workflow(self) -> WorkflowReport:
        """Execute the complete agentic clinical workflow.

        Runs all phases in sequence: initialization, treatment planning,
        literature review, compliance check, synthesis, and human review.

        Returns:
            WorkflowReport with all results and audit trail.
        """
        start = time.time()
        logger.info("Starting full agentic workflow: %s", self._state.workflow_id)

        phases_completed: list[str] = []

        # Treatment planning
        self.run_treatment_planning()
        phases_completed.append(WorkflowPhase.TREATMENT_PLANNING.value)

        # Literature review
        self.run_literature_review()
        phases_completed.append(WorkflowPhase.LITERATURE_REVIEW.value)

        # Compliance check
        self.run_compliance_check()
        phases_completed.append(WorkflowPhase.COMPLIANCE_CHECK.value)

        # Synthesis
        synthesis = self.run_synthesis()
        phases_completed.append(WorkflowPhase.SYNTHESIS.value)

        # Human review
        self.simulate_human_review()
        phases_completed.append(WorkflowPhase.HUMAN_REVIEW.value)

        # Finalization
        self._state.phase = WorkflowPhase.FINALIZATION
        self._state.completed = True
        phases_completed.append(WorkflowPhase.FINALIZATION.value)

        # Verify audit integrity
        audit_integrity = self._audit.verify_chain()
        total_tools = sum(len(d.tools_used) for d in self._state.decisions)

        # Compile report
        report = WorkflowReport(
            workflow_id=self._state.workflow_id,
            clinical_context=self._state.clinical_context,
            decisions=self._state.decisions,
            synthesis=synthesis,
            total_tools_invoked=total_tools,
            total_audit_events=len(self._audit.events),
            audit_integrity=audit_integrity,
            total_duration_s=time.time() - start,
            phases_completed=phases_completed,
        )

        logger.info(
            "Workflow complete: decisions=%d, tools=%d, audit=%d events (integrity=%s)",
            len(report.decisions),
            report.total_tools_invoked,
            report.total_audit_events,
            "PASS" if report.audit_integrity else "FAIL",
        )
        return report

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Shut down all agents and clean up resources."""
        for agent_type, agent in self._agents.items():
            agent.shutdown()
            logger.info("Agent %s shut down", agent_type.value)
        self._agents.clear()


# ============================================================================
# Section 4 — Demonstration
# ============================================================================


def _print_workflow_report(report: WorkflowReport) -> None:
    """Pretty-print an agentic workflow report."""
    print("\n" + "=" * 80)
    print("  AGENTIC CLINICAL WORKFLOW REPORT")
    print("=" * 80)
    print(f"  Workflow ID:     {report.workflow_id}")
    print(f"  Trial:           {report.clinical_context.trial_id}")
    print(f"  Patient:         {report.clinical_context.patient_id}")
    print(f"  Tumor:           {report.clinical_context.tumor_type}")
    print(f"  Stage:           {report.clinical_context.stage}")
    print(f"  Duration:        {report.total_duration_s:.2f}s")
    print(f"  Tools Invoked:   {report.total_tools_invoked}")
    print(f"  Audit Events:    {report.total_audit_events}")
    print(f"  Audit Integrity: {'PASS' if report.audit_integrity else 'FAIL'}")

    print(f"\n  Phases Completed ({len(report.phases_completed)}):")
    for phase in report.phases_completed:
        print(f"    [DONE] {phase}")

    print(f"\n  Agent Decisions ({len(report.decisions)}):")
    for decision in report.decisions:
        review = f" [{decision.review_status.value}]" if decision.requires_human_review else ""
        print(f"\n    Agent: {decision.agent_type.value}")
        print(f"    Confidence: {decision.confidence.value}{review}")
        print(f"    Tools Used: {', '.join(decision.tools_used) if decision.tools_used else 'none'}")
        print(f"    Evidence: {', '.join(decision.evidence[:3])}")
        content_preview = decision.content[:120] + "..." if len(decision.content) > 120 else decision.content
        print(f"    Output: {content_preview}")

    print("\n  Synthesis Preview:")
    for line in report.synthesis.split("\n")[:10]:
        print(f"    {line}")
    if report.synthesis.count("\n") > 10:
        print(f"    ... ({report.synthesis.count(chr(10)) - 10} more lines)")

    print("\n" + "=" * 80)
    print("  DISCLAIMER: RESEARCH USE ONLY")
    print("=" * 80)


if __name__ == "__main__":
    logger.info("Physical AI Federated Learning — Agentic Clinical Workflow Example v0.5.0")
    logger.info(
        "Optional deps: crewai=%s, langgraph=%s, autogen=%s, anthropic=%s",
        HAS_CREWAI,
        HAS_LANGGRAPH,
        HAS_AUTOGEN,
        HAS_ANTHROPIC,
    )

    # Create clinical context
    context = ClinicalContext(
        trial_id="AGENTIC_NSCLC_001",
        patient_id="PT-NSCLC-042",
        tumor_type="non_small_cell_lung_cancer",
        tumor_location="right_upper_lobe",
        stage="IIIA",
        biomarkers={
            "pdl1_expression": 0.65,
            "egfr_mutation": 0.0,
            "alk_rearrangement": 0.0,
            "kras_g12c": 1.0,
            "tmb_score": 14.2,
        },
        prior_treatments=["carboplatin_paclitaxel_4_cycles"],
        comorbidities=["hypertension", "type_2_diabetes"],
        performance_status=1,
    )

    # Display tool registry
    print("\n--- Tool Registry ---")
    registry = ToolRegistry()
    standard_tools = create_standard_oncology_tools()
    for tool in standard_tools:
        registry.register(tool)
    print(f"  Standard tools registered: {registry.count}")
    print("  Tool format exports: MCP, OpenAI, Anthropic, CrewAI, LangGraph")

    # Show tool format examples
    example_tool = standard_tools[0]
    print(f"\n  Example tool: {example_tool.name}")
    print(f"    MCP format keys:      {list(example_tool.to_mcp_format().keys())}")
    print(f"    OpenAI format keys:   {list(example_tool.to_openai_format().keys())}")
    print(f"    Anthropic format keys: {list(example_tool.to_anthropic_format().keys())}")

    # Run the full workflow
    print("\n--- Running Agentic Clinical Workflow ---")
    orchestrator = AgenticWorkflowOrchestrator(context)
    orchestrator.initialize(backend=AgentBackend.CUSTOM)

    report = orchestrator.run_full_workflow()
    _print_workflow_report(report)

    # Clean up
    orchestrator.shutdown()

    # PHI detection demonstration
    print("\n--- PHI Detection Demo ---")
    phi_detector = PHIDetector()
    safe_text = "Patient PT-042 has stage IIIA NSCLC with PD-L1 60%"
    unsafe_text = "Patient name John Doe, date of birth 01/15/1960, SSN 123-45-6789"
    print(f"  Safe text findings:   {phi_detector.check(safe_text)}")
    print(f"  Unsafe text findings: {phi_detector.check(unsafe_text)}")

    logger.info("Agentic workflow example complete.")
    print("\nDISCLAIMER: RESEARCH USE ONLY. Not for clinical decision-making.")
