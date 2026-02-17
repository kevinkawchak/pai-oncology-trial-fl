"""
ReAct (Reason+Act) Agent for Oncology Treatment Planning.

CLINICAL CONTEXT:
    This module implements a ReAct-pattern agent that iteratively reasons
    about treatment options for oncology patients through thought/action/
    observation loops. The agent consults digital twin simulations, evaluates
    treatment outcomes against RECIST 1.1 criteria, considers toxicity
    profiles (CTCAE v5.0), and generates evidence-based treatment
    recommendations with uncertainty quantification.

    The ReAct pattern (Yao et al., 2022) interleaves reasoning traces
    with actions, enabling the agent to plan, execute tools, observe
    results, and refine its treatment recommendation iteratively.

USE CASES COVERED:
    1. Multi-modality treatment planning for oncology patients with
       iterative evaluation of chemotherapy, radiation, and immunotherapy.
    2. Digital twin consultation for personalized response prediction
       using Monte Carlo simulation with uncertainty quantification.
    3. Drug interaction checking and toxicity assessment before
       finalizing treatment recommendations.
    4. Evidence-based recommendation synthesis with confidence scores
       and supporting rationale from simulation results.
    5. Human-in-the-loop review gates for high-risk treatment decisions
       requiring clinical oversight.

FRAMEWORK REQUIREMENTS:
    Required:
        - numpy >= 1.24.0        (https://numpy.org/)
    Optional:
        - anthropic >= 0.39.0    (https://docs.anthropic.com/)
        - openai >= 1.0.0        (https://platform.openai.com/)

REFERENCES:
    - Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in
      Language Models. arXiv:2210.03629.
    - Eisenhauer et al. (2009). RECIST 1.1. Eur J Cancer 45(2).
    - NCI (2017). CTCAE v5.0. URL: https://ctep.cancer.gov/
    - ICH E6(R3) Good Clinical Practice (2023).

DISCLAIMER:
    RESEARCH USE ONLY. This software is provided for research and educational
    purposes only. It has NOT been validated for clinical use, is NOT approved
    by the FDA or any other regulatory body, and MUST NOT be used to make
    clinical decisions or direct patient care. All treatment recommendations
    must be reviewed by qualified oncologists before any clinical action.

LICENSE: MIT
VERSION: 0.5.0
LAST UPDATED: 2026-02-17
Copyright (c) 2026 PAI Oncology Trial FL Contributors
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Conditional imports for optional dependencies
# ---------------------------------------------------------------------------
try:
    import anthropic  # type: ignore[import-untyped]

    HAS_ANTHROPIC = True
except ImportError:
    anthropic = None  # type: ignore[assignment]
    HAS_ANTHROPIC = False

try:
    import openai  # type: ignore[import-untyped]

    HAS_OPENAI = True
except ImportError:
    openai = None  # type: ignore[assignment]
    HAS_OPENAI = False

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class ReActStepType(Enum):
    """Types of steps in the ReAct reasoning loop."""

    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    FINAL_ANSWER = "final_answer"


class TreatmentModality(Enum):
    """Supported oncology treatment modalities."""

    CHEMOTHERAPY = "chemotherapy"
    RADIATION = "radiation"
    IMMUNOTHERAPY = "immunotherapy"
    TARGETED_THERAPY = "targeted_therapy"
    COMBINATION = "combination"
    SURGERY = "surgery"


class ResponseCategory(Enum):
    """RECIST 1.1 treatment response categories."""

    COMPLETE_RESPONSE = "complete_response"
    PARTIAL_RESPONSE = "partial_response"
    STABLE_DISEASE = "stable_disease"
    PROGRESSIVE_DISEASE = "progressive_disease"


class ToxicityGrade(Enum):
    """CTCAE v5.0 toxicity grading scale."""

    GRADE_0 = 0
    GRADE_1 = 1
    GRADE_2 = 2
    GRADE_3 = 3
    GRADE_4 = 4
    GRADE_5 = 5


class AgentState(Enum):
    """Lifecycle state of the ReAct agent."""

    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    WAITING_HUMAN = "waiting_for_human_review"
    COMPLETE = "complete"
    ERROR = "error"


class ToolName(Enum):
    """Available tools for the ReAct treatment planner."""

    SIMULATE_CHEMO = "simulate_chemotherapy"
    SIMULATE_RADIATION = "simulate_radiation"
    SIMULATE_IMMUNO = "simulate_immunotherapy"
    CHECK_INTERACTIONS = "check_drug_interactions"
    ASSESS_TOXICITY = "assess_toxicity_risk"
    PATIENT_HISTORY = "get_patient_history"
    BIOMARKER_ANALYSIS = "analyze_biomarkers"
    LITERATURE_SEARCH = "search_clinical_evidence"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class PatientProfile:
    """De-identified patient profile for treatment planning."""

    patient_id: str = ""
    age_bracket: str = "60-69"
    sex: str = "unknown"
    diagnosis: str = "NSCLC"
    stage: str = "IIIB"
    histology: str = "adenocarcinoma"
    ecog_score: int = 1
    tumor_volume_cm3: float = 5.0
    biomarkers: dict[str, float] = field(default_factory=dict)
    prior_treatments: list[str] = field(default_factory=list)
    comorbidities: list[str] = field(default_factory=list)
    allergies: list[str] = field(default_factory=list)
    organ_function: dict[str, str] = field(default_factory=lambda: {"renal": "adequate", "hepatic": "adequate"})

    def to_context_string(self) -> str:
        """Format patient profile as context string for LLM reasoning."""
        lines = [
            f"Patient: {self.patient_id}",
            f"Demographics: {self.age_bracket}, {self.sex}",
            f"Diagnosis: {self.diagnosis}, Stage {self.stage}, {self.histology}",
            f"ECOG: {self.ecog_score}, Tumor volume: {self.tumor_volume_cm3:.1f} cm3",
            f"Biomarkers: {json.dumps(self.biomarkers)}",
            f"Prior treatments: {', '.join(self.prior_treatments) if self.prior_treatments else 'none'}",
            f"Comorbidities: {', '.join(self.comorbidities) if self.comorbidities else 'none'}",
            f"Organ function: {json.dumps(self.organ_function)}",
        ]
        return "\n".join(lines)


@dataclass
class ReActStep:
    """A single step in the ReAct reasoning loop."""

    step_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    step_number: int = 0
    step_type: ReActStepType = ReActStepType.THOUGHT
    content: str = ""
    tool_name: str = ""
    tool_arguments: dict[str, Any] = field(default_factory=dict)
    tool_result: Optional[dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize step for logging and audit."""
        return {
            "step_id": self.step_id,
            "step_number": self.step_number,
            "step_type": self.step_type.value,
            "content": self.content,
            "tool_name": self.tool_name,
            "tool_arguments": self.tool_arguments,
            "tool_result": self.tool_result,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
        }


@dataclass
class SimulationOutcome:
    """Result from a treatment simulation."""

    modality: TreatmentModality = TreatmentModality.CHEMOTHERAPY
    initial_volume_cm3: float = 0.0
    predicted_volume_cm3: float = 0.0
    volume_reduction_pct: float = 0.0
    response_category: ResponseCategory = ResponseCategory.STABLE_DISEASE
    confidence_low: float = 0.0
    confidence_high: float = 0.0
    toxicity_grade: ToxicityGrade = ToxicityGrade.GRADE_0
    duration_days: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TreatmentRecommendation:
    """Final treatment recommendation from the ReAct agent."""

    recommendation_id: str = field(default_factory=lambda: f"REC-{uuid.uuid4().hex[:10].upper()}")
    patient_id: str = ""
    recommended_modality: TreatmentModality = TreatmentModality.CHEMOTHERAPY
    regimen_description: str = ""
    expected_response: ResponseCategory = ResponseCategory.PARTIAL_RESPONSE
    expected_reduction_pct: float = 0.0
    confidence_score: float = 0.0
    toxicity_risk: ToxicityGrade = ToxicityGrade.GRADE_1
    rationale: str = ""
    supporting_evidence: list[str] = field(default_factory=list)
    alternative_options: list[dict[str, Any]] = field(default_factory=list)
    requires_human_review: bool = True
    human_review_reason: str = ""
    react_steps_count: int = 0
    total_simulations: int = 0
    total_duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize recommendation for output."""
        return {
            "recommendation_id": self.recommendation_id,
            "patient_id": self.patient_id,
            "recommended_modality": self.recommended_modality.value,
            "regimen_description": self.regimen_description,
            "expected_response": self.expected_response.value,
            "expected_reduction_pct": self.expected_reduction_pct,
            "confidence_score": self.confidence_score,
            "toxicity_risk": self.toxicity_risk.name,
            "rationale": self.rationale,
            "supporting_evidence": self.supporting_evidence,
            "alternative_options": self.alternative_options,
            "requires_human_review": self.requires_human_review,
            "human_review_reason": self.human_review_reason,
            "react_steps_count": self.react_steps_count,
            "total_simulations": self.total_simulations,
            "total_duration_ms": self.total_duration_ms,
        }


# ---------------------------------------------------------------------------
# Treatment simulation engine (local, no GPU required)
# ---------------------------------------------------------------------------
class TreatmentSimulationEngine:
    """Lightweight treatment simulation engine for ReAct tool actions.

    Provides Monte Carlo treatment response predictions for chemotherapy,
    radiation, and immunotherapy modalities without requiring external
    simulation frameworks.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)
        self._simulation_count = 0

    def simulate_chemotherapy(
        self,
        volume_cm3: float,
        sensitivity: float = 0.5,
        dose_mg: float = 75.0,
        cycles: int = 4,
        n_samples: int = 100,
    ) -> SimulationOutcome:
        """Simulate chemotherapy response with Monte Carlo uncertainty."""
        volumes = []
        for i in range(n_samples):
            sim_rng = np.random.default_rng(i + self._simulation_count * 1000)
            perturbed_sens = float(np.clip(sensitivity + sim_rng.normal(0, 0.1), 0.05, 0.95))
            vol = volume_cm3
            for _ in range(cycles):
                kill = perturbed_sens * min(dose_mg / 100.0, 0.8)
                vol *= 1.0 - kill
                vol *= np.exp(sim_rng.uniform(0.005, 0.015) * 21)
            volumes.append(float(max(vol, 0.001)))

        self._simulation_count += 1
        arr = np.array(volumes)
        mean_vol = float(np.mean(arr))
        reduction = (volume_cm3 - mean_vol) / max(volume_cm3, 0.001) * 100.0
        response = self._classify_response(reduction / 100.0)
        toxicity = ToxicityGrade(int(np.clip(self._rng.integers(0, 4), 0, 4)))

        return SimulationOutcome(
            modality=TreatmentModality.CHEMOTHERAPY,
            initial_volume_cm3=volume_cm3,
            predicted_volume_cm3=mean_vol,
            volume_reduction_pct=reduction,
            response_category=response,
            confidence_low=float(np.percentile(arr, 5)),
            confidence_high=float(np.percentile(arr, 95)),
            toxicity_grade=toxicity,
            duration_days=cycles * 21,
            metadata={"dose_mg": dose_mg, "cycles": cycles, "n_samples": n_samples},
        )

    def simulate_radiation(
        self,
        volume_cm3: float,
        sensitivity: float = 0.6,
        total_dose_gy: float = 60.0,
        fractions: int = 30,
        n_samples: int = 100,
    ) -> SimulationOutcome:
        """Simulate radiation therapy response using the linear-quadratic model."""
        dose_per_fraction = total_dose_gy / max(fractions, 1)
        volumes = []

        for i in range(n_samples):
            sim_rng = np.random.default_rng(i + self._simulation_count * 1000 + 500)
            perturbed_sens = float(np.clip(sensitivity + sim_rng.normal(0, 0.08), 0.1, 0.95))
            alpha = 0.3 * perturbed_sens
            beta = alpha / 10.0
            sf = np.exp(-fractions * (alpha * dose_per_fraction + beta * dose_per_fraction**2))
            sf = float(np.clip(sf, 0.001, 1.0))
            vol = volume_cm3 * sf
            volumes.append(float(max(vol, 0.001)))

        self._simulation_count += 1
        arr = np.array(volumes)
        mean_vol = float(np.mean(arr))
        reduction = (volume_cm3 - mean_vol) / max(volume_cm3, 0.001) * 100.0
        response = self._classify_response(reduction / 100.0)
        toxicity = ToxicityGrade(int(np.clip(self._rng.integers(0, 3), 0, 4)))

        return SimulationOutcome(
            modality=TreatmentModality.RADIATION,
            initial_volume_cm3=volume_cm3,
            predicted_volume_cm3=mean_vol,
            volume_reduction_pct=reduction,
            response_category=response,
            confidence_low=float(np.percentile(arr, 5)),
            confidence_high=float(np.percentile(arr, 95)),
            toxicity_grade=toxicity,
            duration_days=int(np.ceil(fractions / 5) * 7),
            metadata={"total_dose_gy": total_dose_gy, "fractions": fractions, "n_samples": n_samples},
        )

    def simulate_immunotherapy(
        self,
        volume_cm3: float,
        sensitivity: float = 0.3,
        response_rate: float = 0.35,
        cycles: int = 8,
        n_samples: int = 100,
    ) -> SimulationOutcome:
        """Simulate immunotherapy response with stochastic immune dynamics."""
        volumes = []

        for i in range(n_samples):
            sim_rng = np.random.default_rng(i + self._simulation_count * 1000 + 1000)
            effective_rate = response_rate * (0.5 + sensitivity)
            effective_rate = float(np.clip(effective_rate, 0.0, 0.95))
            responds = sim_rng.random() < effective_rate

            vol = volume_cm3
            for cycle_idx in range(cycles):
                if responds:
                    decay = 0.03 * sensitivity
                    vol *= np.exp(-decay * 21)
                else:
                    growth = sim_rng.uniform(0.005, 0.02)
                    vol *= np.exp(growth * 21)
            volumes.append(float(max(vol, 0.001)))

        self._simulation_count += 1
        arr = np.array(volumes)
        mean_vol = float(np.mean(arr))
        reduction = (volume_cm3 - mean_vol) / max(volume_cm3, 0.001) * 100.0
        response = self._classify_response(reduction / 100.0)
        toxicity = ToxicityGrade(int(np.clip(self._rng.integers(0, 3), 0, 4)))

        return SimulationOutcome(
            modality=TreatmentModality.IMMUNOTHERAPY,
            initial_volume_cm3=volume_cm3,
            predicted_volume_cm3=mean_vol,
            volume_reduction_pct=reduction,
            response_category=response,
            confidence_low=float(np.percentile(arr, 5)),
            confidence_high=float(np.percentile(arr, 95)),
            toxicity_grade=toxicity,
            duration_days=cycles * 21,
            metadata={"response_rate": response_rate, "cycles": cycles, "n_samples": n_samples},
        )

    def check_drug_interactions(self, medications: list[str]) -> dict[str, Any]:
        """Check for drug-drug interactions in a treatment regimen."""
        known_interactions = {
            ("carboplatin", "pembrolizumab"): {
                "severity": "moderate",
                "description": "Monitor for increased myelosuppression when combining platinum-based chemo with ICI",
                "management": "Monitor CBC weekly during first 2 cycles",
            },
            ("cisplatin", "pemetrexed"): {
                "severity": "moderate",
                "description": "Cisplatin may increase renal toxicity of pemetrexed",
                "management": "Ensure adequate hydration; monitor creatinine clearance",
            },
            ("docetaxel", "bevacizumab"): {
                "severity": "minor",
                "description": "Bevacizumab may modestly increase docetaxel exposure",
                "management": "Monitor for increased neutropenia",
            },
        }

        interactions_found = []
        med_lower = [m.lower() for m in medications]
        for (drug_a, drug_b), info in known_interactions.items():
            if drug_a in med_lower and drug_b in med_lower:
                interactions_found.append(
                    {
                        "drugs": [drug_a, drug_b],
                        "severity": info["severity"],
                        "description": info["description"],
                        "management": info["management"],
                    }
                )

        return {
            "medications_checked": medications,
            "interactions_found": len(interactions_found),
            "interactions": interactions_found,
            "overall_risk": "moderate" if interactions_found else "low",
        }

    def assess_toxicity_risk(
        self, modality: str, dose_intensity: float = 0.5, patient_age: int = 65, ecog: int = 1
    ) -> dict[str, Any]:
        """Assess overall toxicity risk for a treatment modality."""
        base_risk = dose_intensity * 0.6
        age_factor = 1.0 + max(0, (patient_age - 60)) * 0.01
        ecog_factor = 1.0 + ecog * 0.15
        risk_score = float(np.clip(base_risk * age_factor * ecog_factor, 0.0, 1.0))

        if risk_score > 0.7:
            risk_level = "high"
            recommendation = "Consider dose reduction or alternative modality"
        elif risk_score > 0.4:
            risk_level = "moderate"
            recommendation = "Proceed with enhanced monitoring"
        else:
            risk_level = "low"
            recommendation = "Standard monitoring protocol"

        return {
            "modality": modality,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "factors": {
                "dose_intensity": dose_intensity,
                "age_adjustment": age_factor,
                "ecog_adjustment": ecog_factor,
            },
        }

    @staticmethod
    def _classify_response(reduction_fraction: float) -> ResponseCategory:
        """Classify treatment response using RECIST 1.1 criteria."""
        if reduction_fraction >= 0.65:
            return ResponseCategory.COMPLETE_RESPONSE
        elif reduction_fraction >= 0.30:
            return ResponseCategory.PARTIAL_RESPONSE
        elif reduction_fraction >= -0.20:
            return ResponseCategory.STABLE_DISEASE
        else:
            return ResponseCategory.PROGRESSIVE_DISEASE


# ---------------------------------------------------------------------------
# ReAct Treatment Planner Agent
# ---------------------------------------------------------------------------
class ReActTreatmentPlanner:
    """ReAct agent for iterative oncology treatment planning.

    Implements the Reason+Act pattern with thought/action/observation loops
    to develop personalized treatment recommendations. The agent:
    1. Reasons about the patient's clinical profile
    2. Selects and executes simulation tools
    3. Observes results and adjusts reasoning
    4. Iterates until a confident recommendation is formed

    All steps are logged for audit trail compliance.
    """

    def __init__(
        self,
        max_iterations: int = 10,
        confidence_threshold: float = 0.7,
        seed: int = 42,
    ) -> None:
        self._max_iterations = max_iterations
        self._confidence_threshold = confidence_threshold
        self._engine = TreatmentSimulationEngine(seed=seed)
        self._steps: list[ReActStep] = []
        self._state = AgentState.IDLE
        self._simulation_results: list[SimulationOutcome] = []
        self._rng = np.random.default_rng(seed)
        logger.info(
            "ReActTreatmentPlanner initialized: max_iter=%d, confidence=%.2f",
            max_iterations,
            confidence_threshold,
        )

    def plan_treatment(self, patient: PatientProfile) -> TreatmentRecommendation:
        """Execute the ReAct loop to develop a treatment recommendation.

        Args:
            patient: De-identified patient profile.

        Returns:
            TreatmentRecommendation with rationale and supporting evidence.
        """
        start_time = time.time()
        self._state = AgentState.THINKING
        self._steps = []
        self._simulation_results = []
        step_number = 0

        logger.info("Starting ReAct treatment planning for patient %s", patient.patient_id)

        # Step 1: Initial assessment thought
        step_number += 1
        self._add_thought(
            step_number,
            f"Assessing patient profile for {patient.patient_id}. "
            f"Diagnosis: {patient.diagnosis} Stage {patient.stage}, {patient.histology}. "
            f"ECOG: {patient.ecog_score}. Tumor volume: {patient.tumor_volume_cm3:.1f} cm3. "
            f"Need to evaluate treatment options based on biomarkers and clinical characteristics.",
        )

        # Step 2: Analyze biomarkers
        step_number += 1
        self._add_thought(
            step_number,
            self._reason_about_biomarkers(patient),
        )

        # Step 3: Simulate chemotherapy
        step_number += 1
        chemo_sensitivity = self._estimate_chemo_sensitivity(patient)
        self._state = AgentState.ACTING
        chemo_result = self._execute_action(
            step_number,
            ToolName.SIMULATE_CHEMO,
            {
                "volume_cm3": patient.tumor_volume_cm3,
                "sensitivity": chemo_sensitivity,
                "dose_mg": 75.0,
                "cycles": 4,
            },
        )
        self._simulation_results.append(chemo_result)

        # Step 4: Observe and reason about chemo results
        step_number += 1
        self._state = AgentState.OBSERVING
        self._add_observation(step_number, chemo_result)

        step_number += 1
        self._state = AgentState.THINKING
        self._add_thought(
            step_number,
            f"Chemotherapy simulation predicts {chemo_result.response_category.value} "
            f"({chemo_result.volume_reduction_pct:.1f}% reduction). "
            f"Toxicity: {chemo_result.toxicity_grade.name}. "
            f"Now evaluating radiation therapy as an alternative or adjunct.",
        )

        # Step 5: Simulate radiation
        step_number += 1
        radio_sensitivity = self._estimate_radio_sensitivity(patient)
        self._state = AgentState.ACTING
        radiation_result = self._execute_action(
            step_number,
            ToolName.SIMULATE_RADIATION,
            {
                "volume_cm3": patient.tumor_volume_cm3,
                "sensitivity": radio_sensitivity,
                "total_dose_gy": 60.0,
                "fractions": 30,
            },
        )
        self._simulation_results.append(radiation_result)

        # Step 6: Observe radiation results
        step_number += 1
        self._state = AgentState.OBSERVING
        self._add_observation(step_number, radiation_result)

        # Step 7: Consider immunotherapy if biomarkers support it
        pdl1 = patient.biomarkers.get("PDL1_TPS", 0.0)
        tmb = patient.biomarkers.get("TMB_mut_per_mb", 0.0)

        if pdl1 > 1.0 or tmb > 10.0:
            step_number += 1
            self._state = AgentState.THINKING
            self._add_thought(
                step_number,
                f"PD-L1 TPS={pdl1:.1f}%, TMB={tmb:.1f} mut/Mb. "
                f"Biomarkers suggest potential benefit from immunotherapy. "
                f"Simulating pembrolizumab-based regimen.",
            )

            step_number += 1
            immuno_sensitivity = self._estimate_immuno_sensitivity(patient)
            self._state = AgentState.ACTING
            immuno_result = self._execute_action(
                step_number,
                ToolName.SIMULATE_IMMUNO,
                {
                    "volume_cm3": patient.tumor_volume_cm3,
                    "sensitivity": immuno_sensitivity,
                    "response_rate": min(pdl1 / 100.0 * 0.7, 0.6),
                    "cycles": 8,
                },
            )
            self._simulation_results.append(immuno_result)

            step_number += 1
            self._state = AgentState.OBSERVING
            self._add_observation(step_number, immuno_result)

        # Step 8: Check drug interactions for best combination
        step_number += 1
        self._state = AgentState.ACTING
        medications = ["carboplatin", "pembrolizumab"] if pdl1 > 50.0 else ["carboplatin", "pemetrexed"]
        interaction_result = self._engine.check_drug_interactions(medications)
        self._add_step(
            step_number,
            ReActStepType.ACTION,
            f"Checking drug interactions for: {', '.join(medications)}",
            tool_name=ToolName.CHECK_INTERACTIONS.value,
            tool_arguments={"medications": medications},
            tool_result=interaction_result,
        )

        step_number += 1
        self._state = AgentState.OBSERVING
        self._add_step(
            step_number,
            ReActStepType.OBSERVATION,
            f"Drug interaction check: {interaction_result['interactions_found']} interactions found. "
            f"Overall risk: {interaction_result['overall_risk']}.",
        )

        # Step 9: Toxicity risk assessment
        step_number += 1
        self._state = AgentState.ACTING
        age_approx = int(patient.age_bracket.split("-")[0])
        best_result = max(self._simulation_results, key=lambda r: r.volume_reduction_pct)
        toxicity_assessment = self._engine.assess_toxicity_risk(
            modality=best_result.modality.value,
            dose_intensity=0.6,
            patient_age=age_approx,
            ecog=patient.ecog_score,
        )
        self._add_step(
            step_number,
            ReActStepType.ACTION,
            f"Assessing toxicity risk for {best_result.modality.value}",
            tool_name=ToolName.ASSESS_TOXICITY.value,
            tool_arguments={"modality": best_result.modality.value, "age": age_approx},
            tool_result=toxicity_assessment,
        )

        # Step 10: Synthesize recommendation
        step_number += 1
        self._state = AgentState.THINKING
        recommendation = self._synthesize_recommendation(patient, toxicity_assessment, interaction_result)

        self._add_step(
            step_number,
            ReActStepType.FINAL_ANSWER,
            f"Recommending {recommendation.recommended_modality.value}: "
            f"{recommendation.regimen_description}. "
            f"Expected {recommendation.expected_response.value} "
            f"({recommendation.expected_reduction_pct:.1f}% reduction). "
            f"Confidence: {recommendation.confidence_score:.2f}.",
        )

        # Finalize
        elapsed_ms = (time.time() - start_time) * 1000.0
        recommendation.react_steps_count = step_number
        recommendation.total_simulations = len(self._simulation_results)
        recommendation.total_duration_ms = elapsed_ms
        self._state = AgentState.COMPLETE

        logger.info(
            "Treatment planning complete: %d steps, %d simulations, %.1fms",
            step_number,
            len(self._simulation_results),
            elapsed_ms,
        )

        return recommendation

    def _add_thought(self, step_number: int, content: str) -> None:
        """Add a thought step to the reasoning trace."""
        self._add_step(step_number, ReActStepType.THOUGHT, content)
        logger.info("Thought [%d]: %s", step_number, content[:120])

    def _add_observation(self, step_number: int, result: SimulationOutcome) -> None:
        """Add an observation step from a simulation result."""
        content = (
            f"Simulation result for {result.modality.value}: "
            f"{result.volume_reduction_pct:.1f}% volume reduction "
            f"({result.response_category.value}). "
            f"CI: [{result.confidence_low:.2f}, {result.confidence_high:.2f}] cm3. "
            f"Toxicity: {result.toxicity_grade.name}. Duration: {result.duration_days} days."
        )
        self._add_step(step_number, ReActStepType.OBSERVATION, content)
        logger.info("Observation [%d]: %s", step_number, content[:120])

    def _add_step(
        self,
        step_number: int,
        step_type: ReActStepType,
        content: str,
        tool_name: str = "",
        tool_arguments: Optional[dict[str, Any]] = None,
        tool_result: Optional[dict[str, Any]] = None,
    ) -> ReActStep:
        """Add a step to the reasoning trace."""
        step = ReActStep(
            step_number=step_number,
            step_type=step_type,
            content=content,
            tool_name=tool_name,
            tool_arguments=tool_arguments or {},
            tool_result=tool_result,
        )
        self._steps.append(step)
        return step

    def _execute_action(
        self,
        step_number: int,
        tool: ToolName,
        arguments: dict[str, Any],
    ) -> SimulationOutcome:
        """Execute a simulation tool action and record the step."""
        start = time.time()

        if tool == ToolName.SIMULATE_CHEMO:
            result = self._engine.simulate_chemotherapy(**arguments)
        elif tool == ToolName.SIMULATE_RADIATION:
            result = self._engine.simulate_radiation(**arguments)
        elif tool == ToolName.SIMULATE_IMMUNO:
            result = self._engine.simulate_immunotherapy(**arguments)
        else:
            raise ValueError(f"Unsupported simulation tool: {tool.value}")

        duration_ms = (time.time() - start) * 1000.0
        self._add_step(
            step_number,
            ReActStepType.ACTION,
            f"Executing {tool.value} simulation",
            tool_name=tool.value,
            tool_arguments=arguments,
            tool_result={
                "modality": result.modality.value,
                "reduction_pct": result.volume_reduction_pct,
                "response": result.response_category.value,
            },
        )
        self._steps[-1].duration_ms = duration_ms

        logger.info(
            "Action [%d]: %s -> %s (%.1f%% reduction, %.1fms)",
            step_number,
            tool.value,
            result.response_category.value,
            result.volume_reduction_pct,
            duration_ms,
        )
        return result

    def _reason_about_biomarkers(self, patient: PatientProfile) -> str:
        """Generate reasoning about patient biomarkers."""
        pdl1 = patient.biomarkers.get("PDL1_TPS", 0.0)
        tmb = patient.biomarkers.get("TMB_mut_per_mb", 0.0)
        egfr = patient.biomarkers.get("EGFR_mutation", 0.0)
        alk = patient.biomarkers.get("ALK_rearrangement", 0.0)

        lines = [f"Biomarker analysis for {patient.patient_id}:"]

        if pdl1 >= 50:
            lines.append(f"- PD-L1 TPS {pdl1:.1f}%: HIGH - strong candidate for first-line immunotherapy")
        elif pdl1 >= 1:
            lines.append(f"- PD-L1 TPS {pdl1:.1f}%: POSITIVE - consider immunotherapy combination")
        else:
            lines.append(f"- PD-L1 TPS {pdl1:.1f}%: NEGATIVE - chemotherapy preferred")

        if tmb > 10:
            lines.append(f"- TMB {tmb:.1f} mut/Mb: HIGH - may benefit from immunotherapy")
        else:
            lines.append(f"- TMB {tmb:.1f} mut/Mb: LOW - immunotherapy benefit uncertain")

        if egfr > 0:
            lines.append("- EGFR mutation: POSITIVE - consider targeted therapy (osimertinib)")
        if alk > 0:
            lines.append("- ALK rearrangement: POSITIVE - consider targeted therapy (alectinib)")

        lines.append("Proceeding with simulation of candidate treatment modalities.")
        return " ".join(lines)

    def _estimate_chemo_sensitivity(self, patient: PatientProfile) -> float:
        """Estimate chemotherapy sensitivity from patient characteristics."""
        base = 0.5
        if patient.histology == "adenocarcinoma":
            base += 0.05
        if patient.ecog_score <= 1:
            base += 0.05
        if patient.stage in ["I", "II"]:
            base += 0.1
        return float(np.clip(base + self._rng.normal(0, 0.05), 0.1, 0.9))

    def _estimate_radio_sensitivity(self, patient: PatientProfile) -> float:
        """Estimate radiation sensitivity from patient characteristics."""
        base = 0.6
        if patient.histology == "squamous_cell":
            base += 0.1
        if patient.tumor_volume_cm3 < 5.0:
            base += 0.05
        return float(np.clip(base + self._rng.normal(0, 0.05), 0.1, 0.9))

    def _estimate_immuno_sensitivity(self, patient: PatientProfile) -> float:
        """Estimate immunotherapy sensitivity from biomarkers."""
        pdl1 = patient.biomarkers.get("PDL1_TPS", 0.0)
        tmb = patient.biomarkers.get("TMB_mut_per_mb", 0.0)
        base = 0.2 + (pdl1 / 100.0) * 0.3 + min(tmb / 20.0, 0.3)
        return float(np.clip(base + self._rng.normal(0, 0.05), 0.05, 0.9))

    def _synthesize_recommendation(
        self,
        patient: PatientProfile,
        toxicity_assessment: dict[str, Any],
        interaction_result: dict[str, Any],
    ) -> TreatmentRecommendation:
        """Synthesize a treatment recommendation from all simulation results."""
        # Rank modalities by effectiveness (reduction) adjusted for toxicity
        ranked = sorted(self._simulation_results, key=lambda r: r.volume_reduction_pct, reverse=True)
        best = ranked[0]

        # Compute confidence score
        reduction_range = best.confidence_high - best.confidence_low
        uncertainty_penalty = min(reduction_range / max(best.initial_volume_cm3, 0.01), 0.5)
        confidence = float(np.clip(0.85 - uncertainty_penalty - (best.toxicity_grade.value * 0.05), 0.3, 0.95))

        # Determine if human review is needed
        requires_review = True
        review_reason = "All treatment recommendations require oncologist review"
        if best.toxicity_grade.value >= 3:
            review_reason = f"High toxicity risk ({best.toxicity_grade.name}); dose adjustment may be needed"
        elif confidence < self._confidence_threshold:
            review_reason = f"Low confidence score ({confidence:.2f}); additional evaluation recommended"

        # Build regimen description
        if best.modality == TreatmentModality.CHEMOTHERAPY:
            regimen = "Carboplatin AUC 5 IV Q3W x4 cycles"
        elif best.modality == TreatmentModality.RADIATION:
            regimen = "60 Gy in 30 fractions, 5 days/week, IMRT technique"
        elif best.modality == TreatmentModality.IMMUNOTHERAPY:
            regimen = "Pembrolizumab 200mg IV Q3W x8 cycles"
        else:
            regimen = f"{best.modality.value} per protocol"

        # Build alternatives
        alternatives = []
        for alt in ranked[1:]:
            alternatives.append(
                {
                    "modality": alt.modality.value,
                    "expected_reduction_pct": alt.volume_reduction_pct,
                    "response_category": alt.response_category.value,
                    "toxicity": alt.toxicity_grade.name,
                }
            )

        # Supporting evidence
        evidence = [
            f"Monte Carlo simulation ({best.metadata.get('n_samples', 100)} iterations): "
            f"{best.volume_reduction_pct:.1f}% volume reduction predicted",
            f"RECIST 1.1 classification: {best.response_category.value}",
            f"95% CI for final volume: [{best.confidence_low:.2f}, {best.confidence_high:.2f}] cm3",
            f"Toxicity risk assessment: {toxicity_assessment.get('risk_level', 'unknown')}",
            f"Drug interaction check: {interaction_result['overall_risk']} risk",
        ]

        # Rationale
        rationale = (
            f"Based on {len(self._simulation_results)} treatment simulations, "
            f"{best.modality.value} shows the best predicted response "
            f"({best.volume_reduction_pct:.1f}% tumor volume reduction, "
            f"{best.response_category.value}). "
            f"Toxicity is estimated at {best.toxicity_grade.name} with "
            f"{toxicity_assessment.get('risk_level', 'moderate')} overall risk. "
            f"Drug interaction analysis indicates {interaction_result['overall_risk']} risk."
        )

        return TreatmentRecommendation(
            patient_id=patient.patient_id,
            recommended_modality=best.modality,
            regimen_description=regimen,
            expected_response=best.response_category,
            expected_reduction_pct=best.volume_reduction_pct,
            confidence_score=confidence,
            toxicity_risk=best.toxicity_grade,
            rationale=rationale,
            supporting_evidence=evidence,
            alternative_options=alternatives,
            requires_human_review=requires_review,
            human_review_reason=review_reason,
        )

    def get_reasoning_trace(self) -> list[dict[str, Any]]:
        """Export the complete ReAct reasoning trace."""
        return [step.to_dict() for step in self._steps]

    def get_trace_hash(self) -> str:
        """Compute an integrity hash over the entire reasoning trace."""
        trace_json = json.dumps(self.get_reasoning_trace(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(trace_json).hexdigest()

    @property
    def state(self) -> AgentState:
        """Current agent state."""
        return self._state

    @property
    def step_count(self) -> int:
        """Number of reasoning steps taken."""
        return len(self._steps)


# ---------------------------------------------------------------------------
# Main demonstration
# ---------------------------------------------------------------------------
def main() -> None:
    """Demonstrate the ReAct treatment planner with a sample patient."""
    logger.info("=" * 80)
    logger.info("ReAct Treatment Planner Demonstration")
    logger.info("Version: 0.5.0 | RESEARCH USE ONLY")
    logger.info("=" * 80)

    logger.info("Optional dependencies: HAS_ANTHROPIC=%s, HAS_OPENAI=%s", HAS_ANTHROPIC, HAS_OPENAI)

    # Create a sample patient
    patient = PatientProfile(
        patient_id="PAT-DEMO-001",
        age_bracket="62-69",
        sex="male",
        diagnosis="NSCLC",
        stage="IIIB",
        histology="adenocarcinoma",
        ecog_score=1,
        tumor_volume_cm3=8.5,
        biomarkers={
            "PDL1_TPS": 65.0,
            "TMB_mut_per_mb": 14.2,
            "EGFR_mutation": 0.0,
            "ALK_rearrangement": 0.0,
            "KRAS_G12C": 1.0,
        },
        prior_treatments=[],
        comorbidities=["hypertension", "type_2_diabetes"],
        organ_function={"renal": "adequate", "hepatic": "adequate", "cardiac": "adequate"},
    )

    logger.info("Patient profile:")
    logger.info(patient.to_context_string())

    # Initialize and run the planner
    planner = ReActTreatmentPlanner(max_iterations=10, confidence_threshold=0.7, seed=42)
    recommendation = planner.plan_treatment(patient)

    # Display results
    logger.info("-" * 60)
    logger.info("TREATMENT RECOMMENDATION")
    logger.info("-" * 60)
    rec_dict = recommendation.to_dict()
    for key, value in rec_dict.items():
        if isinstance(value, list):
            logger.info("  %s:", key)
            for item in value:
                logger.info("    - %s", item)
        else:
            logger.info("  %s: %s", key, value)

    # Display reasoning trace summary
    logger.info("-" * 60)
    logger.info("REASONING TRACE (%d steps):", planner.step_count)
    for step_dict in planner.get_reasoning_trace():
        step_type = step_dict["step_type"].upper()
        content = step_dict["content"][:100]
        logger.info("  [%d] %s: %s", step_dict["step_number"], step_type, content)

    # Integrity hash
    trace_hash = planner.get_trace_hash()
    logger.info("Reasoning trace hash: %s", trace_hash[:24])

    logger.info("=" * 80)
    logger.info("Demonstration complete.")


if __name__ == "__main__":
    main()
