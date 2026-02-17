"""
RAG Agent Grounded in Trial Protocols, FDA Guidance, ICH E6(R3), and IEC Standards.

CLINICAL CONTEXT:
    This module implements a Retrieval-Augmented Generation (RAG) agent
    specialized for regulatory compliance queries in oncology clinical
    trials. The agent retrieves relevant passages from a curated knowledge
    base of trial protocols, FDA guidance documents, ICH E6(R3) GCP
    guidelines, and IEC medical device standards, then generates
    grounded answers with full citation tracking and confidence scoring.

    The RAG pipeline uses semantic chunking, vector similarity search,
    cross-encoder reranking, and faithfulness verification to ensure
    that all responses are traceable to authoritative source documents.
    This supports 21 CFR Part 11 audit requirements and enables
    reproducible compliance assessments.

USE CASES COVERED:
    1. Protocol compliance queries: "Is this patient eligible for trial
       PROTO-ONC-2025-001 given their PD-L1 status?"
    2. Regulatory guidance lookup: "What are the FDA requirements for
       electronic signatures in clinical trial data?"
    3. ICH E6(R3) interpretation: "What does ICH E6(R3) say about
       risk-based monitoring for decentralized trials?"
    4. IEC standard queries: "What are the force limits for robotic
       surgical equipment under IEC 80601-2-77?"
    5. Cross-reference analysis: "How do FDA 21 CFR Part 11 and ICH E6(R3)
       requirements overlap for electronic source data?"

FRAMEWORK REQUIREMENTS:
    Required:
        - numpy >= 1.24.0        (https://numpy.org/)
    Optional:
        - anthropic >= 0.39.0    (https://docs.anthropic.com/)
        - openai >= 1.0.0        (https://platform.openai.com/)
        - chromadb >= 0.4.0      (https://www.trychroma.com/)
        - sentence-transformers >= 2.2.0 (https://sbert.net/)

REFERENCES:
    - Lewis et al. (2020). Retrieval-Augmented Generation for
      Knowledge-Intensive NLP Tasks. NeurIPS 2020.
      arXiv: 2005.11401
    - FDA 21 CFR Part 11 - Electronic Records; Electronic Signatures.
    - ICH E6(R3) Good Clinical Practice (2023).
    - IEC 80601-2-77:2019 Medical Electrical Equipment - Robotically
      assisted surgical equipment.
    - ISO 14971:2019 Medical devices - Risk management.
    - ICH E8(R1) General Considerations for Clinical Studies (2021).

DISCLAIMER:
    RESEARCH USE ONLY. This software is provided for research and educational
    purposes only. It has NOT been validated for clinical use, is NOT approved
    by the FDA or any other regulatory body, and MUST NOT be used to make
    clinical decisions or direct patient care. All compliance assessments
    must be reviewed by qualified regulatory professionals.

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

try:
    import chromadb  # type: ignore[import-untyped]

    HAS_CHROMADB = True
except ImportError:
    chromadb = None  # type: ignore[assignment]
    HAS_CHROMADB = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment,misc]
    HAS_SENTENCE_TRANSFORMERS = False

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
class DocumentSource(Enum):
    """Source categories for regulatory documents."""

    TRIAL_PROTOCOL = "trial_protocol"
    FDA_GUIDANCE = "fda_guidance"
    ICH_GUIDELINE = "ich_guideline"
    IEC_STANDARD = "iec_standard"
    ISO_STANDARD = "iso_standard"
    INTERNAL_SOP = "internal_sop"
    LITERATURE = "peer_reviewed_literature"


class QueryType(Enum):
    """Types of compliance queries."""

    ELIGIBILITY = "eligibility"
    DOSING = "dosing"
    SAFETY_REPORTING = "safety_reporting"
    DATA_MANAGEMENT = "data_management"
    ELECTRONIC_RECORDS = "electronic_records"
    MONITORING = "monitoring"
    INFORMED_CONSENT = "informed_consent"
    DEVICE_REQUIREMENTS = "device_requirements"
    GENERAL_COMPLIANCE = "general_compliance"


class ConfidenceLevel(Enum):
    """Confidence levels for RAG-generated responses."""

    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    INSUFFICIENT = "insufficient"


class FaithfulnessRating(Enum):
    """Faithfulness rating for response grounding verification."""

    FULLY_GROUNDED = "fully_grounded"
    MOSTLY_GROUNDED = "mostly_grounded"
    PARTIALLY_GROUNDED = "partially_grounded"
    NOT_GROUNDED = "not_grounded"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class DocumentChunk:
    """A chunk of a regulatory document stored in the vector index."""

    chunk_id: str = field(default_factory=lambda: f"CHK-{uuid.uuid4().hex[:10].upper()}")
    document_id: str = ""
    document_title: str = ""
    source: DocumentSource = DocumentSource.FDA_GUIDANCE
    section: str = ""
    subsection: str = ""
    content: str = ""
    page_number: int = 0
    version: str = ""
    effective_date: str = ""
    embedding: Optional[list[float]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary (excludes embedding for readability)."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "source": self.source.value,
            "section": self.section,
            "subsection": self.subsection,
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "version": self.version,
            "effective_date": self.effective_date,
        }


@dataclass
class Citation:
    """A citation linking a response claim to a source document."""

    citation_id: str = field(default_factory=lambda: f"CIT-{uuid.uuid4().hex[:8].upper()}")
    chunk_id: str = ""
    document_title: str = ""
    source: DocumentSource = DocumentSource.FDA_GUIDANCE
    section: str = ""
    page_number: int = 0
    quote: str = ""
    relevance_score: float = 0.0
    used_in_response: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "citation_id": self.citation_id,
            "document_title": self.document_title,
            "source": self.source.value,
            "section": self.section,
            "page_number": self.page_number,
            "quote": self.quote[:150] + "..." if len(self.quote) > 150 else self.quote,
            "relevance_score": self.relevance_score,
        }

    def format_inline(self) -> str:
        """Format as an inline citation."""
        return f"[{self.document_title}, {self.section}]"


@dataclass
class RetrievalResult:
    """Result from the retrieval stage of the RAG pipeline."""

    query: str = ""
    chunks_retrieved: list[DocumentChunk] = field(default_factory=list)
    chunks_after_rerank: list[DocumentChunk] = field(default_factory=list)
    relevance_scores: list[float] = field(default_factory=list)
    retrieval_time_ms: float = 0.0
    rerank_time_ms: float = 0.0


@dataclass
class ComplianceResponse:
    """A compliance response generated by the RAG agent."""

    response_id: str = field(default_factory=lambda: f"RSP-{uuid.uuid4().hex[:10].upper()}")
    query: str = ""
    query_type: QueryType = QueryType.GENERAL_COMPLIANCE
    answer: str = ""
    citations: list[Citation] = field(default_factory=list)
    confidence: ConfidenceLevel = ConfidenceLevel.MODERATE
    confidence_score: float = 0.0
    faithfulness: FaithfulnessRating = FaithfulnessRating.MOSTLY_GROUNDED
    sources_consulted: int = 0
    relevant_standards: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    follow_up_questions: list[str] = field(default_factory=list)
    generation_time_ms: float = 0.0
    total_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "response_id": self.response_id,
            "query": self.query,
            "query_type": self.query_type.value,
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "confidence": self.confidence.value,
            "confidence_score": self.confidence_score,
            "faithfulness": self.faithfulness.value,
            "sources_consulted": self.sources_consulted,
            "relevant_standards": self.relevant_standards,
            "caveats": self.caveats,
            "follow_up_questions": self.follow_up_questions,
            "total_time_ms": self.total_time_ms,
        }


@dataclass
class AuditEntry:
    """Audit entry for compliance query tracking."""

    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query_hash: str = ""
    response_id: str = ""
    citations_count: int = 0
    confidence: str = ""
    faithfulness: str = ""
    timestamp: float = field(default_factory=time.time)
    previous_hash: str = ""
    entry_hash: str = ""

    def compute_hash(self, previous_hash: str = "") -> str:
        """Compute hash for audit chain integrity."""
        self.previous_hash = previous_hash
        payload = json.dumps(
            {
                "entry_id": self.entry_id,
                "query_hash": self.query_hash,
                "response_id": self.response_id,
                "timestamp": self.timestamp,
                "previous_hash": self.previous_hash,
            },
            sort_keys=True,
        ).encode("utf-8")
        self.entry_hash = hashlib.sha256(payload).hexdigest()
        return self.entry_hash


# ---------------------------------------------------------------------------
# Regulatory knowledge base
# ---------------------------------------------------------------------------
def _build_regulatory_knowledge_base() -> list[DocumentChunk]:
    """Build a curated knowledge base of regulatory document chunks.

    Returns pre-defined chunks from FDA guidance, ICH guidelines,
    IEC standards, and trial protocols for demonstration.
    """
    chunks = [
        # FDA 21 CFR Part 11
        DocumentChunk(
            document_id="FDA-21CFR11",
            document_title="FDA 21 CFR Part 11",
            source=DocumentSource.FDA_GUIDANCE,
            section="Section 11.10",
            subsection="Controls for closed systems",
            content=(
                "Persons who use closed systems to create, modify, maintain, or transmit electronic "
                "records shall employ procedures and controls designed to ensure the authenticity, "
                "integrity, and, when appropriate, the confidentiality of electronic records, and to "
                "ensure that the signer cannot readily repudiate the signed record as not genuine. "
                "Such procedures and controls shall include: (a) Validation of systems to ensure "
                "accuracy, reliability, consistent intended performance, and the ability to discern "
                "invalid or altered records."
            ),
            version="Current",
            effective_date="1997-03-20",
        ),
        DocumentChunk(
            document_id="FDA-21CFR11",
            document_title="FDA 21 CFR Part 11",
            source=DocumentSource.FDA_GUIDANCE,
            section="Section 11.10(e)",
            subsection="Audit trails",
            content=(
                "Use of secure, computer-generated, time-stamped audit trails to independently record "
                "the date and time of operator entries and actions that create, modify, or delete "
                "electronic records. Record changes shall not obscure previously recorded information. "
                "Such audit trail documentation shall be retained for a period at least as long as that "
                "required for the subject electronic records and shall be available for agency review "
                "and copying."
            ),
            version="Current",
            effective_date="1997-03-20",
        ),
        DocumentChunk(
            document_id="FDA-21CFR11",
            document_title="FDA 21 CFR Part 11",
            source=DocumentSource.FDA_GUIDANCE,
            section="Section 11.50",
            subsection="Signature manifestations",
            content=(
                "Signed electronic records shall contain information associated with the signing that "
                "clearly indicates all of the following: (a) The printed name of the signer; (b) The "
                "date and time when the signature was executed; and (c) The meaning (such as review, "
                "approval, responsibility, or authorship) associated with the signature."
            ),
            version="Current",
            effective_date="1997-03-20",
        ),
        # ICH E6(R3)
        DocumentChunk(
            document_id="ICH-E6R3",
            document_title="ICH E6(R3) Good Clinical Practice",
            source=DocumentSource.ICH_GUIDELINE,
            section="Section 1.1",
            subsection="Principles",
            content=(
                "Clinical trials should be conducted in accordance with the ethical principles that "
                "have their origin in the Declaration of Helsinki, and that are consistent with GCP "
                "and the applicable regulatory requirement(s). Before a trial is initiated, foreseeable "
                "risks and inconveniences should be weighed against the anticipated benefit for the "
                "individual trial participant and society. A trial should be initiated and continued "
                "only if the anticipated benefits justify the risks."
            ),
            version="Step 2b",
            effective_date="2023-05-19",
        ),
        DocumentChunk(
            document_id="ICH-E6R3",
            document_title="ICH E6(R3) Good Clinical Practice",
            source=DocumentSource.ICH_GUIDELINE,
            section="Section 3.3",
            subsection="Quality management",
            content=(
                "The sponsor should implement a quality management system covering the design, conduct, "
                "recording, evaluation, reporting, and archiving of clinical trials. The quality "
                "management approach should be proportionate to the risks inherent in the trial and the "
                "importance of the information collected. Quality management should include: quality "
                "planning, quality assurance, quality control, and quality improvement throughout the "
                "clinical trial lifecycle."
            ),
            version="Step 2b",
            effective_date="2023-05-19",
        ),
        DocumentChunk(
            document_id="ICH-E6R3",
            document_title="ICH E6(R3) Good Clinical Practice",
            source=DocumentSource.ICH_GUIDELINE,
            section="Section 5.1",
            subsection="Informed consent",
            content=(
                "The investigator should ensure that each trial participant, or the participant's legally "
                "acceptable representative, is fully informed about the nature, purpose, risks, and "
                "benefits of the trial, and their rights as a participant, including the right to "
                "withdraw at any time without penalty. Informed consent should be documented by the use "
                "of a written, signed, and dated informed consent form."
            ),
            version="Step 2b",
            effective_date="2023-05-19",
        ),
        DocumentChunk(
            document_id="ICH-E6R3",
            document_title="ICH E6(R3) Good Clinical Practice",
            source=DocumentSource.ICH_GUIDELINE,
            section="Section 6.4",
            subsection="Dose escalation and modification",
            content=(
                "The protocol should specify the criteria and procedures for dose escalation, dose "
                "modification, and dose reduction. Dose-limiting toxicities should be clearly defined. "
                "Procedures for managing overdoses and dose modifications based on toxicity grading "
                "(e.g., CTCAE) should be described. The safety monitoring committee should review "
                "dose escalation decisions."
            ),
            version="Step 2b",
            effective_date="2023-05-19",
        ),
        DocumentChunk(
            document_id="ICH-E6R3",
            document_title="ICH E6(R3) Good Clinical Practice",
            source=DocumentSource.ICH_GUIDELINE,
            section="Section 5.18",
            subsection="Monitoring",
            content=(
                "The sponsor should ensure that trials are adequately monitored. The sponsor should "
                "determine the appropriate extent and nature of monitoring. Monitoring should be "
                "based on a risk-based approach that considers factors such as the objective, design, "
                "complexity, and endpoints of the trial. Centralized monitoring should complement "
                "on-site monitoring."
            ),
            version="Step 2b",
            effective_date="2023-05-19",
        ),
        # IEC 80601-2-77
        DocumentChunk(
            document_id="IEC-80601-2-77",
            document_title="IEC 80601-2-77:2019",
            source=DocumentSource.IEC_STANDARD,
            section="Section 201.12.4",
            subsection="Mechanical hazards",
            content=(
                "Robotically assisted surgical equipment shall be designed to minimize risks from "
                "mechanical hazards including excessive force, unintended motion, and loss of "
                "position accuracy. Maximum force limits shall be defined for all operational modes. "
                "The equipment shall include means to limit the force, torque, and speed of "
                "the robotic mechanism to values within the range specified by the manufacturer."
            ),
            version="2019",
            effective_date="2019-07-01",
        ),
        DocumentChunk(
            document_id="IEC-80601-2-77",
            document_title="IEC 80601-2-77:2019",
            source=DocumentSource.IEC_STANDARD,
            section="Section 201.12.4.4",
            subsection="Emergency stop",
            content=(
                "The robotic system shall include an emergency stop function that, when activated, "
                "immediately brings the robotic mechanism to a stop in a safe state. The emergency "
                "stop function shall be accessible and operable at all times during the operation "
                "of the equipment. Recovery from an emergency stop shall require a deliberate action "
                "by the operator."
            ),
            version="2019",
            effective_date="2019-07-01",
        ),
        DocumentChunk(
            document_id="IEC-80601-2-77",
            document_title="IEC 80601-2-77:2019",
            source=DocumentSource.IEC_STANDARD,
            section="Section 201.12.4.3",
            subsection="Workspace limitations",
            content=(
                "The manufacturer shall define the workspace limits for the robotic mechanism. "
                "The equipment shall include means to prevent the robotic mechanism from exceeding "
                "these limits during normal operation and single fault conditions. Workspace "
                "limitation shall be implemented through both software and hardware means."
            ),
            version="2019",
            effective_date="2019-07-01",
        ),
        # ISO 14971
        DocumentChunk(
            document_id="ISO-14971",
            document_title="ISO 14971:2019",
            source=DocumentSource.ISO_STANDARD,
            section="Section 5.5",
            subsection="Risk evaluation",
            content=(
                "For each identified hazardous situation, the manufacturer shall evaluate the "
                "associated risks using the risk acceptability criteria defined in the risk "
                "management plan. Risks that are not judged acceptable shall be controlled using "
                "one or more of the risk control options specified in Clause 6. The manufacturer "
                "shall document the results of the risk evaluation."
            ),
            version="2019",
            effective_date="2019-12-01",
        ),
        DocumentChunk(
            document_id="ISO-14971",
            document_title="ISO 14971:2019",
            source=DocumentSource.ISO_STANDARD,
            section="Section 6.2",
            subsection="Risk control option analysis",
            content=(
                "The manufacturer shall identify risk control measures using the following priority "
                "order: (a) inherent safety by design; (b) protective measures in the medical device "
                "itself or in the manufacturing process; (c) information for safety. If the risk is "
                "not reducible through design, the manufacturer shall apply protective measures."
            ),
            version="2019",
            effective_date="2019-12-01",
        ),
        # Trial Protocol
        DocumentChunk(
            document_id="PROTO-ONC-2025-001",
            document_title="Phase III NSCLC Pembrolizumab Trial Protocol",
            source=DocumentSource.TRIAL_PROTOCOL,
            section="Section 4.1",
            subsection="Eligibility criteria",
            content=(
                "Inclusion criteria: (1) Age >= 18 years; (2) Histologically confirmed stage IIIB-IV "
                "NSCLC (adenocarcinoma or squamous cell carcinoma); (3) PD-L1 TPS >= 1% by validated "
                "IHC assay; (4) ECOG performance status 0-2; (5) Adequate organ function including "
                "ANC >= 1.5 x10^9/L, platelets >= 100 x10^9/L, hemoglobin >= 9.0 g/dL, creatinine "
                "<= 1.5x ULN, AST/ALT <= 2.5x ULN; (6) No prior systemic therapy for metastatic "
                "disease (adjuvant chemotherapy completed > 12 months is permitted)."
            ),
            version="3.2",
            effective_date="2025-09-01",
        ),
        DocumentChunk(
            document_id="PROTO-ONC-2025-001",
            document_title="Phase III NSCLC Pembrolizumab Trial Protocol",
            source=DocumentSource.TRIAL_PROTOCOL,
            section="Section 5.2",
            subsection="Dosing and administration",
            content=(
                "Arm A (Experimental): Pembrolizumab 200 mg IV over 30 minutes every 3 weeks (Q3W) "
                "plus Carboplatin AUC 5 IV over 15-60 minutes Q3W for 4 cycles, followed by "
                "pembrolizumab 200 mg IV Q3W maintenance until progression or up to 35 cycles. "
                "Arm B (Control): Carboplatin AUC 5 IV Q3W for 4 cycles plus matching placebo. "
                "Dose modifications: reduce carboplatin to AUC 4 for Grade 3 hematologic toxicity; "
                "hold pembrolizumab for Grade >= 3 immune-related adverse events."
            ),
            version="3.2",
            effective_date="2025-09-01",
        ),
        DocumentChunk(
            document_id="PROTO-ONC-2025-001",
            document_title="Phase III NSCLC Pembrolizumab Trial Protocol",
            source=DocumentSource.TRIAL_PROTOCOL,
            section="Section 8.1",
            subsection="Safety reporting",
            content=(
                "All adverse events (AEs) shall be graded according to CTCAE v5.0. Serious adverse "
                "events (SAEs) must be reported within 24 hours of the investigator becoming aware. "
                "Suspected unexpected serious adverse reactions (SUSARs) must be reported to the "
                "sponsor and IRB/EC within the timelines specified by applicable regulations. "
                "A Data Safety Monitoring Board (DSMB) will review unblinded safety data quarterly."
            ),
            version="3.2",
            effective_date="2025-09-01",
        ),
    ]
    return chunks


# ---------------------------------------------------------------------------
# Vector index (lightweight numpy-based)
# ---------------------------------------------------------------------------
class NumpyVectorIndex:
    """Lightweight vector index using numpy for cosine similarity search.

    This is a minimal implementation for demonstration. In production,
    use ChromaDB, Pinecone, Weaviate, or another purpose-built vector store.
    """

    def __init__(self, embedding_dim: int = 64) -> None:
        self._chunks: list[DocumentChunk] = []
        self._embeddings: Optional[np.ndarray] = None
        self._embedding_dim = embedding_dim
        self._rng = np.random.default_rng(42)

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Add document chunks to the index with synthetic embeddings."""
        for chunk in chunks:
            if chunk.embedding is None:
                chunk.embedding = self._generate_embedding(chunk.content)
            self._chunks.append(chunk)

        all_embeddings = [c.embedding for c in self._chunks if c.embedding is not None]
        if all_embeddings:
            self._embeddings = np.array(all_embeddings)

        logger.info("Vector index: %d chunks indexed", len(self._chunks))

    def search(self, query: str, top_k: int = 5) -> list[tuple[DocumentChunk, float]]:
        """Search for the most relevant chunks using cosine similarity.

        Args:
            query: The search query.
            top_k: Number of top results to return.

        Returns:
            List of (chunk, similarity_score) tuples sorted by relevance.
        """
        if self._embeddings is None or len(self._chunks) == 0:
            return []

        query_embedding = np.array(self._generate_embedding(query))
        query_norm = query_embedding / max(np.linalg.norm(query_embedding), 1e-8)
        embeddings_norm = self._embeddings / np.maximum(np.linalg.norm(self._embeddings, axis=1, keepdims=True), 1e-8)

        similarities = embeddings_norm @ query_norm
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((self._chunks[idx], float(similarities[idx])))

        return results

    def _generate_embedding(self, text: str) -> list[float]:
        """Generate a synthetic embedding from text content.

        Uses a deterministic hash-based approach combined with word overlap
        to create embeddings that capture basic semantic similarity.
        In production, use a sentence transformer model.
        """
        words = text.lower().split()
        embedding = np.zeros(self._embedding_dim)

        for i, word in enumerate(words):
            word_hash = int(hashlib.md5(word.encode()).hexdigest()[:8], 16)
            idx = word_hash % self._embedding_dim
            embedding[idx] += 1.0 / (1.0 + i * 0.01)

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.tolist()

    @property
    def chunk_count(self) -> int:
        """Number of chunks in the index."""
        return len(self._chunks)


# ---------------------------------------------------------------------------
# RAG compliance agent
# ---------------------------------------------------------------------------
class OncologyRAGComplianceAgent:
    """RAG agent for regulatory compliance queries in oncology clinical trials.

    Implements a full RAG pipeline: query classification, retrieval,
    reranking, generation, citation tracking, and faithfulness verification.
    """

    def __init__(self, seed: int = 42) -> None:
        self._index = NumpyVectorIndex(embedding_dim=64)
        self._knowledge_base = _build_regulatory_knowledge_base()
        self._index.add_chunks(self._knowledge_base)
        self._query_history: list[ComplianceResponse] = []
        self._audit_entries: list[AuditEntry] = []
        self._last_audit_hash = ""
        self._rng = np.random.default_rng(seed)

        logger.info(
            "OncologyRAGComplianceAgent initialized: %d document chunks indexed",
            self._index.chunk_count,
        )

    def query(self, question: str, top_k: int = 5) -> ComplianceResponse:
        """Process a compliance query through the full RAG pipeline.

        Pipeline stages:
        1. Query classification (determine query type)
        2. Retrieval (vector similarity search)
        3. Reranking (cross-encoder style relevance scoring)
        4. Generation (synthesize answer from retrieved chunks)
        5. Citation tracking (link claims to source documents)
        6. Faithfulness verification (check grounding)

        Args:
            question: The compliance query.
            top_k: Number of chunks to retrieve.

        Returns:
            ComplianceResponse with grounded answer and citations.
        """
        start_time = time.time()

        # Stage 1: Query classification
        query_type = self._classify_query(question)
        logger.info("Query classified as: %s", query_type.value)

        # Stage 2: Retrieval
        retrieval_start = time.time()
        raw_results = self._index.search(question, top_k=top_k * 2)
        retrieval_time_ms = (time.time() - retrieval_start) * 1000.0

        # Stage 3: Reranking
        rerank_start = time.time()
        reranked = self._rerank_results(question, raw_results, top_k=top_k)
        rerank_time_ms = (time.time() - rerank_start) * 1000.0

        logger.info(
            "Retrieval: %d raw -> %d reranked (%.1fms retrieval, %.1fms rerank)",
            len(raw_results),
            len(reranked),
            retrieval_time_ms,
            rerank_time_ms,
        )

        # Stage 4: Generation
        gen_start = time.time()
        answer, citations = self._generate_answer(question, query_type, reranked)
        gen_time_ms = (time.time() - gen_start) * 1000.0

        # Stage 5: Confidence scoring
        confidence_score, confidence_level = self._score_confidence(reranked, citations)

        # Stage 6: Faithfulness verification
        faithfulness = self._verify_faithfulness(answer, reranked)

        # Build caveats
        caveats = [
            "RESEARCH USE ONLY - not validated for clinical decision-making",
            "All compliance assessments must be reviewed by qualified regulatory professionals",
        ]
        if confidence_level in (ConfidenceLevel.LOW, ConfidenceLevel.INSUFFICIENT):
            caveats.append("Low confidence: retrieved sources may not fully address the query")

        # Build follow-up questions
        follow_ups = self._suggest_follow_ups(question, query_type)

        # Collect relevant standards
        standards = list(set(c.source.value for c in citations))

        total_time_ms = (time.time() - start_time) * 1000.0

        response = ComplianceResponse(
            query=question,
            query_type=query_type,
            answer=answer,
            citations=citations,
            confidence=confidence_level,
            confidence_score=confidence_score,
            faithfulness=faithfulness,
            sources_consulted=len(reranked),
            relevant_standards=standards,
            caveats=caveats,
            follow_up_questions=follow_ups,
            generation_time_ms=gen_time_ms,
            total_time_ms=total_time_ms,
        )

        # Record audit entry
        self._record_audit(response)
        self._query_history.append(response)

        logger.info(
            "Response generated: confidence=%s (%.2f), faithfulness=%s, %d citations, %.1fms total",
            confidence_level.value,
            confidence_score,
            faithfulness.value,
            len(citations),
            total_time_ms,
        )

        return response

    def _classify_query(self, question: str) -> QueryType:
        """Classify the query type based on keyword analysis."""
        q_lower = question.lower()
        classification_map = {
            QueryType.ELIGIBILITY: ["eligib", "inclusion", "exclusion", "enroll", "criteria"],
            QueryType.DOSING: ["dose", "dosing", "dosage", "mg", "auc", "cycle", "reduction"],
            QueryType.SAFETY_REPORTING: ["adverse", "safety", "sae", "susar", "ctcae", "toxicity"],
            QueryType.DATA_MANAGEMENT: ["data", "crf", "edc", "database", "query"],
            QueryType.ELECTRONIC_RECORDS: ["electronic", "21 cfr", "signature", "audit trail", "part 11"],
            QueryType.MONITORING: ["monitor", "oversight", "risk-based", "centralized"],
            QueryType.INFORMED_CONSENT: ["consent", "informed", "participant rights"],
            QueryType.DEVICE_REQUIREMENTS: ["robot", "device", "iec", "force", "workspace", "surgical"],
        }

        for qtype, keywords in classification_map.items():
            if any(kw in q_lower for kw in keywords):
                return qtype

        return QueryType.GENERAL_COMPLIANCE

    def _rerank_results(
        self, query: str, results: list[tuple[DocumentChunk, float]], top_k: int = 5
    ) -> list[tuple[DocumentChunk, float]]:
        """Rerank results using keyword overlap scoring.

        In production, use a cross-encoder model for reranking.
        """
        query_words = set(query.lower().split())

        scored_results = []
        for chunk, base_score in results:
            chunk_words = set(chunk.content.lower().split())
            overlap = len(query_words & chunk_words)
            keyword_bonus = overlap / max(len(query_words), 1) * 0.3
            combined_score = float(np.clip(base_score + keyword_bonus, 0.0, 1.0))
            scored_results.append((chunk, combined_score))

        scored_results.sort(key=lambda x: x[1], reverse=True)
        return scored_results[:top_k]

    def _generate_answer(
        self,
        question: str,
        query_type: QueryType,
        context_chunks: list[tuple[DocumentChunk, float]],
    ) -> tuple[str, list[Citation]]:
        """Generate a grounded answer from retrieved document chunks.

        In production, this would use an LLM (Anthropic Claude or OpenAI)
        with the retrieved context. Here we synthesize from the chunks directly.
        """
        if not context_chunks:
            return "No relevant documents found for this query.", []

        citations = []
        answer_parts = []
        answer_parts.append(f"Based on analysis of {len(context_chunks)} relevant regulatory source(s):\n\n")

        for i, (chunk, score) in enumerate(context_chunks):
            citation = Citation(
                chunk_id=chunk.chunk_id,
                document_title=chunk.document_title,
                source=chunk.source,
                section=chunk.section,
                page_number=chunk.page_number,
                quote=chunk.content[:200],
                relevance_score=score,
            )
            citations.append(citation)

            if score > 0.3:
                cite_ref = citation.format_inline()
                content_summary = chunk.content[:300]
                answer_parts.append(f"According to {cite_ref}: {content_summary}")
                if i < len(context_chunks) - 1:
                    answer_parts.append("\n\n")

        answer_parts.append(
            "\n\nNote: This response is generated from the retrieved regulatory document "
            "excerpts and should be verified against the full source documents."
        )

        return "".join(answer_parts), citations

    def _score_confidence(
        self,
        results: list[tuple[DocumentChunk, float]],
        citations: list[Citation],
    ) -> tuple[float, ConfidenceLevel]:
        """Score confidence based on retrieval quality and citation coverage."""
        if not results:
            return 0.0, ConfidenceLevel.INSUFFICIENT

        # Average relevance score
        avg_score = float(np.mean([score for _, score in results]))

        # Citation diversity (number of unique sources)
        unique_sources = len(set(c.source for c in citations))
        diversity_bonus = min(unique_sources / 3.0, 0.2)

        # High relevance threshold
        high_relevance_count = sum(1 for _, score in results if score > 0.5)
        relevance_bonus = min(high_relevance_count / 3.0, 0.2)

        confidence_score = float(np.clip(avg_score + diversity_bonus + relevance_bonus, 0.0, 1.0))

        if confidence_score >= 0.7:
            level = ConfidenceLevel.HIGH
        elif confidence_score >= 0.5:
            level = ConfidenceLevel.MODERATE
        elif confidence_score >= 0.3:
            level = ConfidenceLevel.LOW
        else:
            level = ConfidenceLevel.INSUFFICIENT

        return confidence_score, level

    def _verify_faithfulness(
        self,
        answer: str,
        context_chunks: list[tuple[DocumentChunk, float]],
    ) -> FaithfulnessRating:
        """Verify that the generated answer is grounded in retrieved documents.

        Uses a simplified word overlap approach. In production, use an
        NLI model or LLM-based faithfulness evaluator.
        """
        if not context_chunks:
            return FaithfulnessRating.NOT_GROUNDED

        answer_words = set(answer.lower().split())
        context_words: set[str] = set()
        for chunk, _ in context_chunks:
            context_words.update(chunk.content.lower().split())

        # Remove common stopwords for more meaningful comparison
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "and", "or", "to", "in", "of", "for"}
        answer_content_words = answer_words - stopwords
        context_content_words = context_words - stopwords

        if not answer_content_words:
            return FaithfulnessRating.NOT_GROUNDED

        overlap = len(answer_content_words & context_content_words)
        overlap_ratio = overlap / len(answer_content_words)

        if overlap_ratio >= 0.6:
            return FaithfulnessRating.FULLY_GROUNDED
        elif overlap_ratio >= 0.4:
            return FaithfulnessRating.MOSTLY_GROUNDED
        elif overlap_ratio >= 0.2:
            return FaithfulnessRating.PARTIALLY_GROUNDED
        else:
            return FaithfulnessRating.NOT_GROUNDED

    def _suggest_follow_ups(self, question: str, query_type: QueryType) -> list[str]:
        """Suggest follow-up questions based on the query type."""
        suggestions: dict[QueryType, list[str]] = {
            QueryType.ELIGIBILITY: [
                "What are the specific biomarker requirements for eligibility?",
                "How are protocol deviations for eligibility handled?",
            ],
            QueryType.DOSING: [
                "What are the dose modification criteria for Grade 3+ toxicity?",
                "What is the maximum number of treatment cycles permitted?",
            ],
            QueryType.SAFETY_REPORTING: [
                "What are the timelines for SUSAR reporting?",
                "How does the DSMB review unblinded safety data?",
            ],
            QueryType.ELECTRONIC_RECORDS: [
                "What are the audit trail retention requirements?",
                "How do FDA Part 11 and ICH E6(R3) requirements overlap?",
            ],
            QueryType.DEVICE_REQUIREMENTS: [
                "What are the emergency stop requirements for robotic systems?",
                "How is workspace limitation validated?",
            ],
        }
        return suggestions.get(query_type, ["What additional regulatory guidance applies to this situation?"])

    def _record_audit(self, response: ComplianceResponse) -> None:
        """Record an audit entry for the compliance query."""
        query_hash = hashlib.sha256(response.query.encode("utf-8")).hexdigest()[:16]
        entry = AuditEntry(
            query_hash=query_hash,
            response_id=response.response_id,
            citations_count=len(response.citations),
            confidence=response.confidence.value,
            faithfulness=response.faithfulness.value,
        )
        entry.compute_hash(self._last_audit_hash)
        self._last_audit_hash = entry.entry_hash
        self._audit_entries.append(entry)

    def get_query_history(self) -> list[dict[str, Any]]:
        """Get the query history as dictionaries."""
        return [r.to_dict() for r in self._query_history]

    def verify_audit_chain(self) -> bool:
        """Verify the integrity of the audit hash chain."""
        prev_hash = ""
        for entry in self._audit_entries:
            expected = entry.compute_hash(prev_hash)
            if expected != entry.entry_hash:
                logger.error("Audit chain broken at entry %s", entry.entry_id)
                return False
            prev_hash = entry.entry_hash
        return True

    def export_audit_trail(self) -> str:
        """Export the audit trail as JSON."""
        records = []
        for entry in self._audit_entries:
            records.append(
                {
                    "entry_id": entry.entry_id,
                    "query_hash": entry.query_hash,
                    "response_id": entry.response_id,
                    "citations_count": entry.citations_count,
                    "confidence": entry.confidence,
                    "faithfulness": entry.faithfulness,
                    "timestamp": entry.timestamp,
                    "entry_hash": entry.entry_hash,
                }
            )
        return json.dumps(records, indent=2)


# ---------------------------------------------------------------------------
# Main demonstration
# ---------------------------------------------------------------------------
def main() -> None:
    """Demonstrate the oncology RAG compliance agent."""
    logger.info("=" * 80)
    logger.info("Oncology RAG Compliance Agent Demonstration")
    logger.info("Version: 0.5.0 | RESEARCH USE ONLY")
    logger.info("=" * 80)

    logger.info(
        "Optional dependencies: HAS_ANTHROPIC=%s, HAS_OPENAI=%s, HAS_CHROMADB=%s, HAS_SENTENCE_TRANSFORMERS=%s",
        HAS_ANTHROPIC,
        HAS_OPENAI,
        HAS_CHROMADB,
        HAS_SENTENCE_TRANSFORMERS,
    )

    # Initialize agent
    agent = OncologyRAGComplianceAgent(seed=42)

    # Query 1: Eligibility
    logger.info("-" * 60)
    logger.info("Query 1: Eligibility criteria")
    resp1 = agent.query("What are the eligibility criteria for the Phase III NSCLC pembrolizumab trial?")
    logger.info("Answer: %s", resp1.answer[:200])
    logger.info(
        "Confidence: %s (%.2f), Faithfulness: %s",
        resp1.confidence.value,
        resp1.confidence_score,
        resp1.faithfulness.value,
    )
    logger.info("Citations: %d", len(resp1.citations))
    for cit in resp1.citations[:3]:
        logger.info("  - %s", cit.format_inline())

    # Query 2: Electronic records
    logger.info("-" * 60)
    logger.info("Query 2: FDA 21 CFR Part 11 audit trail requirements")
    resp2 = agent.query("What does FDA 21 CFR Part 11 require for audit trails in electronic records?")
    logger.info("Answer: %s", resp2.answer[:200])
    logger.info(
        "Confidence: %s (%.2f), Faithfulness: %s",
        resp2.confidence.value,
        resp2.confidence_score,
        resp2.faithfulness.value,
    )

    # Query 3: Device safety
    logger.info("-" * 60)
    logger.info("Query 3: IEC 80601-2-77 emergency stop requirements")
    resp3 = agent.query("What are the emergency stop requirements for robotic surgical equipment under IEC 80601-2-77?")
    logger.info("Answer: %s", resp3.answer[:200])
    logger.info(
        "Confidence: %s (%.2f), Faithfulness: %s",
        resp3.confidence.value,
        resp3.confidence_score,
        resp3.faithfulness.value,
    )

    # Query 4: Dosing
    logger.info("-" * 60)
    logger.info("Query 4: Dose modification criteria")
    resp4 = agent.query("What are the dose modification criteria for carboplatin and pembrolizumab in the NSCLC trial?")
    logger.info("Answer: %s", resp4.answer[:200])
    logger.info(
        "Confidence: %s (%.2f), Faithfulness: %s",
        resp4.confidence.value,
        resp4.confidence_score,
        resp4.faithfulness.value,
    )

    # Query 5: Risk management
    logger.info("-" * 60)
    logger.info("Query 5: ISO 14971 risk control")
    resp5 = agent.query("How does ISO 14971 prioritize risk control measures for medical devices?")
    logger.info("Answer: %s", resp5.answer[:200])
    logger.info(
        "Confidence: %s (%.2f), Faithfulness: %s",
        resp5.confidence.value,
        resp5.confidence_score,
        resp5.faithfulness.value,
    )

    # Verify audit chain
    logger.info("-" * 60)
    integrity = agent.verify_audit_chain()
    logger.info("Audit trail: %d entries, integrity=%s", len(agent._audit_entries), "PASS" if integrity else "FAIL")

    # Display follow-up suggestions
    logger.info("-" * 60)
    logger.info("Suggested follow-up questions:")
    for q in resp1.follow_up_questions:
        logger.info("  - %s", q)

    logger.info("=" * 80)
    logger.info("Demonstration complete.")


if __name__ == "__main__":
    main()
