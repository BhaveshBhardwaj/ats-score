"""
NEXUS Multi-Agent System — Base Classes

Defines the core abstractions for the adversarial multi-agent
candidate evaluation architecture:
  - Evidence: A single piece of evidence about a candidate
  - Verdict: An agent's complete evaluation of a candidate
  - BaseAgent: Abstract base for all evaluation agents
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class EvidenceType(Enum):
    """Classification of evidence strength."""
    CORROBORATED = "corroborated"      # Confirmed by multiple sources
    SELF_REPORTED = "self_reported"     # Only claimed by candidate
    INFERRED = "inferred"              # Derived from indirect signals
    CONTRADICTED = "contradicted"      # Contradicted by other evidence
    FORENSIC = "forensic"              # From cross-validation analysis


class EvidencePolarity(Enum):
    """Whether evidence supports or opposes hiring."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class Evidence:
    """
    A single piece of evidence about a candidate.
    
    The fundamental unit of the NEXUS system. Each agent produces
    evidence, and the debate protocol challenges/validates it.
    """
    claim: str                              # What is being claimed
    source: str                             # Where the evidence comes from
    polarity: EvidencePolarity              # positive/negative/neutral
    strength: float                         # 0.0-1.0 raw strength
    credibility: float = 1.0               # 0.0-1.0 credibility after debate
    evidence_type: EvidenceType = EvidenceType.SELF_REPORTED
    agent_id: str = ""                      # Which agent produced this
    challenge_survived: bool = False         # Whether it survived debate
    details: str = ""                       # Human-readable explanation
    
    @property
    def effective_weight(self) -> float:
        """Effective weight = strength × credibility × polarity modifier."""
        polarity_mod = 1.0 if self.polarity == EvidencePolarity.POSITIVE else (
            -1.0 if self.polarity == EvidencePolarity.NEGATIVE else 0.0
        )
        return self.strength * self.credibility * polarity_mod


@dataclass
class Verdict:
    """
    An agent's complete evaluation of a candidate.
    Contains the score, evidence chain, and reasoning.
    """
    agent_id: str
    candidate_id: str
    score: float                            # 0.0-1.0 agent's score
    confidence: float                       # 0.0-1.0 how confident the agent is
    evidence: List[Evidence] = field(default_factory=list)
    reasoning: str = ""                     # Human-readable reasoning
    
    @property
    def weighted_score(self) -> float:
        """Score adjusted by confidence."""
        return self.score * self.confidence
    
    def top_evidence(self, n: int = 3) -> List[Evidence]:
        """Return top N evidence items by effective weight magnitude."""
        return sorted(
            self.evidence,
            key=lambda e: abs(e.effective_weight),
            reverse=True
        )[:n]


class BaseAgent:
    """
    Abstract base class for all NEXUS evaluation agents.
    
    Each agent independently evaluates a candidate and produces
    a Verdict with supporting Evidence. Agents are designed to
    have different perspectives and biases, which are then
    resolved through the adversarial debate protocol.
    """
    
    AGENT_ID = "base"
    
    def evaluate(self, candidate: dict) -> Verdict:
        """
        Evaluate a candidate and return a Verdict.
        Must be implemented by subclasses.
        """
        raise NotImplementedError
    
    def challenge(self, evidence: Evidence, candidate: dict) -> Optional[Evidence]:
        """
        Challenge a piece of evidence from another agent.
        Returns counter-evidence if the challenge succeeds, None if it fails.
        Default implementation: no challenges.
        """
        return None
    
    def _make_evidence(
        self,
        claim: str,
        source: str,
        polarity: EvidencePolarity,
        strength: float,
        evidence_type: EvidenceType = EvidenceType.SELF_REPORTED,
        details: str = "",
    ) -> Evidence:
        """Helper to create Evidence with this agent's ID."""
        return Evidence(
            claim=claim,
            source=source,
            polarity=polarity,
            strength=min(1.0, max(0.0, strength)),
            evidence_type=evidence_type,
            agent_id=self.AGENT_ID,
            details=details,
        )
