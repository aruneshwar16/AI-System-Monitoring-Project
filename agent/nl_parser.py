"""Natural language query parser with enhanced rule-based and Gemini support."""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from prompts.gemini_prompts import INTENT_DETECTION_PROMPT
from utils.constants import (
    DEFAULT_FORECAST_MONTHS,
    DEFAULT_THRESHOLD,
    METRIC_CPU,
    METRIC_DISK,
    METRIC_MEMORY,
    QueryIntent,
    SUPPORTED_METRICS,
)
from utils.gemini_client import GeminiClient
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ParsedQuery:
    """Structured representation of a user query."""

    raw_query: str
    intent: QueryIntent
    metrics: list[str] = field(default_factory=list)
    threshold: float = DEFAULT_THRESHOLD
    forecast_months: int = DEFAULT_FORECAST_MONTHS
    growth_multiplier: float = 1.0
    parse_confidence: float = 0.85
    source: str = "gemini"
    time_unit: str = "months"
    compare_type: Optional[str] = None  # e.g., "faster", "slower", "greater"


class NLParser:
    """Parses natural language capacity planning queries with robust rule-based fallback."""

    def __init__(self, api_key: Optional[str] = None):
        self.gemini = GeminiClient(api_key=api_key)

    def parse(self, query: str) -> ParsedQuery:
        """Parse user query into structured parameters."""
        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty.")

        try:
            return self._parse_with_gemini(query)
        except Exception as exc:
            logger.warning("Gemini parsing failed (%s). Using rule-based fallback.", exc)
            return self._parse_rule_based(query)

    def _parse_with_gemini(self, query: str) -> ParsedQuery:
        prompt = INTENT_DETECTION_PROMPT.format(query=query)
        text = self.gemini.generate(prompt)
        data = self._extract_json(text)
        return self._build_parsed_query(query, data, source="gemini")

    def _parse_rule_based(self, query: str) -> ParsedQuery:
        """Enhanced fallback parser with broad natural language coverage."""
        q = query.lower().strip()
        
        # Extract core parameters
        metrics = self._extract_metrics_rule(q)
        threshold = self._extract_threshold_rule(q)
        forecast_months = self._extract_months_rule(q)
        growth_multiplier = self._extract_growth_rule(q)
        
        # Detect intent with comprehensive pattern matching
        intent, detected_metrics = self._detect_intent(q, metrics)
        
        # Use detected metrics from intent if broader scope needed
        if detected_metrics and not metrics:
            metrics = detected_metrics
        if not metrics:
            metrics = [METRIC_CPU]

        # Determine compare type for comparison queries
        compare_type = self._extract_compare_type(q)

        return ParsedQuery(
            raw_query=query,
            intent=intent,
            metrics=metrics,
            threshold=threshold,
            forecast_months=forecast_months,
            growth_multiplier=growth_multiplier,
            parse_confidence=0.7 if intent != QueryIntent.GENERAL else 0.5,
            source="rule_based",
            compare_type=compare_type,
        )

    def _detect_intent(self, q: str, detected_metrics: list[str]) -> tuple[QueryIntent, list[str]]:
        """Detect intent using comprehensive pattern matching."""
        
        # ========== THRESHOLD CROSSING ==========
        # "when will X hit Y%", "reach", "exceed", "cross", "breach"
        threshold_patterns = [
            r"when\s+(will|does|is|would)\s+(\w+\s+)?(hit|reach|exceed|cross|breach|touch)",
            r"(hit|reach|exceed|cross|breach|touch)\s+(a\s+)?\d+\s*%",
            r"(threshold|crossing|breach)",
            r"how\s+(long|soon)\s+(before|until|till)\s+(\w+\s+)?(hit|reach|exceed)",
            r"will\s+(\w+\s+)?(hit|reach|exceed|cross)\s+\d+",
            r"time\s+(to|until|before)\s+(hit|reach|exceed|cross)",
            r"when\s+will\s+(it|this|that|the)\s+(hit|reach|exceed|cross)",
            r"(\w+\s+usage\s+)?(hits|reaches|exceeds|crosses|breaches)\s+\d+",
            r"at\s+what\s+(point|time|date|moment)\s+(will|does)\s+",
        ]
        if any(re.search(p, q) for p in threshold_patterns):
            return QueryIntent.THRESHOLD_CROSSING, detected_metrics or list(SUPPORTED_METRICS)

        # ========== WHAT-IF / SCENARIO ANALYSIS ==========
        what_if_patterns = [
            r"what\s+if",
            r"what.?if",
            r"(scenario|hypothetical|simulate?)\s+",
            r"(grows?|grown|increase[d]?|decrease[d]?|accelerate[d]?|slow[s]?ed?)\s+\d+.*%",
            r"\d+.*%\s+faster",
            r"\d+.*%\s+slower",
            r"(double|triple|halve)\s+(the\s+)?(growth|rate|speed)",
            r"suppose\s+",
            r"imagine\s+",
            r"if\s+(we\s+)?(increase|decrease|grow|slow|accelerate)",
            r"((under|with|using)\s+)?(worst|best|optimistic|pessimistic)\s+case",
        ]
        if any(re.search(p, q) for p in what_if_patterns):
            return QueryIntent.WHAT_IF, detected_metrics or [METRIC_CPU]

        # ========== COMPARE RESOURCES ==========
        compare_patterns = [
            r"(which|what)\s+(resource|metric|one|component).*(first|soonest|earliest|before)",
            r"which\s+.*(exceed|hit|reach|cross)\s+(threshold|limit|capacity)",
            r"compare\s+(all\s+)?(resources|metrics|usage)",
            r"(comparison|compare|versus|vs\.?)\s+(between|of|all)",
            r"which\s+(is|has|shows)\s+(the\s+)?(highest|lowest|most|worst|best)",
            r"(top|worst|most|biggest|largest|highest)\s+(risk|usage|concern)",
            r"rank\s+(the\s+)?(resources|metrics|usage)",
            r"prioritize\s+",
            r"(difference|differ)\s+(between|among)\s+",
            r"which\s+(one|resource|metric)\s+(will\s+)?(be\s+)?(first|soonest)",
        ]
        if any(re.search(p, q) for p in compare_patterns):
            return QueryIntent.COMPARE_RESOURCES, list(SUPPORTED_METRICS)

        # ========== RISK ANALYSIS ==========
        risk_patterns = [
            r"\brisk\b",
            r"(risk\s+analysis|risk\s+assessment|risk\s+report|risk\s+evaluation)",
            r"(analyze|assess|evaluate|check)\s+(all\s+)?(risk|threat|danger|vulnerability)",
            r"how\s+(risky|dangerous|vulnerable)\s+",
            r"(what\s+are\s+the|show\s+me\s+the|list\s+the)\s+(risks|concerns|issues)",
            r"(health|status|state)\s+(check|report|overview|summary)",
            r"(overall|general|total)\s+(health|status|condition|state)",
            r"is\s+(everything|anything|something)\s+(ok|okay|fine|good|alright)",
            r"should\s+(I|we)\s+be\s+(worried|concerned|alarmed)",
        ]
        if any(re.search(p, q) for p in risk_patterns):
            return QueryIntent.RISK_ANALYSIS, list(SUPPORTED_METRICS)

        # ========== TREND ANALYSIS ==========
        trend_patterns = [
            r"(trend|trending|direction|pattern|trajectory)",
            r"is\s+(\w+\s+usage)\s+(increasing|decreasing|growing|declining|rising|falling|stable)",
            r"how\s+(is|are)\s+(\w+\s+usage)\s+(changing|trending|behaving|performing)",
            r"(upward|downward|upwards|downwards)\s+(trend|direction|movement)",
            r"(slope|rate)\s+of\s+(growth|change|increase|decrease)",
            r"(growing|declining|rising|falling)\s+(faster|slower|steadily|rapidly|gradually)",
            r"(acceleration|deceleration)\s+of",
            r"(show|plot|graph|visualize|chart)\s+(the\s+)?(trend|pattern|trajectory)",
        ]
        if any(re.search(p, q) for p in trend_patterns):
            # For trend, specific metrics are important
            return QueryIntent.FORECAST_PERIOD, detected_metrics or list(SUPPORTED_METRICS)

        # ========== FORECAST / PREDICTION ==========
        forecast_patterns = [
            r"(forecast|predict|project|estimate|anticipate|expect|outlook)",
            r"next\s+\d+\s+(month|week|year|quarter|day)",
            r"(coming|upcoming|future|next)\s+\d+\s+(month|week|year|quarter|day)s?",
            r"(how\s+much|how\s+far)\s+(will|would|might|could)\s+(\w+\s+usage)",
            r"(what|how)\s+(will|would|could|might)\s+(\w+\s+usage)\s+(be|look|become)",
            r"(look\s+ahead|look\s+forward|outlook|projection)",
            r"(short.?term|long.?term|medium.?term)\s+(forecast|prediction|outlook)",
            r"where\s+(will|is|are)\s+(\w+\s+usage)\s+(be|going|headed|heading)",
            r"(\w+\s+usage)\s+(in|by|for)\s+\d+\s+(month|week|year|quarter)s?",
        ]
        if any(re.search(p, q) for p in forecast_patterns):
            return QueryIntent.FORECAST_PERIOD, detected_metrics or list(SUPPORTED_METRICS)

        # ========== CAPACITY PLANNING ==========
        capacity_patterns = [
            r"(capacity|scaling|scale|upgrade|expand|expansion|provisioning)",
            r"(do\s+I|do\s+we|should\s+I|should\s+we|need\s+to)\s+(need|add|increase|upgrade|expand|scale)",
            r"(how\s+much|enough)\s+(capacity|headroom|room|space|resources)",
            r"(outgrow|outgrown|running\s+out|run\s+out\s+of|exhaust)",
            r"(capacity\s+plan|capacity\s+planning|resource\s+planning)",
            r"(insufficient|not\s+enough|too\s+little|lacking)",
            r"(bottleneck|bottlenecks|constraint|constraints)",
            r"(plan\s+for|planning\s+for|prepare\s+for)\s+(growth|expansion|future)",
        ]
        if any(re.search(p, q) for p in capacity_patterns):
            return QueryIntent.RISK_ANALYSIS, list(SUPPORTED_METRICS)

        # ========== ANOMALY / SPIKE DETECTION ==========
        anomaly_patterns = [
            r"(anomaly|anomalies|outlier|outliers|spike|spikes|abnormal|unusual)",
            r"(sudden|unexpected|unusual|abnormal|irregular)\s+(change|increase|decrease|drop|rise|spike)",
            r"(something\s+)?(wrong|off|weird|strange|odd)\s+(with|in)",
            r"(detect|find|identify|spot|flag)\s+(anomalies|problems|issues|outliers)",
            r"(is\s+there|are\s+there|any)\s+(anomalies|spikes|outliers|irregularities)",
        ]
        if any(re.search(p, q) for p in anomaly_patterns):
            return QueryIntent.GENERAL, detected_metrics or list(SUPPORTED_METRICS)

        # ========== RECOMMENDATION / ADVICE ==========
        recommendation_patterns = [
            r"(recommend|suggest|advise|guidance|tip|recommendation)",
            r"(what\s+should|what\s+would\s+you|any\s+suggestion)",
            r"(how\s+can\s+I|how\s+can\s+we|best\s+way\s+to)\s+(improve|optimize|fix|address|handle)",
            r"(action|actions|steps|measures)\s+(plan|to\s+take|needed|required)",
            r"(give\s+me|provide)\s+(some\s+)?(advice|recommendations|suggestions)",
        ]
        if any(re.search(p, q) for p in recommendation_patterns):
            return QueryIntent.RISK_ANALYSIS, list(SUPPORTED_METRICS)

        # ========== EXPLANATION / HOW-TO ==========
        explanation_patterns = [
            r"(how\s+(does|do|can|is|are|would|will)|explain|what\s+is|what\s+are|describe)",
            r"(tell\s+me\s+about|help\s+me\s+understand|curious\s+about)",
            r"(how\s+this\s+works|how\s+does\s+this\s+work)",
            r"(what\s+does\s+it\s+mean|meaning\s+of|definition)",
        ]
        if any(re.search(p, q) for p in explanation_patterns):
            # Check if it's a how-to about the tool itself
            if any(w in q for w in ["tool", "app", "this", "you", "system", "platform", "software"]):
                return QueryIntent.GENERAL, [METRIC_CPU]
            # Otherwise treat as forecast/analysis question
            return QueryIntent.FORECAST_PERIOD, detected_metrics or list(SUPPORTED_METRICS)

        # ========== GENERAL / DEFAULT ==========
        # Check for any metric-related question
        if detected_metrics:
            # Check if it's a simple status question
            if any(w in q for w in ["what", "how", "show", "tell", "current", "now", "status", "state"]):
                return QueryIntent.GENERAL, detected_metrics
        
        return QueryIntent.GENERAL, detected_metrics or [METRIC_CPU]

    def _build_parsed_query(self, query: str, data: dict[str, Any], source: str) -> ParsedQuery:
        intent_str = data.get("intent", "general")
        try:
            intent = QueryIntent(intent_str)
        except ValueError:
            intent = QueryIntent.GENERAL

        metrics = data.get("metrics") or [METRIC_CPU]
        metrics = [m for m in metrics if m in SUPPORTED_METRICS]
        if not metrics:
            metrics = [METRIC_CPU]

        if intent in (QueryIntent.COMPARE_RESOURCES, QueryIntent.RISK_ANALYSIS):
            metrics = list(SUPPORTED_METRICS)

        threshold = float(data.get("threshold") or DEFAULT_THRESHOLD)
        forecast_months = int(data.get("forecast_months") or DEFAULT_FORECAST_MONTHS)
        growth_multiplier = float(data.get("growth_multiplier") or 1.0)
        parse_confidence = float(data.get("confidence") or 0.85)

        return ParsedQuery(
            raw_query=query,
            intent=intent,
            metrics=metrics,
            threshold=threshold,
            forecast_months=min(max(forecast_months, 1), 24),
            growth_multiplier=max(0.1, growth_multiplier),
            parse_confidence=parse_confidence,
            source=source,
        )

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return json.loads(text)

    @staticmethod
    def _extract_metrics_rule(q: str) -> list[str]:
        """Extract mentioned metrics with broader keyword matching."""
        found = []
        
        # CPU patterns
        cpu_patterns = [
            r"\bcpu\b", r"\bcpus?\b", r"\bprocessor\b", r"\bprocessors?\b",
            r"\bcompute\b", r"\bcores?\b", r"\bvcores?\b", r"\bcpu\s+usage\b",
            r"\bcpu\s+utilization\b", r"\bcpu\s+load\b", r"\bprocessing\b",
        ]
        if any(re.search(p, q) for p in cpu_patterns):
            found.append(METRIC_CPU)
        
        # Memory patterns
        memory_patterns = [
            r"\bmemory\b", r"\bram\b", r"\bram\s+usage\b", r"\bmemory\s+usage\b",
            r"\bmemory\s+utilization\b", r"\bmem\b", r"\bheap\b",
        ]
        if any(re.search(p, q) for p in memory_patterns):
            found.append(METRIC_MEMORY)
        
        # Disk patterns
        disk_patterns = [
            r"\bdisk\b", r"\bstorage\b", r"\bdrive\b", r"\bhd[d]?\b",
            r"\bss[d]?\b", r"\bdisk\s+usage\b", r"\bstorage\s+usage\b",
            r"\bdisc\b", r"\bvolume\b", r"\bfilesystem\b", r"\bfs\b",
            r"\bpartition\b", r"\bdisk\s+space\b", r"\bstorage\s+space\b",
            r"\bcapacity\b",
        ]
        if any(re.search(p, q) for p in disk_patterns):
            found.append(METRIC_DISK)
        
        # If "all" or "everything" or "both" or general terms
        if any(w in q for w in ["all", "everything", "each", "every", "overall"]):
            return list(SUPPORTED_METRICS)
        
        return found

    @staticmethod
    def _extract_threshold_rule(q: str) -> float:
        """Extract threshold percentage with broader pattern matching."""
        # Skip if preceded by "grow/growth/increase/decrease" (those are growth rates)
        # Pattern: "X%" - but not when it's a growth rate modifier
        # First check if this is a what-if/growth query — skip threshold from growth rates
        is_growth_query = bool(re.search(
            r"(grows?|grown|growth|increase[d]?|decrease[d]?|accelerate|slow|faster|slower)\s+(\d+(?:\.\d+)?)\s*%",
            q
        ))
        
        match = re.search(r"(\d+(?:\.\d+)?)\s*%", q)
        if match:
            val = float(match.group(1))
            # If this looks like a growth rate (part of "X% faster/grow/increase"), skip it
            growth_context = re.search(
                r"(?:grows?|grown|growth|faster|slower|increase|decrease|accelerate|reduce)"
                r"\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*%",
                q
            )
            if growth_context and float(growth_context.group(1)) == val:
                # This % is a growth rate, not a threshold — skip
                pass
            elif 0 < val <= 100:
                return val
        
        # Pattern: "hit/reach/exceed X" 
        match = re.search(r"(?:hit|reach|exceed|cross|breach|at|touch|of)\s+(\d+)(?!\s*%)", q)
        if match:
            val = float(match.group(1))
            if 0 < val <= 100:
                return val
        
        # Pattern: "X percent" 
        match = re.search(r"(\d+(?:\.\d+)?)\s*percent", q)
        if match:
            val = float(match.group(1))
            if 0 < val <= 100:
                return val
        
        return DEFAULT_THRESHOLD

    @staticmethod
    def _extract_months_rule(q: str) -> int:
        """Extract forecast duration with support for various time units."""
        # Explicit months
        match = re.search(r"(\d+)\s*months?\b", q)
        if match:
            return int(match.group(1))
        
        # Quarters
        match = re.search(r"(\d+)\s*quarters?\b", q)
        if match:
            return int(match.group(1)) * 3
        
        # Years
        match = re.search(r"(\d+)\s*years?\b", q)
        if match:
            return int(match.group(1)) * 12
        
        # Weeks
        match = re.search(r"(\d+)\s*weeks?\b", q)
        if match:
            return max(1, int(match.group(1)) // 4)
        
        # Days
        match = re.search(r"(\d+)\s*days?\b", q)
        if match:
            return max(1, int(match.group(1)) // 30)
        
        # "next X" without time unit (default to months)
        match = re.search(r"next\s+(\d+)", q)
        if match:
            return int(match.group(1))
        
        # Short-term / long-term indicators
        if any(w in q for w in ["short.?term", "near.?term", "immediate", "soon"]):
            return 3
        if any(w in q for w in ["medium.?term", "mid.?term"]):
            return 6
        if any(w in q for w in ["long.?term", "far.?term", "distant"]):
            return 12
        
        return DEFAULT_FORECAST_MONTHS

    @staticmethod
    def _extract_growth_rule(q: str) -> float:
        """Extract growth multiplier with broader pattern matching."""
        # "X% faster" or "grow X% faster"
        match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*faster", q)
        if match:
            return 1.0 + float(match.group(1)) / 100.0
        
        # "grows X%"
        match = re.search(r"(?:grows?|grown|increase[ds]?|growing)\s+(\d+(?:\.\d+)?)\s*%", q)
        if match:
            return 1.0 + float(match.group(1)) / 100.0
        
        # "increase by X%" 
        match = re.search(r"(?:increase|decrease|reduce|slow|accelerate)\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*%", q)
        if match:
            val = float(match.group(1))
            # Slower/reduce means growth < 1.0
            if any(w in q for w in ["decrease", "reduce", "slow", "slower"]):
                return max(0.1, 1.0 - val / 100.0)
            return 1.0 + val / 100.0
        
        # "double" = 2x growth
        if "double" in q:
            return 2.0
        # "triple" = 3x growth
        if "triple" in q:
            return 3.0
        
        return 1.0

    @staticmethod
    def _extract_compare_type(q: str) -> Optional[str]:
        """Extract comparison type."""
        if any(w in q for w in ["faster", "fast", "accelerate"]):
            return "faster"
        if any(w in q for w in ["slower", "slow", "decelerate"]):
            return "slower"
        if any(w in q for w in ["greater", "larger", "higher", "bigger"]):
            return "greater"
        if any(w in q for w in ["less", "lower", "smaller", "fewer"]):
            return "less"
        return None