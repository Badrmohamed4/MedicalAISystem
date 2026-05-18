"""
LangGraph Medical AI Pipeline
==============================
Graph-based orchestration for the medical chatbot.

Nodes:
  1. sanitize_node      — blocks prompt injection (no LLM)
  2. safety_node        — detects urgency keywords (no LLM)
  3. nlp_extract_node   — BioBERT + Ollama extraction (LLM)
  4. normalize_node     — SapBERT normalization (model, no LLM)
  5. risk_assess_node   — rule-based risk scoring (no LLM)
  6. response_node      — Ollama response generation (LLM)

Only nodes 3 and 6 use an LLM.
"""

import os
import sys
from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END
import time

# ── Pipeline Logger ───────────────────────────────────────────────────────────
class PipelineLogger:
    """Tracks timing and state for each LangGraph node."""

    def __init__(self):
        self._start_times = {}
        self._request_start = None

    def request_start(self):
        self._request_start = time.time()
        print("\n" + "─" * 49)
        print(" LANGGRAPH PIPELINE — NEW REQUEST")
        print("─" * 49)

    def node_start(self, node_name: str):
        self._start_times[node_name] = time.time()

    def node_end(self, node_name: str, details: str = ""):
        elapsed = time.time() - self._start_times.get(node_name, time.time())
        print(f"[{node_name:<12}] ✅ {elapsed:5.3f}s | {details}")

    def node_blocked(self, node_name: str, reason: str = ""):
        elapsed = time.time() - self._start_times.get(node_name, time.time())
        print(f"[{node_name:<12}] 🚫 {elapsed:5.3f}s | BLOCKED — {reason}")

    def node_urgent(self, node_name: str):
        elapsed = time.time() - self._start_times.get(node_name, time.time())
        print(f"[{node_name:<12}] ⚠️  {elapsed:5.3f}s | URGENT — emergency path")

    def request_end(self):
        if self._request_start:
            total = time.time() - self._request_start
            print("─" * 49)
            print(f" TOTAL: {total:.3f}s")
            print("─" * 49 + "\n")

_logger = PipelineLogger()

# ── Path setup ────────────────────────────────────────────────────────────────
_pipeline_dir = os.path.dirname(os.path.abspath(__file__))
_ai_project_dir = os.path.dirname(_pipeline_dir)
_project_root = os.path.dirname(_ai_project_dir)

sys.path.insert(0, _ai_project_dir)
sys.path.insert(0, _project_root)

# ── Lazy imports (loaded once at module level) ─────────────────────────────────
_nlp_processor = None
_ollama_client = None


def _get_nlp_processor():
    global _nlp_processor
    if _nlp_processor is None:
        from nlp.processor import NLPProcessor
        _nlp_processor = NLPProcessor()
    return _nlp_processor


def _get_ollama_client():
    global _ollama_client
    if _ollama_client is None:
        from llm.ollama_client import OllamaClient
        _ollama_client = OllamaClient()
    return _ollama_client


# ── Urgency keywords (deterministic — no LLM needed) ──────────────────────────
URGENCY_KEYWORDS = [
    "can't breathe", "cannot breathe", "chest pain", "heart attack",
    "stroke", "unconscious", "seizure", "bleeding heavily", "severe pain",
    "emergency", "dying", "collapsed", "overdose", "suicidal",
]


# ══════════════════════════════════════════════════════════════════════════════
# STATE DEFINITION
# Each node receives this state dict and returns an updated version.
# ══════════════════════════════════════════════════════════════════════════════

class MedicalState(TypedDict):
    # Input
    raw_input: str
    session_context: dict

    # After sanitize_node
    clean_input: str
    is_safe: bool
    unsafe_reason: str

    # After safety_node
    is_urgent: bool

    # After nlp_extract_node
    intent: str
    symptoms: List[str]
    medical_context: str
    severity: Optional[str]
    duration: Optional[str]

    # After normalize_node
    normalized_symptoms: List[str]

    # After risk_assess_node
    risk_level: str

    # After response_node
    response: str


# ══════════════════════════════════════════════════════════════════════════════
# NODE 1 — SANITIZE (no LLM)
# Blocks prompt injection before anything reaches the LLM.
# ══════════════════════════════════════════════════════════════════════════════

def sanitize_node(state: MedicalState) -> MedicalState:
    """Sanitizes raw input. Blocks injection attempts."""
    _logger.node_start("sanitize")
    try:
        from medical_chatbot.utils.input_sanitizer import sanitize_input
    except ImportError:
        sys.path.insert(0, _project_root)
        from medical_chatbot.utils.input_sanitizer import sanitize_input

    result = sanitize_input(state["raw_input"])

    if not result["is_safe"]:
        _logger.node_blocked("sanitize", result.get("reason", "injection detected"))
    else:
        _logger.node_end("sanitize", f"safe=True | length={len(result['clean_text'])}")

    return {
        **state,
        "clean_input": result["clean_text"],
        "is_safe": result["is_safe"],
        "unsafe_reason": result.get("reason", ""),
    }

# ══════════════════════════════════════════════════════════════════════════════
# NODE 2 — SAFETY CHECK (no LLM)
# Detects medical emergencies before spending time on NLP.
# ══════════════════════════════════════════════════════════════════════════════

def safety_node(state: MedicalState) -> MedicalState:
    """Detects urgency keywords. No LLM — must be fast and deterministic."""
    _logger.node_start("safety")
    text_lower = state["clean_input"].lower()
    is_urgent = any(kw in text_lower for kw in URGENCY_KEYWORDS)

    if is_urgent:
        _logger.node_urgent("safety")
    else:
        _logger.node_end("safety", "urgent=False")

    return {
        **state,
        "is_urgent": is_urgent,
    }

# ══════════════════════════════════════════════════════════════════════════════
# NODE 3 — NLP EXTRACTION (uses Ollama LLM via BioBERT)
# Extracts intent, symptoms, context, severity, duration.
# ══════════════════════════════════════════════════════════════════════════════

def nlp_extract_node(state: MedicalState) -> MedicalState:
    """Runs BioBERT + Ollama extraction on clean input."""
    _logger.node_start("extract")
    try:
        processor = _get_nlp_processor()
        result = processor.process(state["clean_input"])

        symptoms = result.get("original_symptoms", [])
        intent = result.get("intent", "others")
        context = result.get("medical_context", "none")

        _logger.node_end("extract",
            f"intent={intent} | context={context} | symptoms={symptoms}")

        return {
            **state,
            "intent": intent,
            "symptoms": symptoms,
            "normalized_symptoms": result.get("normalized_symptoms", []),
            "medical_context": context,
            "severity": result.get("severity"),
            "duration": result.get("duration"),
        }
    except Exception as e:
        print(f"[extract      ] ❌ Error: {e}")
        return {
            **state,
            "intent": "others",
            "symptoms": [],
            "normalized_symptoms": [],
            "medical_context": "none",
            "severity": None,
            "duration": None,
        }

# ══════════════════════════════════════════════════════════════════════════════
# NODE 4 — NORMALIZATION (SapBERT — no LLM)
# Maps extracted symptoms to standard medical terminology.
# ══════════════════════════════════════════════════════════════════════════════

def normalize_node(state: MedicalState) -> MedicalState:
    """Pass-through — normalization already done in nlp_extract_node."""
    _logger.node_start("normalize")
    normalized = state.get("normalized_symptoms", [])
    if not normalized:
        normalized = state.get("symptoms", [])
    _logger.node_end("normalize", f"normalized={normalized}")
    return {
        **state,
        "normalized_symptoms": normalized,
    }
# ══════════════════════════════════════════════════════════════════════════════
# NODE 5 — RISK ASSESSMENT (no LLM — deterministic rules)
# Assigns low/medium/high risk based on symptoms and severity.
# ══════════════════════════════════════════════════════════════════════════════

HIGH_RISK_SYMPTOMS = [
    "seizure", "unconscious", "stroke", "heart attack",
    "myocardial infarction", "epileptic seizure", "syncope",
    "embolism", "thrombosis", "necrosis", "sepsis",
]

MEDIUM_RISK_SYMPTOMS = [
    "chest pain", "angina", "dyspnea", "shortness of breath",
    "cephalgia", "vertigo", "edema", "tumor", "neoplasm",
    "vision changes", "visual disturbance", "confusion", "disorientation",
]


def risk_assess_node(state: MedicalState) -> MedicalState:
    """Deterministic risk scoring. No LLM."""
    _logger.node_start("risk_assess")
    symptoms = state.get("normalized_symptoms", []) + state.get("symptoms", [])
    severity = state.get("severity", "")
    symptoms_lower = [s.lower() for s in symptoms]

    risk = "Low"
    if severity in ["severe", "extreme"]:
        risk = "High"
    elif severity == "moderate":
        risk = "Medium"

    for s in symptoms_lower:
        if any(h in s for h in HIGH_RISK_SYMPTOMS):
            risk = "High"
            break
        if any(m in s for m in MEDIUM_RISK_SYMPTOMS):
            if risk != "High":
                risk = "Medium"

    _logger.node_end("risk_assess", f"risk={risk} | severity={severity or 'none'}")
    return {
        **state,
        "risk_level": risk,
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 6 — RESPONSE GENERATION (uses Ollama LLM)
# Generates a contextual medical response based on full state.
# ══════════════════════════════════════════════════════════════════════════════

def response_node(state: MedicalState) -> MedicalState:
    """Generates a response using Ollama."""
    _logger.node_start("respond")
    ollama = _get_ollama_client()

    if not ollama.is_online():
        _logger.node_end("respond", "ollama=offline | using fallback")
        return {
            **state,
            "response": (
                "I'm having trouble connecting to my AI backend. "
                "Please describe your symptoms and I'll do my best to help."
            ),
        }

    symptoms_str = ", ".join(state.get("normalized_symptoms", [])
                             or state.get("symptoms", []))
    context = state.get("medical_context", "unknown")
    risk = state.get("risk_level", "Low")
    session_ctx = state.get("session_context", {})

    system_prompt = (
        "You are MediBot, a professional medical AI assistant. "
        "You specialize in brain tumors, lung cancer, and skin diseases. "
        "RULES:\n"
        "- Never provide a definitive diagnosis\n"
        "- Always recommend consulting a doctor for serious concerns\n"
        "- Be empathetic but professional\n"
        "- Respond in 2-3 sentences maximum\n"
        "- Focus on asking clarifying questions about symptoms\n"
        "- Never recommend specific medications\n"
        f"Patient context: {context} | Risk: {risk} | "
        f"Symptoms: {symptoms_str or 'none yet'}"
    )

    parts = []
    for chunk in ollama.stream_chat(system_prompt, state["clean_input"]):
        parts.append(chunk)

    response = "".join(parts) if parts else (
        "Could you describe your symptoms in more detail?"
    )

    _logger.node_end("respond",
        f"response_length={len(response)} chars | ollama=online")
    _logger.request_end()

    return {
        **state,
        "response": response,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ROUTING FUNCTIONS
# Determine which node to visit next based on current state.
# ══════════════════════════════════════════════════════════════════════════════

def route_after_sanitize(state: MedicalState) -> str:
    """After sanitization: block unsafe input, continue if safe."""
    if not state["is_safe"]:
        return "blocked"
    return "safety_check"


def route_after_safety(state: MedicalState) -> str:
    """After safety check: urgent goes straight to response, normal continues."""
    if state["is_urgent"]:
        return "urgent"
    return "extract"


# ══════════════════════════════════════════════════════════════════════════════
# SPECIAL TERMINAL NODES
# These set the final response and end the graph.
# ══════════════════════════════════════════════════════════════════════════════

def blocked_node(state: MedicalState) -> MedicalState:
    """Returns safe response for injection attempts."""
    _logger.node_end("sanitize", "BLOCKED — injection attempt")
    _logger.request_end()
    return {
        **state,
        "response": (
            "I'm a medical assistant and can only help with "
            "health-related questions. Please describe your symptoms."
        ),
    }


def urgent_node(state: MedicalState) -> MedicalState:
    """Returns immediate emergency response."""
    _logger.node_urgent("urgent")
    _logger.request_end()
    return {
        **state,
        "risk_level": "High",
        "response": (
            "⚠️ This sounds like a medical emergency. "
            "Please call emergency services (123) immediately "
            "or go to the nearest emergency room. "
            "Do not wait — seek help now."
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════

def build_medical_graph():
    """Builds and compiles the LangGraph medical pipeline."""
    graph = StateGraph(MedicalState)

    # Register all nodes
    graph.add_node("sanitize", sanitize_node)
    graph.add_node("safety_check", safety_node)
    graph.add_node("extract", nlp_extract_node)
    graph.add_node("normalize", normalize_node)
    graph.add_node("risk_assess", risk_assess_node)
    graph.add_node("respond", response_node)
    graph.add_node("blocked", blocked_node)
    graph.add_node("urgent", urgent_node)

    # Entry point
    graph.set_entry_point("sanitize")

    # Edges
    graph.add_conditional_edges(
        "sanitize",
        route_after_sanitize,
        {
            "blocked": "blocked",
            "safety_check": "safety_check",
        }
    )

    graph.add_conditional_edges(
        "safety_check",
        route_after_safety,
        {
            "urgent": "urgent",
            "extract": "extract",
        }
    )

    graph.add_edge("extract", "normalize")
    graph.add_edge("normalize", "risk_assess")
    graph.add_edge("risk_assess", "respond")

    # Terminal nodes
    graph.add_edge("blocked", END)
    graph.add_edge("urgent", END)
    graph.add_edge("respond", END)

    return graph.compile()


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC INTERFACE
# This is what PatientAgent will call.
# ══════════════════════════════════════════════════════════════════════════════

# Build once at module load
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_medical_graph()
    return _graph


def run_pipeline(user_input: str, session_context: dict = None) -> dict:
    """
    Main entry point for the LangGraph pipeline.

    Args:
        user_input: Raw text from the user
        session_context: Current session state (symptoms, context, etc.)

    Returns:
        Full state dict including 'response', 'risk_level', 'symptoms', etc.
    """
    graph = get_graph()

    initial_state: MedicalState = {
        "raw_input": user_input,
        "session_context": session_context or {},
        "clean_input": "",
        "is_safe": True,
        "unsafe_reason": "",
        "is_urgent": False,
        "intent": "others",
        "symptoms": [],
        "medical_context": "none",
        "severity": None,
        "duration": None,
        "normalized_symptoms": [],
        "risk_level": "Low",
        "response": "",
    }

    _logger.request_start()
    result = graph.invoke(initial_state)
    return result