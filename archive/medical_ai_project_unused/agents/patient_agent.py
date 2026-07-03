import sys
import os

# Ensure systems/ is importable
_agents_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_agents_dir)
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from systems.rag_retriever import get_rag


def get_patient_prompt(user_input, symptoms, intent, domain, image_prediction=None):
    """
    Generates the prompt for the Patient Agent.
    Acts as a clinical triage assistant.
    RAG-enhanced: retrieves domain-specific medical knowledge via SapBERT.
    """
    # ── RAG RETRIEVAL (SapBERT) ──────────────────────────────────────────────
    rag = get_rag()
    query = f"{' '.join(symptoms)} {domain} {user_input}".strip()
    retrieved_context = rag.retrieve(query, domain_filter=domain)
    # ─────────────────────────────────────────────────────────────────────────

    rag_section = (
        f"\nRELEVANT MEDICAL REFERENCE:\n{retrieved_context}\n"
        if retrieved_context
        else ""
    )

    system_prompt = (
        "You are a warm, reassuring clinical triage assistant. "
        "Your goal is to gather information and gradually narrow down a diagnosis without causing panic.\n"
        f"{rag_section}"
        "Rules:\n"
        "1. Provide warm reassurance.\n"
        "2. Ask 1-2 short follow-up questions to clarify symptoms.\n"
        "3. Keep your response to a maximum of 4 sentences.\n"
        "4. Avoid long explanations.\n"
        "5. DO NOT mention medical imaging, scans, MRI, or X-rays unless the user explicitly mentioned them or uploaded an image.\n"
        "6. If the medical reference above is relevant, use it to inform your response — but never directly quote it verbatim to the patient.\n"
    )

    prompt = f"Domain focus: {domain}\n"
    if image_prediction:
        prompt += f"An image analysis has already been performed. The prediction is: {image_prediction}. Use this as the primary symptom/indicator.\n"

    if symptoms:
        prompt += f"Extracted Symptoms: {', '.join(symptoms)}\n"

    prompt += f"User message: {user_input}\n\nRespond to the user naturally following your rules."

    return system_prompt, prompt
