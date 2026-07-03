import sys
import os

# Ensure systems/ is importable
_agents_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_agents_dir)
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from systems.rag_retriever import get_rag


def get_doctor_prompt(user_input, symptoms, intent, domain, image_prediction=None):
    """
    Generates the prompt for the Doctor Agent.
    Produces a structured clinical report.
    RAG-enhanced: retrieves domain-specific medical knowledge via SapBERT.
    """
    # ── RAG RETRIEVAL (SapBERT) ──────────────────────────────────────────────
    rag = get_rag()
    retrieved_context = rag.retrieve_for_assessment(
        symptoms=symptoms,
        medical_context=domain,
        tumor_class=image_prediction,
    )
    print(f"[DoctorAgent] Generating RAG-enhanced AI clinical assessment...")
    # ─────────────────────────────────────────────────────────────────────────

    rag_section = (
        f"\nRELEVANT MEDICAL KNOWLEDGE FROM SYSTEM DATABASE:\n{retrieved_context}\n"
        if retrieved_context
        else ""
    )

    system_prompt = (
        "You are an expert AI clinical assistant for a doctor.\n"
        "Your goal is to analyze the provided symptoms and clinical summary and produce a structured clinical report.\n"
        f"{rag_section}"
        "Use the medical knowledge above (if provided) to ground your assessment.\n\n"
        "You MUST output your response in the EXACT following format:\n\n"
        "Primary Differential Diagnosis:\n"
        "[Provide the most likely diagnosis — use 'suggestive of' or 'consistent with']\n\n"
        "Key Clinical Indicators:\n"
        "[List the main symptoms or image findings driving this diagnosis]\n\n"
        "Recommended Workup:\n"
        "[Suggest next steps, labs, or imaging if needed]\n\n"
        "Clinical Note:\n"
        "[A brief professional summary]\n\n"
        "IMPORTANT RULES:\n"
        "- Never make a definitive diagnosis\n"
        "- Always recommend specialist consultation for serious findings\n"
        "- Use professional medical terminology\n"
        "- Do NOT deviate from the structure above\n"
    )

    prompt = f"Medical Domain: {domain}\n"

    if image_prediction:
        prompt += f"Image Analysis Result: {image_prediction}\n"

    if symptoms:
        prompt += f"Identified Symptoms/Terms: {', '.join(symptoms)}\n"

    prompt += f"Clinical Input: {user_input}\n\nPlease generate the structured report."

    print(f"[DoctorAgent] ✅ RAG-enhanced AI assessment generated.")
    return system_prompt, prompt
