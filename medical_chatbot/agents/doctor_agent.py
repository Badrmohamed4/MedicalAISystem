import os
import sys

from medical_chatbot.systems.report_generator import ReportGenerator

# Load Ollama client
_ollama_client = None
try:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.join(project_root, "medical_ai_project"))
    from llm.ollama_client import OllamaClient
    _ollama_client = OllamaClient()
    print("[DoctorAgent] ✅ Ollama client loaded for AI assessment.")
except Exception as e:
    print(f"[DoctorAgent] ⚠️ Ollama unavailable: {e}")


class DoctorAgent:
    def __init__(self, session):
        self.session = session
        self.reporter = ReportGenerator()

    def process_query(self, text):
        self.session.add_message("doctor", text, mode="doctor")
        text_lower = text.lower()
        response = ""

        if "report" in text_lower or "summary" in text_lower:
            response = self._full_report()

        elif "symptom" in text_lower:
            symptoms = self.session.context["extracted_entities"]["symptoms"]
            response = f"Patient reported symptoms: {', '.join(symptoms) if symptoms else 'None recorded'}"

        elif "image" in text_lower or "scan" in text_lower:
            cls = self.session.context.get("tumor_class")
            if cls:
                conf = self.session.context.get("tumor_confidence", 0)
                response = f"Imaging Result: {cls} (Confidence: {conf*100:.1f}%)"
            else:
                response = "No imaging scan has been processed yet."

        elif "assess" in text_lower or "analysis" in text_lower or "ai" in text_lower:
            response = self._generate_ai_assessment()

        elif "history" in text_lower:
            hist = ""
            for msg in self.session.history:
                hist += f"[{msg['role'].upper()}]: {msg['content']}\n"
            response = hist if hist else "No history recorded."

        else:
            # Default: full report with AI assessment
            response = self._full_report()

        self.session.add_message("system", response, mode="doctor")
        return response

    # ------------------------------------------------------------------ #
    #  FULL REPORT = Structured Report + AI Assessment                    #
    # ------------------------------------------------------------------ #
    def _full_report(self):
        """Generates the structured report then appends AI clinical assessment."""
        structured = self.reporter.generate_report(self.session)
        ai_assessment = self._generate_ai_assessment()
        full = structured + "\n" + ai_assessment
        self.session.update_context("generated_report", full)
        return full

    # ------------------------------------------------------------------ #
    #  AI CLINICAL ASSESSMENT (Ollama)                                    #
    # ------------------------------------------------------------------ #
    def _generate_ai_assessment(self):
        """
        Uses Ollama to generate a real clinical assessment based on
        the patient's collected data. This is what makes DoctorAgent
        an actual AI agent, not just a report formatter.
        """
        if not _ollama_client or not _ollama_client.is_online():
            return "[ AI Assessment unavailable — Ollama offline ]"

        ctx = self.session.context
        entities = ctx.get("extracted_entities", {})
        symptoms = entities.get("symptoms", [])
        severity = entities.get("severity", "unknown")
        duration = entities.get("duration", "unknown")
        medical_context = ctx.get("medical_context", "unknown")
        risk_level = ctx.get("risk_level", "unknown")
        tumor_class = ctx.get("tumor_class")
        tumor_confidence = ctx.get("tumor_confidence", 0)
        question_answers = ctx.get("question_answers", {})

        # Build a structured patient summary for the LLM
        patient_summary = f"""
Patient Data:
- Medical Area: {medical_context}
- Reported Symptoms: {', '.join(symptoms) if symptoms else 'None'}
- Severity: {severity}
- Duration: {duration}
- Risk Level: {risk_level}
- Imaging Result: {f'{tumor_class} ({tumor_confidence*100:.1f}% confidence)' if tumor_class else 'No imaging performed'}
- Follow-up Q&A: {str(question_answers) if question_answers else 'None'}
"""

        system_prompt = """You are an experienced medical AI assistant helping a doctor review a patient case.
Based on the patient data provided, generate a brief clinical assessment covering:
1. Most likely condition or differential diagnosis (2-3 possibilities)
2. Key clinical concerns based on symptoms and risk level
3. Recommended next steps (tests, referrals, monitoring)
4. Urgency level and reasoning

IMPORTANT RULES:
- Be concise and clinical in tone
- Never make a definitive diagnosis — use "suggestive of" or "consistent with"
- Always recommend specialist consultation for serious findings
- Keep the assessment under 200 words
- Format with clear numbered sections
"""

        print("\n[DoctorAgent] Generating AI clinical assessment...")
        parts = []
        for chunk in _ollama_client.stream_chat(system_prompt, patient_summary):
            parts.append(chunk)

        assessment = "".join(parts) if parts else "AI assessment could not be generated."

        result = f"""
============================================================
              AI CLINICAL ASSESSMENT
         (Generated by MediBot — Not a substitute for medical advice)
============================================================
{assessment}
============================================================
"""
        print("[DoctorAgent] ✅ AI assessment generated.")
        return result