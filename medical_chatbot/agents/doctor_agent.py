import os
import sys

from medical_chatbot.systems.report_generator import ReportGenerator

# Load Ollama client and RAG retriever
_ollama_client = None
_rag = None
try:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.join(project_root, "medical_ai_project"))
    from llm.ollama_client import OllamaClient
    _ollama_client = OllamaClient()
    print("[DoctorAgent] ✅ Ollama client loaded for AI assessment.")
except Exception as e:
    print(f"[DoctorAgent] ⚠️ Ollama unavailable: {e}")

try:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.join(project_root, "medical_ai_project"))
    from systems.rag_retriever import get_rag
    _rag = get_rag()
    print("[DoctorAgent] ✅ RAG retriever loaded.")
except Exception as e:
    print(f"[DoctorAgent] ⚠️ RAG retriever unavailable: {e}")


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

        # Persist to PostgreSQL reports table
        try:
            from medical_chatbot.database.db_manager import save_report
            ctx = self.session.context
            entities = ctx.get("extracted_entities", {})
            save_report(
                session_id=self.session.session_id,
                structured_report=structured,
                ai_assessment=ai_assessment,
                risk_level=ctx.get("risk_level", "Unknown"),
                symptoms_snapshot=entities.get("symptoms", [])
            )
            print("[DoctorAgent] Report saved to PostgreSQL.")
        except Exception as _save_err:
            print(f"[DoctorAgent] Could not save report to DB: {_save_err}")

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

        # ── RAG RETRIEVAL ──────────────────────────────────────────────────
        rag_section = ""
        if _rag:
            try:
                rag_retrieved = _rag.retrieve_for_assessment(
                    symptoms=entities.get("symptoms", []),
                    medical_context=ctx.get("medical_context", "unknown"),
                    tumor_class=ctx.get("tumor_class"),
                )
                if rag_retrieved:
                    rag_section = f"\nRELEVANT MEDICAL KNOWLEDGE FROM SYSTEM DATABASE:\n{rag_retrieved}\n"
                else:
                    rag_section = (
                        "\nRELEVANT MEDICAL KNOWLEDGE FROM SYSTEM DATABASE:\n"
                        "No specific reference documents matched this case. "
                        "The assessment below is based on general medical knowledge only. "
                        "Recommend consulting specialist literature for rare or atypical presentations.\n"
                    )
            except Exception as _rag_err:
                print(f"[DoctorAgent] ⚠️ RAG retrieval error: {_rag_err}")
        # ──────────────────────────────────────────────────────────────────

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

        system_prompt = f"""You are an experienced medical AI assistant helping a doctor review a patient case.
{rag_section}Based on the patient data provided, generate a brief clinical assessment covering:
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

        print("\n[DoctorAgent] Generating RAG-enhanced AI clinical assessment...")
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
        print("[DoctorAgent] ✅ RAG-enhanced AI assessment generated.")
        return result