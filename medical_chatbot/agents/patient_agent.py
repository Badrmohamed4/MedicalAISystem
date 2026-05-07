import os
import sys
from medical_chatbot.nlp.pipeline import IntentClassifier, EntityExtractor
from medical_chatbot.systems.decision_engine import DecisionEngine
from medical_chatbot.utils.image_processor import ModelWrapper

# Import OllamaClient for LLM-driven response generation
_ollama_client = None
try:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.join(project_root, "medical_ai_project"))
    from llm.ollama_client import OllamaClient
    _ollama_client = OllamaClient()
    print("[Agent] ✅ Ollama client loaded for response generation.")
except Exception as e:
    print(f"[Agent] ⚠️ Ollama unavailable for responses: {e}")


class PatientAgent:
    def __init__(self, session):
        self.session = session
        self.intent_clf = IntentClassifier()
        self.ner = EntityExtractor()
        self.decision_engine = DecisionEngine()
        self.model = ModelWrapper()

    # ------------------------------------------------------------------ #
    #  MAIN ENTRY POINT                                                    #
    # ------------------------------------------------------------------ #
    def process_input(self, text, image_path=None):
        self.session.add_message("patient", text)

        # ---------- 1. NLP Extraction (BioBERT + Ollama) ----------
        intent = self.intent_clf.predict(text)
        entities = self.ner.extract(text)

        self.session.update_context("current_intent", intent)

        # Check if we are answering a question
        last_q = self.session.context.get("last_asked_question")

        # Only extract and save NEW symptoms if the patient is NOT just answering a question!
        # This prevents the AI from extracting "cough" if the patient says "when I cough"
        # in response to a question that had nothing to do with coughing.
        if not last_q:
            self.session.update_context("extracted_entities", entities)

            # AI-driven medical context detection
            ai_context = entities.get("medical_context", "none")
            if ai_context in ["brain", "lung", "skin"]:
                self.session.update_context("medical_context", ai_context)
        else:
            # We ARE answering a question. Save the exact answer for the doctor's report.
            q_answers = self.session.context.get("question_answers", {})
            q_answers[last_q] = text
            self.session.update_context("question_answers", q_answers)
            
            # Parse severity and duration directly from the answer text
            current_ents = self.session.context["extracted_entities"]
            
            # Severity: check if the answer contains severity keywords
            text_lower = text.lower()
            if any(w in text_lower for w in ["sharp", "severe", "extreme", "worst", "intense"]):
                current_ents["severity"] = "severe"
            elif any(w in text_lower for w in ["throbbing", "moderate", "dull"]):
                current_ents["severity"] = "moderate"
            elif any(w in text_lower for w in ["mild", "slight", "minor"]):
                current_ents["severity"] = "mild"
            elif entities.get("severity"):
                current_ents["severity"] = entities["severity"]
            
            # Duration: check if the answer contains time expressions
            import re
            duration_match = re.search(r'(\d+)\s*(day|week|month|year|hour)s?', text_lower)
            if duration_match:
                current_ents["duration"] = duration_match.group(0)
            elif entities.get("duration"):
                current_ents["duration"] = entities["duration"]

        current_context = self.session.context.get("medical_context")

        # If the AI didn't detect a context, infer it from the symptoms
        if not current_context or current_context == "none":
            symptoms = self.session.context["extracted_entities"]["symptoms"]
            inferred = self._infer_context_from_symptoms(symptoms)
            if inferred:
                current_context = inferred
                self.session.update_context("medical_context", inferred)

        # ---------- 2. Image handling ----------
        if image_path:
            self.session.update_context("image_uploaded", True)
            self.session.update_context("image_path", image_path)
            # Use the detected medical context for model selection
            model_type = current_context if current_context in ["brain", "lung", "skin"] else "brain"
            return self._process_image_inference(image_path, model_type)

        # ---------- 3. Answer handling ----------
        if last_q:
            return self._handle_answer(text, entities)

        # ---------- 4. Intent-based response ----------
        return self._handle_intent(intent, entities, text)

    # ------------------------------------------------------------------ #
    #  ANSWER HANDLER                                                      #
    # ------------------------------------------------------------------ #
    def _handle_answer(self, text, entities):
        """Process the user's answer to a previously asked question."""
        self.session.update_context("last_asked_question", None)

        # Find and ask the next relevant question
        all_symptoms = self.session.context["extracted_entities"]["symptoms"]
        next_q, key = self._find_next_question(all_symptoms)

        if next_q:
            answered = self.session.context.get("answered_questions", [])
            answered.append(next_q)
            self.session.update_context("answered_questions", answered)
            self.session.update_context("last_asked_question", next_q)
            response = f"Understood. {next_q}"
        else:
            # All questions exhausted → give preliminary assessment
            response = self._generate_assessment()

        self.session.add_message("system", response)
        return response

    # ------------------------------------------------------------------ #
    #  INTENT HANDLER                                                      #
    # ------------------------------------------------------------------ #
    def _handle_intent(self, intent, entities, text):
        response = ""

        if intent == "greet":
            response = (
                "Hello! I'm your Medical AI Assistant. "
                "I can help analyze brain, lung, and skin conditions. "
                "Please describe your symptoms or upload a medical scan."
            )

        elif intent == "describe_symptoms":
            symptoms = self.session.context["extracted_entities"]["symptoms"]

            if symptoms:
                # Find a follow-up question for the extracted symptoms
                next_q, key = self._find_next_question(symptoms)

                if next_q:
                    answered = self.session.context.get("answered_questions", [])
                    answered.append(next_q)
                    self.session.update_context("answered_questions", answered)
                    self.session.update_context("last_asked_question", next_q)

                    # Build a natural acknowledgement
                    response = f"I understand. To better assess your condition: {next_q}"
                else:
                    # No more questions → assessment
                    response = self._generate_assessment()
            else:
                response = self._ask_with_ollama(text)

        elif intent == "upload_image":
            response = "Please use the 📷 upload button to share your medical scan."

        elif intent == "ask_diagnosis":
            response = self.decision_engine.get_diagnosis_message(self.session)

        elif intent in ["ask_what_to_do", "ask_urgency"]:
            advice = self.decision_engine.get_advice(self.session)
            response = "Based on your input, here is some guidance:\n- " + "\n- ".join(advice)

        elif intent == "end_conversation":
            response = "Thank you for using MediBot. Take care and consult a specialist if symptoms persist."

        else:
            response = self._ask_with_ollama(text)

        self.session.add_message("system", response)
        return response

    # ------------------------------------------------------------------ #
    #  SYMPTOM-TO-CONTEXT INFERENCE                                         #
    # ------------------------------------------------------------------ #
    SYMPTOM_CONTEXT_MAP = {
        "brain": ["headache", "head", "dizzy", "dizziness", "seizure", "seizures",
                  "vision", "vision loss", "confusion", "memory", "nausea",
                  "migraine", "concussion", "fainting", "unconscious"],
        "lung":  ["chest", "chest pain", "cough", "coughing", "breathing",
                  "shortness of breath", "wheezing", "lung", "respiratory",
                  "phlegm", "mucus", "pneumonia", "asthma", "bronchitis"],
        "skin":  ["skin", "rash", "itch", "itchy", "red", "redness", "mole",
                  "acne", "pimple", "eczema", "psoriasis", "bumps", "lesion",
                  "melanoma", "blister", "hives", "swelling"],
    }

    def _infer_context_from_symptoms(self, symptoms):
        """Infer medical context from symptom keywords when the AI fails to detect it."""
        scores = {"brain": 0, "lung": 0, "skin": 0}
        for symptom in symptoms:
            s_lower = symptom.lower()
            for domain, keywords in self.SYMPTOM_CONTEXT_MAP.items():
                for kw in keywords:
                    if kw in s_lower or s_lower in kw:
                        scores[domain] += 1
                        break
        best = max(scores, key=scores.get)
        if scores[best] > 0:
            return best
        return None

    # ------------------------------------------------------------------ #
    #  FOLLOW-UP QUESTION FINDER                                           #
    # ------------------------------------------------------------------ #
    def _find_next_question(self, symptoms):
        """
        Searches FOLLOW_UP_QUESTIONS for the first un-asked question that
        matches any of the patient's accumulated symptoms, strictly within their medical context.
        """
        from medical_chatbot.systems.medical_knowledge import FOLLOW_UP_QUESTIONS

        answered = self.session.context.get("answered_questions", [])
        current_context = self.session.context.get("medical_context", "none")

        # 1. Try to find a symptom-specific question in the active medical context
        if current_context in FOLLOW_UP_QUESTIONS:
            domain_questions = FOLLOW_UP_QUESTIONS[current_context]
            for symptom in symptoms:
                s_lower = symptom.lower()
                for key, questions in domain_questions.items():
                    # Match: key is substring of symptom OR symptom is substring of key
                    if key in s_lower or s_lower in key:
                        for q in questions:
                            if q not in answered:
                                return q, key

        # 2. If no context-specific question is found, fall back to general checkup questions
        for q in FOLLOW_UP_QUESTIONS["general"]["general_checkup"]:
            if q not in answered:
                return q, "general"

        return None, None

    # ------------------------------------------------------------------ #
    #  PRELIMINARY ASSESSMENT                                              #
    # ------------------------------------------------------------------ #
    def _generate_assessment(self):
        """Provide a preliminary diagnosis based on all collected symptoms."""
        symptoms = self.session.context["extracted_entities"]["symptoms"]
        context = self.session.context.get("medical_context", "general")
        severity = self.session.context["extracted_entities"].get("severity")

        if not symptoms:
            return "I need more information. Could you describe what you're feeling?"

        risk, risk_msg = self.decision_engine.evaluate_text_risk(self.session)
        self.session.update_context("risk_level", risk)

        symptoms_str = ", ".join(symptoms)

        response = "📋 **Preliminary Assessment**\n\n"
        response += f"**Symptoms recorded**: {symptoms_str}\n"
        response += f"**Medical area**: {context}\n"
        if severity:
            response += f"**Severity**: {severity}\n"
        response += f"**Risk level**: {risk}\n\n"
        response += risk_msg
        response += "\n\nFor a more accurate diagnosis, please upload a medical scan using the 📷 button."

        return response

    # ------------------------------------------------------------------ #
    #  OLLAMA RESPONSE GENERATION                                          #
    # ------------------------------------------------------------------ #
    def _ask_with_ollama(self, user_text):
        """Use Ollama LLM to generate a contextual medical response."""
        if not _ollama_client or not _ollama_client.is_online():
            return (
                "Could you describe your symptoms in more detail? "
                "For example: 'I have a headache', 'my skin is red and itchy', or 'I have chest pain'."
            )

        symptoms = self.session.context["extracted_entities"]["symptoms"]
        context = self.session.context.get("medical_context", "unknown")

        system_prompt = (
            "You are a medical AI chatbot. The patient's known info:\n"
            f"- Medical area: {context}\n"
            f"- Known symptoms: {', '.join(symptoms) if symptoms else 'none yet'}\n"
            "You help with brain tumors, lung cancer, and skin diseases.\n"
            "Respond in 2-3 sentences max. Ask about their symptoms. Be empathetic."
        )

        parts = []
        for chunk in _ollama_client.stream_chat(system_prompt, user_text):
            parts.append(chunk)

        return "".join(parts) if parts else "Could you describe your symptoms in more detail?"

    # ------------------------------------------------------------------ #
    #  IMAGE INFERENCE                                                     #
    # ------------------------------------------------------------------ #
    def _process_image_inference(self, image_path, context):
        label, conf = self.model.predict(image_path, model_type=context)
        self.session.update_context("tumor_class", label)
        self.session.update_context("tumor_confidence", conf)
        diag_msg = self.decision_engine.get_diagnosis_message(self.session)
        self.session.add_message("system", diag_msg)
        return diag_msg
