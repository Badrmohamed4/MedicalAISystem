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

# Import RAG retriever
_rag = None
try:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.join(project_root, "medical_ai_project"))
    from systems.rag_retriever import get_rag
    _rag = get_rag()
    print("[Agent] ✅ RAG retriever loaded.")
except Exception as e:
    print(f"[Agent] ⚠️ RAG retriever unavailable: {e}")

# LangGraph pipeline
_langgraph_pipeline = None
try:
    from medical_ai_project.pipeline.langgraph_pipeline import run_pipeline as _run_pipeline
    _langgraph_pipeline = _run_pipeline
    print("[Agent] ✅ LangGraph pipeline loaded.")
except Exception as e:
    print(f"[Agent] ⚠️ LangGraph pipeline unavailable: {e}")

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

     # ---------- 0. Urgency Check (always first, no LLM) ----------
        if text and self._is_urgent(text):
            print(f"\n{'─'*49}")
            print(f" PATIENT AGENT — URGENT INPUT DETECTED")
            print(f"{'─'*49}")
            print(f"[urgency     ] ⚠️  EMERGENCY PATH — bypassing NLP")
            urgent_response = (
                "⚠️ This sounds like a medical emergency. "
                "Please call emergency services (123) immediately "
                "or go to the nearest emergency room. "
                "Do not wait — seek help now."
            )
            self.session.update_context("risk_level", "High")
            self.session.add_message("system", urgent_response)
            return urgent_response


        # ---------- 1. NLP Extraction (BioBERT + Ollama) ----------
        import time
        _t0 = time.time()
        print(f"\n{'─'*49}")
        print(f" PATIENT AGENT — PROCESSING INPUT")
        print(f"{'─'*49}")
        print(f" Input: \"{text[:60]}\"")

        intent = self.intent_clf.predict(text)
        entities = self.ner.extract(text)

        _t1 = time.time()
        symptoms = entities.get("symptoms", [])
        context = entities.get("medical_context", "none")
        severity = entities.get("severity", "none")
        print(f"[nlp_extract ] ✅ {_t1-_t0:.3f}s | intent={intent} | context={context}")
        print(f"[nlp_extract ] ✅ symptoms={symptoms} | severity={severity}")

        self.session.update_context("current_intent", intent)
        print(f"[session] context updated")

        # Check if we are answering a question
        last_q = self.session.context.get("last_asked_question")

        # Only extract and save NEW symptoms if the patient is NOT just answering a question!
        # This prevents the AI from extracting "cough" if the patient says "when I cough"
        # in response to a question that had nothing to do with coughing.
        if not last_q:
            # IMPORTANT: If this is an image upload (empty text), do NOT overwrite
            # the already-accumulated entities and medical context.
            if text.strip():
                self.session.update_context("extracted_entities", entities)
                self.session.update_context("normalized_map",
                    entities.get("normalized_map", {}))

                # AI-driven medical context detection
                ai_context = entities.get("medical_context", "none")
                if ai_context in ["brain", "lung", "skin"]:
                    self.session.update_context("medical_context", ai_context)
            elif not image_path:
                # Only overwrite if it's a non-image, non-empty scenario
                self.session.update_context("extracted_entities", entities)
        else:
            # We ARE answering a question. Save the exact answer for the doctor's report.
            q_answers = self.session.context.get("question_answers", {})
            q_answers[last_q] = text
            self.session.update_context("question_answers", q_answers)

            # If this is the final "anything else?" prompt, the patient may describe
            # entirely NEW symptoms. Run NLP extraction and merge new symptoms in.
            is_final_answer = self.session.context.get("asked_final_prompt", False)
            if is_final_answer:
                new_symptoms = entities.get("symptoms", [])
                current_ents = self.session.context["extracted_entities"]
                for s in new_symptoms:
                    if s not in current_ents["symptoms"]:
                        current_ents["symptoms"].append(s)

                # Also catch common free-text symptom keywords the NLP might miss
                FREE_TEXT_SYMPTOMS = {
                    "cold": "chills / feeling cold",
                    "chill": "chills / feeling cold",
                    "shiver": "shivering",
                    "fever": "fever",
                    "tired": "fatigue",
                    "fatigue": "fatigue",
                    "weak": "weakness",
                    "sweat": "excessive sweating",
                    "numb": "numbness",
                    "tingling": "tingling",
                    "appetite": "loss of appetite",
                    "sleep": "sleep disturbance",
                    "insomnia": "insomnia",
                    "anxious": "anxiety",
                    "depressed": "depression",
                    "palpitation": "palpitations",
                    "heart racing": "palpitations",
                    "bloat": "bloating",
                    "constipat": "constipation",
                    "diarr": "diarrhea",
                }
                text_lower_final = text.lower()
                for keyword, symptom_label in FREE_TEXT_SYMPTOMS.items():
                    if keyword in text_lower_final:
                        if symptom_label not in current_ents["symptoms"]:
                            current_ents["symptoms"].append(symptom_label)
            
            # Parse severity and duration directly from the answer text
            current_ents = self.session.context["extracted_entities"]
            
            # Severity: check if the answer contains severity keywords
            text_lower = text.lower()
            if any(w in text_lower for w in ["sharp", "stabbing", "severe", "extreme", "worst", "intense"]):
                current_ents["severity"] = "severe"
            elif any(w in text_lower for w in ["throbbing", "moderate", "dull", "pressure", "heavy"]):
                current_ents["severity"] = "moderate"
            elif any(w in text_lower for w in ["mild", "slight", "minor", "light"]):
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

            # --- Enrich symptoms from follow-up answers ---
            # When the patient confirms something in a follow-up answer,
            # add the relevant clinical finding to the symptoms list.
            last_q_lower = last_q.lower()
            is_positive = any(w in text_lower for w in ["yes", "yeah", "yep", "correct", "true", "i do", "i have"])

            if is_positive:
                ANSWER_SYMPTOM_MAP = {
                    "weight loss": "unintentional weight loss",
                    "cough": "aggravated by cough",
                    "deep breath": "aggravated by deep breathing",
                    "blood": "coughing blood",
                    "shoulder": "pain radiating to shoulder",
                    "back": "pain radiating to back",
                    "respiratory infection": "recent respiratory infection",
                    "seizure": "seizures",
                    "nausea": "nausea",
                    "vomit": "vomiting",
                    "vision": "vision changes",
                    "smoke": "smoking history",
                    "fever": "fever",
                    "spreading": "spreading lesion",
                    "itchy": "pruritus",
                    "worse in the morning": "worse in morning",
                    "swelling": "swelling",
                    "wheezing": "wheezing",
                    "consciousness": "loss of consciousness",
                }
                for keyword, symptom_label in ANSWER_SYMPTOM_MAP.items():
                    if keyword in last_q_lower or keyword in text_lower:
                        if symptom_label not in current_ents["symptoms"]:
                            current_ents["symptoms"].append(symptom_label)

            # Also check if the answer itself contains a direct symptom mention
            # e.g. patient says "shoulder" when asked about radiating pain
            DIRECT_ANSWER_MAP = {
                "stabbing": "stabbing pain",
                "sharp": "sharp pain",
                "pressure": "heavy pressure",
                "shoulder": "pain radiating to shoulder",
                "back": "pain radiating to back",
                "both": None,  # not a symptom on its own
            }
            for keyword, symptom_label in DIRECT_ANSWER_MAP.items():
                if keyword in text_lower and symptom_label:
                    if symptom_label not in current_ents["symptoms"]:
                        current_ents["symptoms"].append(symptom_label)

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
            if current_context in ["brain", "lung", "skin"]:
                model_type = current_context
            else:
                fname = image_path.lower() if image_path else ""
                if any(k in fname for k in ["skin", "acne", "rash", "mole", "eczema", "melanoma", "derm"]):
                    model_type = "skin"
                elif any(k in fname for k in ["lung", "chest", "pulm", "cancer", "nodule"]):
                    model_type = "lung"
                elif any(k in fname for k in ["brain", "tumor", "mri", "glioma", "meningioma"]):
                    model_type = "brain"
                else:
                    model_type = "brain"
                print(f"[Image] No context set - inferred model_type={model_type} from filename")
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
        last_q = self.session.context.get("last_asked_question")
        self.session.update_context("last_asked_question", None)

        # If answering the continuation question, extract new symptoms from the answer
        CONTINUATION_QUESTIONS = [
            "Of course. What else would you like to tell me about your condition?",
            "Before I provide my assessment"
        ]
        is_continuation_answer = any(cq in (last_q or "") for cq in CONTINUATION_QUESTIONS)

        if is_continuation_answer and text.strip():
            cont_ents = self.session.context["extracted_entities"]
            text_lower_cont = text.lower()

            # Extract from NLP entities first
            for s in entities.get("symptoms", []):
                if s and s not in cont_ents["symptoms"]:
                    cont_ents["symptoms"].append(s)

            # Also check free text keywords for common symptoms
            CONT_KEYWORDS = {
                "short of breath": "shortness of breath",
                "shortness of breath": "shortness of breath",
                "cant breathe": "shortness of breath",
                "arm numb": "left arm numbness",
                "arm feels numb": "left arm numbness",
                "numb": "numbness",
                "fever": "fever", "feverish": "fever",
                "chills": "chills", "nausea": "nausea",
                "vomit": "vomiting", "dizzy": "dizziness",
                "tired": "fatigue", "weak": "weakness",
                "sweat": "sweating", "cough": "cough",
                "palpitat": "palpitations",
                "racing heart": "palpitations",
                "swollen": "swelling", "swell": "swelling",
                "blurry": "blurry vision", "vision": "vision changes",
                "headache": "headache", "head pain": "headache",
            }
            for kw, label in CONT_KEYWORDS.items():
                if kw in text_lower_cont and label not in cont_ents["symptoms"]:
                    cont_ents["symptoms"].append(label)
                    print(f"[Continuation] Added symptom: {label}")

        all_symptoms = self.session.context["extracted_entities"]["symptoms"]
        context = self.session.context.get("medical_context", "none")

        # Step 1: Try dynamic follow-up based on this specific answer
        dynamic_q = self._get_dynamic_followup(
            last_question=last_q or "",
            last_answer=text,
            symptoms=all_symptoms,
            context=context
        )

        if dynamic_q:
            answered = self.session.context.get("answered_questions", [])
            # Only use dynamic question if not already asked
            if dynamic_q not in answered:
                answered.append(dynamic_q)
                self.session.update_context("answered_questions", answered)
                self.session.update_context("last_asked_question", dynamic_q)
                response = f"Understood. {dynamic_q}"
                self.session.add_message("system", response)
                return response

        # Step 2: Fall back to static question bank
        next_q, key = self._find_next_question(all_symptoms)

        if next_q:
            answered = self.session.context.get("answered_questions", [])
            answered.append(next_q)
            self.session.update_context("answered_questions", answered)
            self.session.update_context("last_asked_question", next_q)
            response = f"Understood. {next_q}"
        else:
            # No more questions — ask final prompt if not yet asked
            if not self.session.context.get("asked_final_prompt"):
                self.session.update_context("asked_final_prompt", True)
                final_q = "Before I provide my assessment — is there anything else you would like to tell me about your condition?"
                self.session.update_context("last_asked_question", final_q)
                response = f"Thank you for your answers. {final_q}"
            else:
                # Final prompt already asked — check what patient said
                text_lower = text.lower()
                said_no = any(w in text_lower for w in ["no", "nope", "nothing", "thats all", "that's all", "im done", "i'm done", "no more", "that is all"])
                said_yes = any(w in text_lower for w in ["yes", "yeah", "yep", "actually", "also", "one more", "there is", "i also", "i have", "forgot"])
                if said_no:
                    response = self._generate_assessment()
                elif said_yes:
                    continuation_q = "Of course. What else would you like to tell me about your condition?"
                    self.session.update_context("last_asked_question", continuation_q)
                    self.session.update_context("asked_final_prompt", False)
                    response = continuation_q
                else:
                    response = self._generate_assessment()

        self.session.add_message("system", response)
        return response

    # ------------------------------------------------------------------ #
    #  INTENT HANDLER                                                      #
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    #  RAG RETRIEVAL HELPER                                                #
    # ------------------------------------------------------------------ #
    def _retrieve_rag(self, user_text="") -> str:
        """Retrieve RAG context based on current session symptoms and domain."""
        if not _rag:
            return ""
        try:
            symptoms = self.session.context["extracted_entities"]["symptoms"]
            context = self.session.context.get("medical_context", "unknown")
            query = f"{' '.join(symptoms)} {context} {user_text}".strip()
            retrieved = _rag.retrieve(query, domain_filter=context)
            if retrieved:
                print(f"[Agent] ✅ RAG retrieved context for domain={context}")
            return retrieved or ""
        except Exception as e:
            print(f"[Agent] ⚠️ RAG retrieval error: {e}")
            return ""

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

                    # RAG: retrieve and log relevant context (informs future Ollama calls)
                    self._retrieve_rag(text)

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
    #       URGENCY DETECTION (no LLM — deterministic)                   #
    # ------------------------------------------------------------------ #
    URGENCY_KEYWORDS = [
        "can't breathe", "cannot breathe", "cant breathe",
        "can't breathe", "chest pain", "heart attack",
        "stroke", "unconscious", "seizure", "bleeding heavily",
        "severe pain", "emergency", "dying", "collapsed",
        "overdose", "suicidal", "can not breathe",
    ]

    def _is_urgent(self, text):
        """Deterministic urgency check. No LLM. Always runs first."""
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.URGENCY_KEYWORDS)

    # ------------------------------------------------------------------ #
    #  DYNAMIC FOLLOW-UP GENERATION (Ollama-driven)                       #
    # ------------------------------------------------------------------ #
    # Max dynamic follow-ups allowed per symptom branch
    MAX_DYNAMIC_DEPTH = 2

    # Questions that ALWAYS need clarification if patient says yes/some
    ALWAYS_CLARIFY = {
        "medication": "Which medications are you currently taking?",
        "medicine": "Which medications are you currently taking?",
        "drug": "Which medications are you currently taking?",
        "family history": "Which specific condition runs in your family?",
        "cancer": "What type of cancer was diagnosed?",
        "surgery": "What type of surgery did you have?",
        "allergy": "What are you allergic to?",
    }

    # Phrases that indicate Ollama is leaking reasoning instead of asking a question
    REASONING_LEAK_PHRASES = [
        "based on the patient",
        "i would ask",
        "i would not ask",
        "this question aims",
        "the patient's answer",
        "it seems",
        "at this time",
        "in the future",
        "suggests a potential",
        "asking more questions",
        "exploring other aspects",
        "chain of thought",
        "my reasoning",
        "internal",
    ]

    def _get_dynamic_followup(self, last_question, last_answer, symptoms, context):
        """
        Uses Ollama to generate a context-aware follow-up question.
        Hard-limited to MAX_DYNAMIC_DEPTH per conversation.
        Sanitizes output to prevent reasoning leakage.
        """
        if not _ollama_client or not _ollama_client.is_online():
            return None

        # Priority check — some questions ALWAYS need clarification regardless of depth
        last_q_lower = (last_question or "").lower()
        answer_lower = (last_answer or "").lower()
        said_yes = any(w in answer_lower for w in ["yes", "yeah", "yep", "i do", "i am", "some", "a few"])

        if said_yes:
            for keyword, clarification_q in self.ALWAYS_CLARIFY.items():
                if keyword in last_q_lower:
                    answered = self.session.context.get("answered_questions", [])
                    if clarification_q not in answered:
                        dynamic_count = self.session.context.get("dynamic_followup_count", 0)
                        self.session.update_context("dynamic_followup_count", dynamic_count + 1)
                        return clarification_q

        # Depth check — stop if we have asked too many dynamic questions
        dynamic_count = self.session.context.get("dynamic_followup_count", 0)
        if dynamic_count >= self.MAX_DYNAMIC_DEPTH:
            return None

        # Check cache first
        cache = self.session.context.get("_dynamic_followup_cache", {})
        cache_key = f"{last_question}::{last_answer}"
        if cache_key in cache:
            return cache[cache_key]

        answered = self.session.context.get("answered_questions", [])
        answered_str = ", ".join(answered[-5:]) if answered else "none"
        symptoms_str = ", ".join(symptoms) if symptoms else "none"

        system_prompt = (
            "You are a medical triage assistant. "
            "Decide if ONE specific follow-up question is needed based on the patient's last answer. "
            "STRICT RULES:\n"
            "- Output ONLY the question text, nothing else\n"
            "- If no follow-up is needed, output exactly: NONE\n"
            "- Maximum one sentence\n"
            "- No explanations, no reasoning, no preamble\n"
            "- No phrases like 'Based on...' or 'I would ask...'\n"
            "- Only ask if the answer reveals a critical clinical detail needing immediate clarification\n"
            "- Examples: 'localized' needs WHERE; 'sharp' needs if it radiates; 'yes fever' needs duration"
        )

        user_prompt = (
            f"Context: {context} | Symptoms: {symptoms_str}\n"
            f"Already asked (recent): {answered_str}\n"
            f"Last question: {last_question}\n"
            f"Patient answered: {last_answer}\n"
            f"One follow-up question or NONE:"
        )

        parts = []
        for chunk in _ollama_client.stream_chat(system_prompt, user_prompt):
            parts.append(chunk)

        response = "".join(parts).strip()

        # Sanitize: reject if response contains reasoning leak phrases
        response_lower = response.lower()
        for phrase in self.REASONING_LEAK_PHRASES:
            if phrase in response_lower:
                cache[cache_key] = None
                self.session.update_context("_dynamic_followup_cache", cache)
                return None

        # Reject if too long (reasoning tends to be verbose)
        if len(response) > 120:
            cache[cache_key] = None
            self.session.update_context("_dynamic_followup_cache", cache)
            return None

        # Reject NONE responses
        if not response or response.upper().startswith("NONE"):
            cache[cache_key] = None
            self.session.update_context("_dynamic_followup_cache", cache)
            return None

        # Valid question — increment depth counter
        self.session.update_context("dynamic_followup_count", dynamic_count + 1)
        cache[cache_key] = response
        self.session.update_context("_dynamic_followup_cache", cache)
        return response

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
        medical_context = self.session.context.get("medical_context", "none")
        context_map = {
            "brain": "brain or neurological",
            "lung":  "lung or respiratory",
            "skin":  "skin or dermatological",
        }
        context_label = context_map.get(medical_context, "relevant medical")

        for q in FOLLOW_UP_QUESTIONS["general"]["general_checkup"]:
            # Personalize the family history question to match detected context
            personalized_q = q.replace(
                "any relevant medical conditions",
                f"{context_label} conditions"
            )
            if personalized_q not in answered and q not in answered:
                return personalized_q, "general"

        return None, None

    # ------------------------------------------------------------------ #
    #  PRELIMINARY ASSESSMENT                                              #
    # ------------------------------------------------------------------ #
    def _generate_assessment(self):
        """Provide a preliminary diagnosis based on all collected symptoms."""
        symptoms = self.session.context["extracted_entities"]["symptoms"]
        context = self.session.context.get("medical_context", "general")
        severity = self.session.context["extracted_entities"].get("severity")
        # Treat string "null" same as None — BioBERT returns "null" string
        if severity in (None, "null", "none", "", "None"):
            severity = None
        if not severity:
            all_sym = " ".join(self.session.context["extracted_entities"]["symptoms"]).lower()
            q_answers = " ".join(self.session.context.get("question_answers", {}).values()).lower()
            combined = all_sym + " " + q_answers
            if any(w in combined for w in ["sharp", "stabbing", "severe", "intense", "worst", "extreme"]):
                severity = "severe"
            elif any(w in combined for w in ["throbbing", "moderate", "dull", "pressure", "heavy", "strong"]):
                severity = "moderate"
            elif any(w in combined for w in ["mild", "slight", "minor", "light"]):
                severity = "mild"

        if not symptoms:
            return "I need more information. Could you describe what you're feeling?"

        risk, risk_msg = self.decision_engine.evaluate_text_risk(self.session)
        self.session.update_context("risk_level", risk)

        # ── RAG RETRIEVAL ──────────────────────────────────────────────────
         
        rag_note = ""
        rag_retrieved = self._retrieve_rag()
        if rag_retrieved:
            rag_note = f"\n\n📚 **Relevant Information**:\n{rag_retrieved}"
        else:
            rag_note = (
                "\n\n📚 **Relevant Information**:\n"
                "No specific reference documents matched this case. "
                "The assessment below is based on general medical knowledge only."
            )
        # 
        # ──────────────────────────────────────────────────────────────────

        symptoms_str = ", ".join(symptoms)

        response = "📋 **Preliminary Assessment**\n\n"
        response += f"**Symptoms recorded**: {symptoms_str}\n"
        response += f"**Medical area**: {context}\n"
        response += f"**Severity**: {severity or 'Not determined'}\n"
        response += f"**Risk level**: {risk}\n\n"
        response += risk_msg
        response += rag_note
        response += "\n\nFor a more accurate diagnosis, please upload a medical scan using the 📷 button."

        return response

    # ------------------------------------------------------------------ #
    #  OLLAMA RESPONSE GENERATION                                          #
    # ------------------------------------------------------------------ #
    def _ask_with_ollama(self, user_text):
        """
        Use LangGraph pipeline for general inputs.
        Falls back to direct Ollama if pipeline unavailable.
        Also updates session context with pipeline results.
        """
        # ── RAG RETRIEVAL (runs for all paths) ────────────────────────────
        rag_retrieved = self._retrieve_rag(user_text)
        if rag_retrieved:
            rag_section = f"\nRELEVANT MEDICAL REFERENCE:\n{rag_retrieved}\n"
        else:
            rag_section = (
                "\nRELEVANT MEDICAL REFERENCE:\n"
                "No specific reference documents were found for this query. "
                "Provide general medical guidance and recommend consulting a healthcare professional.\n"
            )
        # ──────────────────────────────────────────────────────────────────
        # ──────────────────────────────────────────────────────────────────

        # --- Try LangGraph pipeline first ---
        if _langgraph_pipeline:
            try:
                session_ctx = {
                    "last_response": self.session.history[-1]["content"] if self.session.history else "",
                    "medical_context": self.session.context.get("medical_context", "none"),
                    "symptoms": self.session.context["extracted_entities"].get("symptoms", []),
                    "rag_context": rag_retrieved,  # pass RAG context into pipeline
                }
                result = _langgraph_pipeline(user_text, session_context=session_ctx)

                # Update session context with pipeline findings
                if result.get("symptoms"):
                    existing = self.session.context["extracted_entities"]["symptoms"]
                    for s in result["symptoms"]:
                        if s not in existing:
                            existing.append(s)

                if result.get("medical_context") and result["medical_context"] != "none":
                    self.session.update_context("medical_context", result["medical_context"])

                if result.get("risk_level"):
                    self.session.update_context("risk_level", result["risk_level"])

                return result.get("response", "Could you describe your symptoms in more detail?")

            except Exception as e:
                print(f"[Agent] LangGraph pipeline error: {e}. Falling back to Ollama.")

        # --- Fallback: direct Ollama ---
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
            f"{rag_section}"
            "You help with brain tumors, lung cancer, and skin diseases.\n"
            "If the medical reference above is relevant, use it to inform your response — but never directly quote it verbatim to the patient.\n"
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
