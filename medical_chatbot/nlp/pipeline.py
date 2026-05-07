import re
import sys
import os
import importlib.util

# Ensure the new medical_ai_project is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
chatbot_root = os.path.dirname(current_dir)
project_root = os.path.dirname(chatbot_root)
ai_project_path = os.path.join(project_root, "medical_ai_project")

# Import the new NLP Processor directly from file paths to avoid namespace conflicts
_nlp_processor_instance = None

def _load_module_from_file(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

try:
    processor_path = os.path.join(ai_project_path, "nlp", "processor.py")
    if os.path.exists(processor_path):
        processor_module = _load_module_from_file("ai_processor", processor_path)
        _nlp_processor_instance = processor_module.NLPProcessor()
        print("[NLP] ✅ BioBERT NLP Processor loaded successfully!")
    else:
        print(f"[NLP] ⚠️ Processor file not found at: {processor_path}")
except Exception as e:
    print(f"[NLP] ⚠️ Could not load AI Processor. Error: {e}")

class IntentClassifier:
    def predict(self, text):
        """
        Uses the new AI NLP Processor for intent classification without ANY rule-based fallbacks.
        """
        if not _nlp_processor_instance:
            return "others"

        nlp_result = _nlp_processor_instance.process(text)
        ai_intent = nlp_result.get("intent", "others")

        intent_mapping = {
            "symptom_report": "describe_symptoms",
            "treatment_query": "ask_what_to_do",
            "diagnosis_query": "ask_diagnosis",
            "discuss_image_result": "upload_image",
            "greet": "greet",
            "end_conversation": "end_conversation",
            "ask_urgency": "ask_urgency"
        }

        return intent_mapping.get(ai_intent, "describe_symptoms")

class EntityExtractor:
    def __init__(self):
        pass

    def extract(self, text):
        """
        Uses the AI NLP Processor to intelligently extract medical entities, severity, and duration.
        No rule-based regex or hardcoded keyword fallbacks are used.
        """
        found_symptoms = []
        severity = None
        duration = None
        medical_context = "none"

        if _nlp_processor_instance:
            try:
                nlp_result = _nlp_processor_instance.process(text)
                
                # Combine BOTH original and normalized symptoms
                # Original needed for follow-up question matching (keys like "dizziness")
                # Normalized needed for clinical display (terms like "vertigo")
                original = list(nlp_result.get("original_symptoms", []))
                normalized = list(nlp_result.get("normalized_symptoms", []))
                found_symptoms = list(set(original + normalized))
                    
                severity = nlp_result.get("severity")
                duration = nlp_result.get("duration")
                medical_context = nlp_result.get("medical_context", "none")
            except Exception as e:
                print(f"[NLP] AI extraction failed: {e}")

        return {
            "symptoms": found_symptoms,
            "severity": severity,
            "duration": duration,
            "medical_context": medical_context
        }
