from transformers import AutoTokenizer
import os
import sys

# Import OllamaClient
current_dir = os.path.dirname(os.path.abspath(__file__))
llm_dir = os.path.join(os.path.dirname(current_dir), "llm")
sys.path.append(os.path.dirname(current_dir))
from llm.ollama_client import OllamaClient

class BioBERTExtractor:
    def __init__(self, model_name="dmis-lab/biobert-v1.1"):
        """
        Loads the BioBERT tokenizer to process the text, and then uses Ollama 
        to intelligently extract symptoms and intent without hardcoded rules.
        """
        print(f"Loading tokenizer: {model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f"Warning: Could not load BioBERT tokenizer ({e}). Proceeding without it.")
            self.tokenizer = None
            
        self.ollama = OllamaClient()

    def extract(self, text):
        """
        Uses Ollama to extract intent, symptoms, medical context, severity, and duration.
        """
        # We can still tokenize the text if we want to limit length or do preprocessing,
        # but the core extraction logic is now handed over to the LLM.
        
        system_prompt = """
        You are a medical AI assistant component. Analyze the patient's text and extract the following information strictly as JSON:
        {
            "intent": "string (MUST BE ONE OF: greet, symptom_report, treatment_query, diagnosis_query, discuss_image_result, ask_urgency, end_conversation, others)",
            "symptoms": ["list of strings (extract any medical symptoms, body parts in pain, or conditions. EXCLUDE any symptoms the patient explicitly denies having. E.g. if they say 'I have no cough', do NOT extract 'cough')"],
            "medical_context": "string (MUST BE ONE OF: brain, lung, skin, or none if unspecified)",
            "severity": "string (e.g. mild, moderate, severe, extreme, or null)",
            "duration": "string (e.g. 3 days, 1 week, or null)"
        }
        Do not add any markdown formatting, only output the raw JSON object.
        """
        
        result = self.ollama.generate_json(system_prompt, text)
        
        if "error" in result:
            print(f"[BioBERTExtractor] Error from Ollama: {result['error']}")
            # Fallback to empty if LLM fails
            return {
                "intent": "others",
                "symptoms": [],
                "medical_context": "none",
                "severity": None,
                "duration": None
            }
            
        # Ensure we return the expected structure even if the LLM hallucinates slightly
        return {
            "intent": result.get("intent", "others"),
            "symptoms": result.get("symptoms", []),
            "medical_context": result.get("medical_context", "none"),
            "severity": result.get("severity", None),
            "duration": result.get("duration", None)
        }
