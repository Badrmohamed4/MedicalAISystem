import functools
import os
import importlib.util

# Load biobert and sapbert using absolute file paths to avoid relative import issues
_nlp_dir = os.path.dirname(os.path.abspath(__file__))

def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_biobert_module = _load_module("biobert", os.path.join(_nlp_dir, "biobert.py"))
_sapbert_module = _load_module("sapbert", os.path.join(_nlp_dir, "sapbert.py"))

BioBERTExtractor = _biobert_module.BioBERTExtractor
SapBERTNormalizer = _sapbert_module.SapBERTNormalizer

class NLPProcessor:
    def __init__(self):
        self.extractor = BioBERTExtractor()
        self.normalizer = SapBERTNormalizer()

    @functools.lru_cache(maxsize=128)
    def process(self, text):
        """
        Full NLP pipeline with caching for speed.
        Text -> BioBERT Extraction -> Intent Detection -> Conditional SapBERT Normalization
        """
        # Step 1: Extract intent and symptoms
        extraction_result = self.extractor.extract(text)
        intent = extraction_result["intent"]
        symptoms = extraction_result["symptoms"]

        # Optimization Rules
        if intent == "others" and not symptoms:
            # Skip SapBERT Normalization only if we found absolutely nothing
            normalized_symptoms = []
        else:
            # Step 2: Normalize symptoms using SapBERT logic
            normalized_symptoms = self.normalizer.normalize(symptoms)

        return {
            "intent": intent,
            "original_symptoms": symptoms,
            "normalized_symptoms": [term for term, score in normalized_symptoms],
            "scores": [score for term, score in normalized_symptoms],
            "medical_context": extraction_result.get("medical_context", "none"),
            "severity": extraction_result.get("severity", None),
            "duration": extraction_result.get("duration", None)
        }
