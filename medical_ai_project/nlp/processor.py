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

# Maps any BioBERT-returned context to our 3 supported domains
CONTEXT_NORMALIZATION_MAP = {
    # Brain variants
    "brain":          "brain",
    "neurological":   "brain",
    "neurology":      "brain",
    "head":           "brain",
    "neuro":          "brain",
    "cognitive":      "brain",
    "mental":         "brain",
    "cerebral":       "brain",
    "seizure":        "brain",
    "epilepsy":       "brain",
    "consciousness":  "brain",
    "unconscious":    "brain",
    "headache":       "brain",
    "migraine":       "brain",

    # Lung variants
    "lung":           "lung",
    "lungs":          "lung",
    "pulmonary":      "lung",
    "respiratory":    "lung",
    "chest":          "lung",
    "thoracic":       "lung",
    "cardiovascular": "lung",
    "cardiac":        "lung",
    "heart":          "lung",
    "breathing":      "lung",
    "breathe":        "lung",
    "breath":         "lung",
    "airway":         "lung",
    "oxygen":         "lung",

    # Skin variants
    "skin":           "skin",
    "dermatology":    "skin",
    "dermatological": "skin",
    "dermal":         "skin",
    "cutaneous":      "skin",
}

# Symptom keywords → medical context fallback
SYMPTOM_CONTEXT_KEYWORDS = {
    "brain": [
        "headache", "head", "dizzy", "dizziness", "seizure", "seizures",
        "vision", "blurry", "migraine", "memory", "confusion", "unconscious",
        "collapsed", "fainting", "concussion", "nausea", "vomiting",
        "consciousness", "neurological", "brain", "skull", "forehead",
    ],
    "lung": [
        "chest", "cough", "coughing", "breath", "breathing", "breathe",
        "shortness", "wheezing", "lung", "respiratory", "phlegm", "mucus",
        "inhale", "exhale", "oxygen", "asthma", "bronchitis", "pneumonia",
        "blood", "sputum", "heart attack", "cardiac", "palpitation",
    ],
    "skin": [
        "skin", "rash", "itch", "itchy", "red", "redness", "mole",
        "acne", "pimple", "eczema", "psoriasis", "lesion", "patch",
        "blister", "hives", "flaky", "dry skin", "melanoma", "bump",
    ],
}

def infer_context_from_symptoms(symptoms: list) -> str:
    """Infers medical context from symptom keywords when BioBERT returns none."""
    scores = {"brain": 0, "lung": 0, "skin": 0}
    for symptom in symptoms:
        s_lower = symptom.lower()
        for domain, keywords in SYMPTOM_CONTEXT_KEYWORDS.items():
            for kw in keywords:
                if kw in s_lower or s_lower in kw:
                    scores[domain] += 1
                    break
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "none"

def normalize_context(context: str) -> str:
    """Maps any context string to brain/lung/skin/none."""
    if not context:
        return "none"
    return CONTEXT_NORMALIZATION_MAP.get(context.lower().strip(), "none")

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

        raw_context = extraction_result.get("medical_context", "none")
        mapped_context = normalize_context(raw_context)

        # If BioBERT returned none, infer from symptoms
        if mapped_context == "none" and symptoms:
            mapped_context = infer_context_from_symptoms(symptoms)

        # If still none, try inferring from normalized symptoms too
        if mapped_context == "none" and normalized_symptoms:
            norm_terms = [term for term, score in normalized_symptoms]
            mapped_context = infer_context_from_symptoms(norm_terms)

        return {
            "intent": intent or "others",
            "original_symptoms": symptoms,
            "normalized_symptoms": [term for term, score in normalized_symptoms],
            "scores": [score for term, score in normalized_symptoms],
            "medical_context": mapped_context,
            "severity": extraction_result.get("severity", None),
            "duration": extraction_result.get("duration", None)
        }
