class SapBERTNormalizer:
    """
    Medical term normalizer using SapBERT embeddings.
    Uses lazy loading to avoid consuming memory at startup.
    Falls back to dictionary if model is unavailable.
    """

    # Fallback dictionary for when model is not loaded
    FALLBACK_ONTOLOGY = {
        "chest pain": "angina",
        "heart attack": "myocardial infarction",
        "high blood pressure": "hypertension",
        "headache": "cephalgia",
        "throw up": "emesis",
        "throwing up": "emesis",
        "nausea": "nausea",
        "dizzy": "vertigo",
        "dizziness": "vertigo",
        "tired": "fatigue",
        "swelling": "edema",
        "tumor": "neoplasm",
        "shortness of breath": "dyspnea",
        "blurry vision": "visual disturbance",
        "blurred vision": "visual disturbance",
        "skin rash": "dermatitis",
        "itchy": "pruritus",
        "itching": "pruritus",
        "seizure": "epileptic seizure",
        "fainting": "syncope",
        "fever": "pyrexia",
        "cough": "tussis",
        "chest tightness": "chest tightness",
        "weight loss": "cachexia",
        "memory loss": "amnesia",
        "confusion": "disorientation",
        "numbness": "paresthesia",
        "tingling": "paresthesia",
        "vomiting": "emesis",
        "back pain": "dorsalgia",
        "joint pain": "arthralgia",
        "muscle pain": "myalgia",
        "difficulty breathing": "dyspnea",
        "loss of appetite": "anorexia",
        "excessive sweating": "hyperhidrosis",
        "palpitations": "cardiac palpitations",
    }

    def __init__(self):
        self._model = None
        self._model_loaded = False
        self._load_failed = False

        # Medical concept library for SapBERT to match against
        self.MEDICAL_CONCEPTS = list(set(self.FALLBACK_ONTOLOGY.values())) + [
            "hypertension", "tachycardia", "bradycardia", "arrhythmia",
            "pneumonia", "bronchitis", "asthma", "pleurisy",
            "glioma", "meningioma", "encephalopathy", "neuropathy",
            "melanoma", "eczema", "psoriasis", "acne vulgaris",
            "hepatitis", "cirrhosis", "gastritis", "colitis",
            "fracture", "sprain", "contusion", "laceration",
            "anemia", "leukemia", "thrombosis", "embolism",
            "sepsis", "infection", "inflammation", "necrosis",
        ]

    def _load_model(self):
        """Lazy load SapBERT only when first needed."""
        if self._model_loaded or self._load_failed:
            return

        try:
            print("[SapBERT] Loading model (first use)...")
            from sentence_transformers import SentenceTransformer
            import torch

            self._model = SentenceTransformer(
                "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
            )

            # Pre-encode the concept library once and cache it
            self._concept_embeddings = self._model.encode(
                self.MEDICAL_CONCEPTS,
                convert_to_tensor=True,
                show_progress_bar=False
            )

            self._model_loaded = True
            print("[SapBERT] ✅ Model loaded successfully.")

        except Exception as e:
            print(f"[SapBERT] ⚠️ Could not load model: {e}. Using fallback dictionary.")
            self._load_failed = True

    def _normalize_with_model(self, symptom):
        """Use SapBERT embeddings to find closest medical concept."""
        try:
            import torch
            from sentence_transformers import util

            symptom_embedding = self._model.encode(
                symptom,
                convert_to_tensor=True,
                show_progress_bar=False
            )

            scores = util.cos_sim(symptom_embedding, self._concept_embeddings)[0]
            best_idx = int(scores.argmax())
            best_score = float(scores[best_idx])
            best_concept = self.MEDICAL_CONCEPTS[best_idx]

            # Only use model result if confidence is high enough
            if best_score >= 0.7:
                return best_concept, round(best_score, 2)
            else:
                # Low confidence - return original term
                return symptom.lower(), round(best_score, 2)

        except Exception as e:
            print(f"[SapBERT] Normalization error: {e}")
            return symptom.lower(), 0.50

    def normalize(self, symptoms):
        """
        Normalize a list of symptoms to standard medical terminology.
        Returns a list of tuples: (normalized_term, confidence_score)
        
        Strategy:
        1. Try exact dictionary match first (fast)
        2. Try SapBERT model (accurate, lazy loaded)
        3. Fall back to original term
        """
        if not symptoms:
            return []

        # Try to load model on first actual use
        if not self._model_loaded and not self._load_failed:
            self._load_model()

        results = []
        for symptom in symptoms:
            symptom_lower = symptom.lower().strip()

            # Step 1: Fast dictionary lookup
            if symptom_lower in self.FALLBACK_ONTOLOGY:
                results.append((self.FALLBACK_ONTOLOGY[symptom_lower], 0.95))
                continue

            # Step 2: SapBERT model lookup
            if self._model_loaded:
                normalized, score = self._normalize_with_model(symptom_lower)
                results.append((normalized, score))
                continue

            # Step 3: Return original term as fallback
            results.append((symptom_lower, 0.50))

        return results