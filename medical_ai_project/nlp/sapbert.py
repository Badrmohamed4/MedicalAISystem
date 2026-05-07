class SapBERTNormalizer:
    def __init__(self):
        """
        Initializes the normalizer. 
        In a full implementation, this would load a sentence-transformers model.
        For this lightweight setup, we simulate normalization using a dictionary 
        to avoid heavy memory footprint if not strictly required.
        """
        # A simple medical ontology mapping for demonstration.
        self.ontology = {
            "chest pain": "angina",
            "heart attack": "myocardial infarction",
            "high blood pressure": "hypertension",
            "headache": "cephalgia",
            "throw up": "emesis",
            "throwing up": "emesis",
            "nausea": "nausea",
            "dizzy": "vertigo",
            "tired": "fatigue",
            "swelling": "edema",
            "tumor": "neoplasm"
        }

    def normalize(self, symptoms):
        """
        Normalizes a list of symptoms to standard medical terminology.
        Returns a list of tuples: (normalized_term, score)
        """
        normalized_results = []
        for symptom in symptoms:
            symptom_lower = symptom.lower()
            if symptom_lower in self.ontology:
                normalized_results.append((self.ontology[symptom_lower], 0.95))
            else:
                # If no direct mapping, return the symptom itself with lower confidence
                normalized_results.append((symptom_lower, 0.50))
        return normalized_results
