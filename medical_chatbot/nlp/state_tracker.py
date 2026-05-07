import time

class Session:
    def __init__(self, session_id):
        self.session_id = session_id
        self.start_time = time.time()
        self.history = [] # List of {"role": "patient/doctor/system", "content": "..."}
        self.context = {
            "patient_id": "P_001", # Placeholder
            "current_intent": None,
            "extracted_entities": {
                "symptoms": [],
                "severity": None,
                "duration": None
            },
            "image_uploaded": False,
            "image_path": None,
            "tumor_class": None,
            "tumor_confidence": 0.0,
            "diagnosis_given": False,
            "risk_level": "Unknown" # Low, Medium, High
        }
    
    def add_message(self, role, content, mode="patient"):
        self.history.append({"role": role, "content": content, "mode": mode, "timestamp": time.time()})

    def update_context(self, key, value):
        if key == "extracted_entities":
            # Merge lists, overwrite others
            current = self.context["extracted_entities"]
            new_vals = value
            # Update symptoms set (unique)
            current["symptoms"] = list(set(current["symptoms"] + new_vals.get("symptoms", [])))
            # Overwrite duration/severity if new one found
            if new_vals.get("severity"):
                current["severity"] = new_vals["severity"]
            if new_vals.get("duration"):
                current["duration"] = new_vals["duration"]
        else:
            self.context[key] = value

    def get_summary(self):
        return {
            "symptoms": ", ".join(self.context["extracted_entities"]["symptoms"]),
            "diagnosis": self.context["tumor_class"] if self.context["tumor_class"] else "Pending",
            "risk": self.context["risk_level"]
        }
