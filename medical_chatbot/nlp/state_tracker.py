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
            # Only upgrade severity, never downgrade
            SEVERITY_RANK = {"mild": 1, "moderate": 2, "severe": 3, "extreme": 3, "worst": 3}
            new_sev = new_vals.get("severity")
            cur_sev = current.get("severity")
            if new_sev and new_sev not in (None, "null", "none", "", "None"):
                new_rank = SEVERITY_RANK.get(str(new_sev).lower(), 0)
                cur_rank = SEVERITY_RANK.get(str(cur_sev or "").lower(), 0)
                if new_rank > cur_rank:
                    current["severity"] = new_sev
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
