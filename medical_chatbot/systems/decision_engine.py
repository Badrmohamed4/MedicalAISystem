from medical_chatbot.systems.medical_knowledge import ADVICE, TUMOR_INFO, LUNG_CANCER_INFO, SKIN_INFO, INTERIM_CARE

class DecisionEngine:
    def __init__(self):
        pass

    def assess_risk(self, session):
        """
        Determines risk level based on:
        1. Tumor Classification (Malignant classes -> High)
        2. Symptoms (Seizures/Severe pain -> High)
        """
        tumor = session.context["tumor_class"]
        is_brain_tumor = tumor in ["Glioma", "Meningioma", "Pituitary"]
        is_lung_cancer = tumor in ["Adenocarcinoma", "Large Cell Carcinoma", "Squamous Cell Carcinoma"]
        is_skin_disease = tumor in ["Acne", "Eczema", "Malignant", "Psoriasis"]
        
        is_tumor = is_brain_tumor or is_lung_cancer or (is_skin_disease and tumor == "Malignant")
        symptoms = session.context["extracted_entities"]["symptoms"]
        severity = session.context["extracted_entities"]["severity"]
        
        risk = "Low"
        
        # High Risk Conditions
        if is_tumor:
            risk = "High"
        elif "seizures" in symptoms or "unconscious" in symptoms:
            risk = "High"
        elif "coughing blood" in symptoms or ("shortness of breath" in symptoms and severity in ["severe", "worst"]):
            risk = "High"
        elif "unintentional weight loss" in symptoms:
            risk = "High"
        elif "aggravated by cough" in symptoms or "worse in morning" in symptoms:
            risk = "High"
        elif severity in ["severe", "worst", "extreme"]:
            risk = "High"
        # Medium Risk
        elif "vomiting" in symptoms or "dizziness" in symptoms or "chest pain" in symptoms:
             if risk != "High": 
                risk = "Medium"
        
        return risk

    def get_advice(self, session):
        """
        Returns a list of advice strings, including specific interim care.
        """
        risk = self.assess_risk(session)
        symptoms = session.context["extracted_entities"]["symptoms"]
        
        advice_list = []
        
        # 1. Emergency / High Risk Care
        if risk == "High":
            advice_list.extend(ADVICE["Emergency"])
            advice_list.extend(INTERIM_CARE["High_Risk"])
        
        # 2. Symptom specific Care
        for s in symptoms:
            if s in ["headache"] or "pain" in s:
                 advice_list.extend(ADVICE["Headache"])
                 advice_list.extend(INTERIM_CARE.get("Headache_Care", []))
            elif s in ["seizures", "seizure"]:
                 advice_list.extend(ADVICE["Seizure"])
                 advice_list.extend(INTERIM_CARE.get("Seizure_Care", []))
        
        # 3. General if list is short
        if len(advice_list) < 2:
            advice_list.extend(ADVICE["General"])
            
        return list(set(advice_list)) # Dedup

    def evaluate_text_risk(self, session):
        """
        Evaluates risk based ONLY on symptoms (before MRI).
        Returns: (risk_level, recommendation_message)
        """
        symptoms = session.context["extracted_entities"]["symptoms"]
        severity = session.context["extracted_entities"]["severity"]
        
        risk = "Low"
        reasons = []

        # Check answers to follow-up questions
        q_answers = session.context.get("question_answers", {})
        for q, a in q_answers.items():
            a_lower = a.lower()
            if "weight loss" in q.lower() and ("yes" in a_lower or "kg" in a_lower or "lbs" in a_lower):
                risk = "High"
                reasons.append("Unexplained weight loss")
            if "coughing up blood" in q.lower() and "yes" in a_lower:
                risk = "High"
                reasons.append("Coughing up blood")

        # Risk Rules
        if "seizures" in symptoms or "seizure" in symptoms:
            risk = "High"
            reasons.append("Seizures detected")
        if "unconscious" in symptoms:
            risk = "High"
            reasons.append("Loss of consciousness")
        if severity in ["severe", "worst", "extreme"] and ("headache" in symptoms or "pain" in symptoms):
            risk = "High"
            reasons.append("Severe pain indications")
        if "coughing blood" in symptoms or "blood" in symptoms:
            risk = "High"
            reasons.append("Coughing up blood")
            risk = "High"
            reasons.append("Severe difficulty breathing")
        if "unintentional weight loss" in symptoms:
            risk = "High"
            reasons.append("Unexplained weight loss")
        if "aggravated by cough" in symptoms or "worse in morning" in symptoms:
            risk = "High"
            reasons.append("Signs of increased intracranial pressure")
        
        # Combinations (e.g. Headache + Vomiting + Vision Issues)
        # Checking for >= 2 suspicious symptoms
        suspicious_count = 0
        if "headache" in symptoms or "head" in symptoms: suspicious_count += 1
        if "vomiting" in symptoms or "nausea" in symptoms: suspicious_count += 1
        if "vision" in symptoms or "vision loss" in symptoms: suspicious_count += 1
        if "fever" in symptoms: suspicious_count += 1
        if "cough" in symptoms: suspicious_count += 1
        if "chest pain" in symptoms: suspicious_count += 1
        if "shortness of breath" in symptoms: suspicious_count += 1
        
        if suspicious_count >= 3:
            risk = "High"
            reasons.append("Multiple concerning symptoms")
        elif suspicious_count == 2 and risk != "High":
            risk = "Medium"
            reasons.append("Combination of symptoms")

        # Formulate Recommendation
        if risk == "High":
             msg = "⚠️ **Health Analysis**: Based on your symptoms (" + ", ".join(reasons) + "), this looks suspicious.\n"
             msg += "**Recommendation**: It is highly recommended to get an MRI scan to rule out any underlying causes. Please seek professional medical help."
        elif risk == "Medium":
             msg = "**Health Analysis**: Your symptoms explain some concern (" + ", ".join(reasons) + ").\n"
             msg += "**Recommendation**: An MRI scan would be beneficial to be sure. In the meantime, monitor your condition closely."
        else:
             msg = "**Health Analysis**: Your symptoms appear mild at this stage.\n"
             msg += "**Recommendation**: You can try some home care (rest/hydration). If symptoms persist or worsen, please consult a doctor or upload an MRI scan."

        return risk, msg

    def get_diagnosis_message(self, session):
        if not session.context["tumor_class"]:
            return "No diagnosis available yet. Please upload an MRI scan."
        
        tumor = session.context["tumor_class"]
        conf = session.context["tumor_confidence"]
        
        msg = f"Based on the analysis of your scan, the system has detected signs of: **{tumor}** "
        msg += f"(Confidence: {conf*100:.1f}%).\n\n"
        
        info = TUMOR_INFO.get(tumor, LUNG_CANCER_INFO.get(tumor, SKIN_INFO.get(tumor, "")))
        msg += info
        
        if self.assess_risk(session) == "High":
            msg += "\n\n⚠️ **URGENT**: This requires immediate medical attention."
            
        return msg
