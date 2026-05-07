from datetime import datetime
from medical_chatbot.systems.medical_knowledge import CLINICAL_TERMS

class ReportGenerator:
    def __init__(self):
        pass

    def _map_to_clinical(self, text_list):
        clinical_list = []
        for term in text_list:
            mapped = CLINICAL_TERMS.get(term, term.capitalize())
            clinical_list.append(mapped)
        return clinical_list

    def generate_report(self, session):
        ctx = session.context
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # Clinical Mappings
        symptoms_clinical = self._map_to_clinical(ctx["extracted_entities"]["symptoms"])
        severity = ctx["extracted_entities"]["severity"]
        duration = ctx["extracted_entities"]["duration"]
        
        # Build report sections dynamically
        report = f"""
============================================================
              CLINICAL CONSULTATION REPORT
============================================================
Date: {now}
Patient ID: {ctx['patient_id']}
------------------------------------------------------------
"""
        
        # 1. Chief Complaint (only if symptoms exist)
        if ctx['extracted_entities']['symptoms']:
            report += f"""1. CHIEF COMPLAINT
   "Patient reports {', '.join(ctx['extracted_entities']['symptoms'])}"

"""
        
        # 2. History of Present Illness
        if symptoms_clinical or severity or duration:
            report += "2. HISTORY OF PRESENT ILLNESS\n"
            if symptoms_clinical:
                report += f"   Patient presents with {', '.join(symptoms_clinical)}.\n"
            if severity:
                report += f"   Severity: {severity.capitalize()}\n"
            if duration:
                report += f"   Duration: {duration}\n"
            report += "\n"
        
        # 3. Imaging Findings
        report += f"""3. IMAGING FINDINGS
   Modality: Medical Imaging (Brain/Lung Scan)
   Status: {'Image Uploaded' if ctx['image_uploaded'] else 'No Image Provided'}
"""
        if ctx['tumor_class']:
            report += f"   Findings: {ctx['tumor_class']}\n"
            report += f"   Confidence: {ctx['tumor_confidence']*100:.1f}%\n"
        report += "\n"
        
        # 4. Assessment
        report += f"""4. ASSESSMENT
   Primary Diagnosis: {ctx['tumor_class'] if ctx['tumor_class'] else 'Pending Analysis'}
   Risk Assessment: {ctx['risk_level'].upper()}

"""
        
        # 5. Plan/Recommendations
        report += f"""5. PLAN / RECOMMENDATIONS
   - Immediate Triage: {'Transfer to ER' if ctx['risk_level'] == 'High' else 'Outpatient Follow-up'}
   - Recommend further clinical evaluation and correlation with patient history.
============================================================
"""
        return report
