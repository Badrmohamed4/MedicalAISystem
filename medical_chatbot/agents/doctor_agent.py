from medical_chatbot.systems.report_generator import ReportGenerator

class DoctorAgent:
    def __init__(self, session):
        self.session = session
        self.reporter = ReportGenerator()
        
    def process_query(self, text):
        # Log user query
        self.session.add_message("doctor", text, mode="doctor")
        
        text = text.lower()
        response = ""
        
        if "report" in text or "summary" in text:
            response = self.reporter.generate_report(self.session)
            self.session.update_context("generated_report", response)
            
        elif "symptom" in text:
            symptoms = self.session.context["extracted_entities"]["symptoms"]
            response = f"Patient reported symptoms: {', '.join(symptoms)}"
            
        elif "image" in text or "scan" in text:
            cls = self.session.context["tumor_class"]
            if cls:
                response = f"MRI Analysis Result: {cls} (Confidence: {self.session.context['tumor_confidence']:.2f})"
            else:
                response = "No MRI scan has been processed yet."
            
        elif "history" in text:
            hist = ""
            for msg in self.session.history:
                hist += f"[{msg['role'].upper()}]: {msg['content']}\n"
            response = hist
            
        else:
            # Return the report by default when switching to doctor mode
            response = self.reporter.generate_report(self.session)
            self.session.update_context("generated_report", response)

        # Log system response
        self.session.add_message("system", response, mode="doctor")
        return response
