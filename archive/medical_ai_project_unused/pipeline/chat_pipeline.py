from agents.patient_agent import get_patient_prompt
from agents.doctor_agent import get_doctor_prompt

class ChatPipeline:
    def __init__(self, nlp_processor, llm_client):
        self.nlp = nlp_processor
        self.llm = llm_client

    def run_chat(self, text, agent_type, domain, is_image_result=False, image_prediction=None):
        """
        Orchestrates the chat process.
        """
        # 1. Run NLP processing (unless it's just an image result wrapper)
        nlp_result = {"intent": "discuss_image_result", "normalized_symptoms": []}
        
        if not is_image_result and text.strip():
            nlp_result = self.nlp.process(text)
            
        if is_image_result:
            nlp_result["intent"] = "discuss_image_result"
            # Prediction becomes the primary "symptom/input"
            
        symptoms = nlp_result.get("normalized_symptoms", [])
        intent = nlp_result.get("intent", "others")

        # 2. Select agent & build prompt
        if agent_type == "patient":
            system_prompt, user_prompt = get_patient_prompt(
                text, symptoms, intent, domain, image_prediction
            )
        elif agent_type == "doctor":
            system_prompt, user_prompt = get_doctor_prompt(
                text, symptoms, intent, domain, image_prediction
            )
        else:
            raise ValueError("Unknown agent type")

        # 3. Stream LLM response
        for chunk in self.llm.stream_chat(system_prompt, user_prompt):
            yield chunk
