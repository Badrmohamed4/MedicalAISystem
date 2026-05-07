def get_patient_prompt(user_input, symptoms, intent, domain, image_prediction=None):
    """
    Generates the prompt for the Patient Agent.
    Acts as a clinical triage assistant.
    """
    system_prompt = """You are a warm, reassuring clinical triage assistant. 
Your goal is to gather information and gradually narrow down a diagnosis without causing panic.
Rules:
1. Provide warm reassurance.
2. Ask 1-2 short follow-up questions to clarify symptoms.
3. Keep your response to a maximum of 4 sentences.
4. Avoid long explanations.
5. DO NOT mention medical imaging, scans, MRI, or X-rays unless the user explicitly mentioned them or uploaded an image.
"""
    
    prompt = f"Domain focus: {domain}\n"
    if image_prediction:
        prompt += f"An image analysis has already been performed. The prediction is: {image_prediction}. Use this as the primary symptom/indicator.\n"
    
    if symptoms:
        prompt += f"Extracted Symptoms: {', '.join(symptoms)}\n"
        
    prompt += f"User message: {user_input}\n\nRespond to the user naturally following your rules."
    
    return system_prompt, prompt
