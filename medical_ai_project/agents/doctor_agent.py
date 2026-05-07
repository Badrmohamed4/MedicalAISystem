def get_doctor_prompt(user_input, symptoms, intent, domain, image_prediction=None):
    """
    Generates the prompt for the Doctor Agent.
    Produces a structured clinical report.
    """
    system_prompt = """You are an expert AI clinical assistant for a doctor.
Your goal is to analyze the provided symptoms and clinical summary and produce a structured clinical report.
You MUST output your response in the EXACT following format:

Primary Differential Diagnosis:
[Provide the most likely diagnosis]

Key Clinical Indicators:
[List the main symptoms or image findings driving this diagnosis]

Recommended Workup:
[Suggest next steps, labs, or imaging if needed]

Clinical Note:
[A brief professional summary]

Use professional medical terminology. Do NOT deviate from this structure.
"""

    prompt = f"Medical Domain: {domain}\n"
    
    if image_prediction:
        prompt += f"Image Analysis Result: {image_prediction}\n"
        
    if symptoms:
        prompt += f"Identified Symptoms/Terms: {', '.join(symptoms)}\n"
        
    prompt += f"Clinical Input: {user_input}\n\nPlease generate the structured report."
    
    return system_prompt, prompt
