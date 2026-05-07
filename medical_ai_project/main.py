import sys
import os

from llm.ollama_client import OllamaClient
from nlp.processor import NLPProcessor
from pipeline.chat_pipeline import ChatPipeline
from pipeline.image_pipeline import ImagePipeline

def main():
    print("Initializing System...")
    nlp = NLPProcessor()
    llm = OllamaClient()
    
    if not llm.is_online():
        print("WARNING: Ollama server is offline! Please run 'ollama serve'.")
        
    chat_pipeline = ChatPipeline(nlp, llm)
    image_pipeline = ImagePipeline()
    
    print("\n=== Unified Medical AI Platform ===")
    print("Select Mode:")
    print("1. Chat Mode")
    print("2. Image Upload Mode")
    mode = input("Choice (1/2): ").strip()
    
    domain = input("Enter domain (brain/lung/skin/general): ").strip()
    agent_type = input("Enter agent type (patient/doctor): ").strip()
    
    if mode == "1":
        while True:
            text = input("\nYou: ")
            if text.lower() in ['exit', 'quit']:
                break
                
            print("AI: ", end="", flush=True)
            for chunk in chat_pipeline.run_chat(text, agent_type, domain):
                print(chunk, end="", flush=True)
            print()
            
    elif mode == "2":
        img_path = input("\nEnter path to image file: ").strip()
        if not os.path.exists(img_path):
            print("File not found.")
            return
            
        print("Analyzing image...")
        prediction_text = image_pipeline.predict(img_path, domain)
        print(f"\nResult: {prediction_text}\n")
        
        print("AI Explanation: ", end="", flush=True)
        for chunk in chat_pipeline.run_chat(
            text="", 
            agent_type=agent_type, 
            domain=domain, 
            is_image_result=True, 
            image_prediction=prediction_text
        ):
            print(chunk, end="", flush=True)
        print()
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
