import sys
import os

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from medical_chatbot.nlp.state_tracker import Session
from medical_chatbot.agents.patient_agent import PatientAgent
from medical_chatbot.agents.doctor_agent import DoctorAgent

def main():
    print("Initializing Medical Chatbot System...")
    
    # Create a shared session for this run
    session_id = "SESSION_TEST_001"
    session = Session(session_id)
    
    patient_bot = PatientAgent(session)
    doctor_bot = DoctorAgent(session)
    
    mode = "menu"
    
    while True:
        if mode == "menu":
            print("\n" + "="*30)
            print(" MEDICAL ASSISTANT SYSTEM (MVP)")
            print("="*30)
            print("1. Patient Interface")
            print("2. Doctor Interface")
            print("3. Exit")
            choice = input("Select Mode (1-3): ")
            
            if choice == "1":
                mode = "patient"
                print("\n[Switching to Patient Agent...]")
                print("PATIENT AGENT: Hello. How can I help you today?")
            elif choice == "2":
                mode = "doctor"
                print("\n[Switching to Doctor Agent...]")
                print("DOCTOR AGENT: Enter command (report, history, image result, symptoms) or 'back'.")
            elif choice == "3":
                print("Exiting system.")
                break
        
        elif mode == "patient":
            user_input = input("YOU: ").strip()
            
            if user_input.lower() == "back":
                mode = "menu"
                continue
            
            # Check if input is a file path (simple check for demo)
            if (user_input.endswith(".jpg") or user_input.endswith(".png") or user_input.endswith(".jpeg")) and os.path.exists(user_input):
                print(f"PATIENT AGENT: [Processing Image: {user_input}]")
                response = patient_bot.process_input("", image_path=user_input)
            else:
                response = patient_bot.process_input(user_input)
            
            print(f"AGENT: {response}")

        elif mode == "doctor":
            user_input = input("DOCTOR: ").strip()
            
            if user_input.lower() == "back":
                mode = "menu"
                continue
            
            response = doctor_bot.process_query(user_input)
            print(f"SYSTEM:\n{response}")

if __name__ == "__main__":
    main()
