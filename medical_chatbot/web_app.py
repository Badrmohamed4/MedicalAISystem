import os
import sys
import uuid
from flask import Flask, render_template, request, jsonify, session, send_from_directory

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from medical_chatbot.nlp.state_tracker import Session
from medical_chatbot.agents.patient_agent import PatientAgent
from medical_chatbot.agents.doctor_agent import DoctorAgent
from medical_chatbot.followup.routes import followup_bp

app = Flask(__name__, template_folder="web/templates", static_folder="web/static")
app.secret_key = "secret_key_medical_chatbot_mvp"
app.config['UPLOAD_FOLDER'] = os.path.join(current_dir, 'uploads')
app.register_blueprint(followup_bp)

# Global storage for sessions (MVP only - use DB in prod)
session_store = {}

def get_agent_session():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    
    uid = session['user_id']
    if uid not in session_store:
        # Initialize new session
        new_session = Session(uid)
        session_store[uid] = {
            "session": new_session,
            "patient_bot": PatientAgent(new_session),
            "doctor_bot": DoctorAgent(new_session),
            "mode": "patient" # Default mode
        }
        # Add initial greeting for new sessions
        new_session.add_message("system", "Hello. I am your Medical Assistant. Please describe your symptoms or upload an MRI scan for analysis.")
    return session_store[uid]

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/chat')
def chat_interface():
    return render_template('index.html')

@app.route('/doctor')
def doctor_console():
    # Pass session store to template to render active reports
    return render_template('doctor_console.html', sessions=session_store)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message', '')
    state = get_agent_session()
    
    response_text = ""
    
    if state["mode"] == "patient":
        response_text = state["patient_bot"].process_input(user_input)
    else:
        response_text = state["doctor_bot"].process_query(user_input)
        
    return jsonify({"response": response_text, "mode": state["mode"]})

@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/upload', methods=['POST'])
def upload():
    import time
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Simulate processing time (2 seconds)
        time.sleep(2)
        
        state = get_agent_session()
        print(f"DEBUG: Processing upload for mode: {state.get('mode')}")
        
        # Only patient agent deals with uploads logic mainly
        response_text = state["patient_bot"].process_input("", image_path=filepath)
        print(f"DEBUG: Model Response: {response_text}")
        
        return jsonify({"response": response_text, "mode": state["mode"]})

@app.route('/api/switch_mode', methods=['POST'])
def switch_mode():
    data = request.json
    target_mode = data.get('mode', 'patient')
    state = get_agent_session()
    state["mode"] = target_mode
    
    msg = ""
    if target_mode == "patient":
        msg = "Switched to Patient Interface. How can I help you regarding your symptoms?"
        state["session"].add_message("system", msg, mode="patient")
    else:
        # Auto-generate report
        report = state["doctor_bot"].process_query("report")
        msg = f"Switched to Doctor Interface.\n{report}"
        state["session"].add_message("system", msg, mode="doctor")
        
    return jsonify({"response": msg, "mode": target_mode})

@app.route('/api/set_context', methods=['POST'])
def set_context():
    data = request.json
    context = data.get('context', 'brain')
    state = get_agent_session()
    
    # Update the session context
    state["session"].update_context("medical_context", context)
    
    return jsonify({"status": "success", "context": context})

@app.route('/api/clear', methods=['POST'])
def clear_session():
    if 'user_id' in session:
        uid = session['user_id']
        if uid in session_store:
            del session_store[uid]
    return jsonify({"response": "Session cleared."})

@app.route('/api/history', methods=['GET'])
def get_history():
    state = get_agent_session()
    history = state["session"].history
    
    # Format history for frontend
    formatted_history = []
    for msg in history:
        formatted_history.append({
            "role": msg["role"],
            "content": msg["content"],
            "mode": msg.get("mode", "patient") # Default to patient if missing
        })
    
    return jsonify({"history": formatted_history, "mode": state["mode"]})

@app.route('/api/stats', methods=['GET'])
def get_stats():
    # Calculate stats from session_store
    active_users = len(session_store)
    total_sessions = len(session_store)
    
    brain_scans = 0
    lung_scans = 0
    
    for uid, state in session_store.items():
        ctx = state["session"].context
        if ctx.get("image_uploaded"):
            medical_ctx = ctx.get("medical_context", "brain")
            if medical_ctx == "lung":
                lung_scans += 1
            else:
                brain_scans += 1
    
    return jsonify({
        "active_users": active_users,
        "total_sessions": total_sessions,
        "brain_scans": brain_scans,
        "lung_scans": lung_scans
    })

@app.route('/api/doctor/report/<uid>', methods=['GET'])
def get_doctor_report(uid):
    """Generate a structured clinical report for a specific patient session."""
    from datetime import datetime
    from medical_chatbot.systems.medical_knowledge import CLINICAL_TERMS

    if uid not in session_store:
        return jsonify({"error": "Session not found"}), 404

    state = session_store[uid]
    ctx = state["session"].context
    entities = ctx["extracted_entities"]

    # Map symptoms to clinical terms
    clinical_symptoms = []
    for s in entities.get("symptoms", []):
        mapped = CLINICAL_TERMS.get(s, s.capitalize())
        clinical_symptoms.append(mapped)

    # Build conversation excerpt (last 20 messages)
    conversation = []
    for msg in state["session"].history[-20:]:
        conversation.append({
            "role": msg["role"],
            "content": msg["content"][:300]
        })

    # Risk + recommendation — compute dynamically
    from medical_chatbot.systems.decision_engine import DecisionEngine
    engine = DecisionEngine()
    
    # Use image-based risk if a tumor was detected, otherwise text-based
    if ctx.get("tumor_class"):
        risk = engine.assess_risk(state["session"])
    else:
        risk, _ = engine.evaluate_text_risk(state["session"])
    
    # Override with saved risk_level if it was already set higher
    saved_risk = ctx.get("risk_level", "Unknown")
    risk_priority = {"Unknown": 0, "Low": 1, "Medium": 2, "High": 3}
    if risk_priority.get(saved_risk, 0) > risk_priority.get(risk, 0):
        risk = saved_risk
    
    if risk == "High":
        recommendation = "Immediate medical attention required. Transfer to ER."
    elif risk == "Medium":
        recommendation = "Schedule follow-up appointment. Monitor symptoms closely."
    else:
        recommendation = "Continue home care. Seek help if symptoms worsen."

    report = {
        "patient_id": ctx.get("patient_id", "Unknown"),
        "session_id": uid,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "medical_context": ctx.get("medical_context", "Not specified"),
        "symptoms": entities.get("symptoms", []),
        "clinical_symptoms": clinical_symptoms,
        "severity": entities.get("severity"),
        "duration": entities.get("duration"),
        "image_uploaded": ctx.get("image_uploaded", False),
        "image_path": os.path.basename(ctx.get("image_path", "")) if ctx.get("image_path") else None,
        "diagnosis": ctx.get("tumor_class"),
        "confidence": round(ctx.get("tumor_confidence", 0) * 100, 1),
        "risk_level": risk,
        "recommendation": recommendation,
        "conversation": conversation,
        "question_answers": ctx.get("question_answers", {}),
        "followup_plan": ctx.get("doctor_followup_plan", None)
    }

    return jsonify({"report": report})

@app.route('/api/doctor/followup/save', methods=['POST'])
def save_followup_plan():
    data = request.json
    uid = data.get('uid')
    if not uid:
        return jsonify({"error": "Missing session ID"}), 400
        
    state = store.get_session(uid)
    if not state:
        return jsonify({"error": "Session not found"}), 404
        
    # Save the follow-up plan to the session context
    state["session"].update_context("doctor_followup_plan", {
        "medications": data.get("medications", []),
        "next_scan": data.get("nextScan", ""),
        "visit_date": data.get("visitDate", ""),
        "lab_tests": data.get("labTests", ""),
        "notes": data.get("notes", ""),
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M")
    })
    
    return jsonify({"status": "success"})

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    # Disable reloader to prevent TensorFlow mutex lock errors on macOS
    app.run(debug=True, port=5001, use_reloader=False)
