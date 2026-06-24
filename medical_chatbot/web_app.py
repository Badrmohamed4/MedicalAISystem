import os
import sys
import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, send_from_directory, redirect, url_for
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from medical_chatbot.followup.routes import followup_bp
from medical_chatbot.database.db_manager import (
    init_db, save_session, save_message, save_symptoms,
    save_image, save_prediction, save_report,
    register_patient, get_patient_by_username,
    get_conversations, get_conversation_messages,
    rename_conversation, delete_conversation,
    link_session_to_patient, update_conversation_title
)

from medical_chatbot.followup import store as followup_store
from medical_chatbot.utils.input_sanitizer import sanitize_input, safe_medical_response

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from medical_chatbot.nlp.state_tracker import Session
from medical_chatbot.agents.patient_agent import PatientAgent
from medical_chatbot.agents.doctor_agent import DoctorAgent

app = Flask(__name__, template_folder="web/templates", static_folder="web/static")
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = os.path.join(current_dir, 'uploads')
app.register_blueprint(followup_bp)

_db_available = False
try:
    init_db()
    _db_available = True
except Exception as _db_err:
    print(f"[DB] Warning: PostgreSQL not available — running without persistence. ({_db_err.__class__.__name__})")


# Global storage for sessions (MVP only - use DB in prod)
session_store = {}


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('patient_id'):
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated

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
            "mode": "patient",
            "title": "New Conversation"
        }
        # Add initial greeting for new sessions
        new_session.add_message("system", "Hello. I am your Medical Assistant. Please describe your symptoms.")
        # Link session to logged-in patient in DB
        if session.get('patient_id') and _db_available:
            try:
                link_session_to_patient(uid, session['patient_id'])
            except Exception:
                pass
    return session_store[uid]

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if session.get('patient_id'):
        return redirect(url_for('dashboard'))
    if request.method == 'GET':
        return render_template('login.html')
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '').strip()
    patient = get_patient_by_username(username)
    if not patient or not check_password_hash(patient['password_hash'], password):
        return render_template('login.html', error='Incorrect username or password.', active_tab='login')
    session['patient_id'] = patient['patient_id']
    session['patient_name'] = patient['full_name']
    session['username'] = patient['username']
    return redirect(url_for('dashboard'))


@app.route('/register', methods=['POST'])
def register_page():
    full_name        = request.form.get('full_name', '').strip()
    username         = request.form.get('username', '').strip()
    national_id      = request.form.get('national_id', '').strip()
    password         = request.form.get('password', '').strip()
    confirm_password = request.form.get('confirm_password', '').strip()
    if not all([full_name, username, national_id, password, confirm_password]):
        return render_template('login.html', error='All fields are required.', active_tab='register')
    if password != confirm_password:
        return render_template('login.html', error='Passwords do not match.', active_tab='register')
    if len(national_id) != 14 or not national_id.isdigit():
        return render_template('login.html', error='National ID must be exactly 14 digits.', active_tab='register')
    hashed = generate_password_hash(password)
    ok = register_patient(full_name, username, national_id, hashed)
    if not ok:
        return render_template('login.html', error='Username or National ID already registered.', active_tab='register')
    return render_template('login.html', success='Account created! You can now sign in.', active_tab='login')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login_page'))


@app.route('/')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/chat')
@login_required
def chat_interface():
    return render_template('index.html')

@app.route('/doctor')
@login_required
def doctor_console():
    # Pass session store to template to render active reports
    return render_template('doctor_console.html', sessions=session_store)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message', '')
    state = get_agent_session()

    # --- Input Sanitization & Injection Guard ---
    sanitized = sanitize_input(user_input)
    if not sanitized["is_safe"]:
        return jsonify({
            "response": safe_medical_response(),
            "mode": state["mode"]
        })
    user_input = sanitized["clean_text"]
    # -------------------------------------------

    response_text = ""

    if state["mode"] == "patient":
        response_text = state["patient_bot"].process_input(user_input)
    else:
        response_text = state["doctor_bot"].process_query(user_input)


    # Persist to PostgreSQL
    if _db_available:
        try:
            session_obj = state["session"]
            uid = session["user_id"]
            save_session(uid,
                context_dict=session_obj.context,
                medical_context=session_obj.context.get("medical_context"),
                risk_level=session_obj.context.get("risk_level"),
                mode=state["mode"])
            save_message(uid, "patient", user_input)
            if _db_available and session.get('patient_id'):
                try:
                    title = user_input.strip()[:60]
                    update_conversation_title(uid, title)
                    session_store[uid]['title'] = title
                except Exception:
                    pass
            save_message(uid, "system", response_text)
            raw_symptoms = session_obj.context["extracted_entities"]["symptoms"]
            norm_map = session_obj.context.get("normalized_map", {})
            save_symptoms(uid,
                raw_symptoms,
                severity=session_obj.context["extracted_entities"].get("severity"),
                normalized_map=norm_map)
        except Exception as e:
            print(f"[DB] Warning: {e}")

    return jsonify({"response": response_text, "mode": state["mode"]})

@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/followup_uploads/<filename>')
def serve_followup_upload(filename):
    followup_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'followup_uploads')
    return send_from_directory(followup_dir, filename)

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
        

        # Persist image and prediction to PostgreSQL
        if _db_available:
            try:
                uid = session["user_id"]
                session_obj = state["session"]
                image_id = save_image(uid, file.filename, filepath,
                    image_type=session_obj.context.get("medical_context", "unknown"))
                if session_obj.context.get("tumor_class"):
                    save_prediction(uid,
                        model_used=session_obj.context.get("medical_context", "unknown"),
                        predicted_class=session_obj.context["tumor_class"],
                        confidence=session_obj.context.get("tumor_confidence", 0),
                        image_id=image_id)
            except Exception as e:
                print(f"[DB] Image persist warning: {e}")

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
    skin_scans = 0
    
    for uid, state in session_store.items():
        ctx = state["session"].context
        if ctx.get("image_uploaded"):
            medical_ctx = ctx.get("medical_context", "brain")
            if medical_ctx == "lung":
                lung_scans += 1
            elif medical_ctx == "skin":
                skin_scans += 1
            else:
                brain_scans += 1
    
    return jsonify({
        "active_users": active_users,
        "total_sessions": total_sessions,
        "brain_scans": brain_scans,
        "lung_scans": lung_scans,
        "skin_scans": skin_scans
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

    # Fetch followup reminders — keyed by session uid, same as followup store
    fu_reminders = followup_store.get_reminders(uid)
    print(f"[DEBUG REPORT] uid={uid[:8]}  reminders_found={len(fu_reminders)}")
    for r in fu_reminders:
        print(f"  → type={r['type']} title={r['title']} link={r.get('link')} scan={r.get('attached_upload')} status={r['status']}")

    # Resolve scan filenames from upload store
    def _get_upload_filename(upload_id):
        for u in followup_store.get_uploads(uid):
            if u["id"] == upload_id:
                return u["filename"]
        return None

    followup_results = {
        "lab_links":    [{"title": r["title"], "link": r["link"], "status": r["status"]}
                         for r in fu_reminders if r.get("type") == "lab" and r.get("link")],
        "scan_uploads": [{"title": r["title"], "date": r["date_time"],
                          "upload_id": r.get("attached_upload"),
                          "filename": _get_upload_filename(r.get("attached_upload")),
                          "status": r["status"]}
                         for r in fu_reminders if r.get("type") == "scan" and r.get("attached_upload")],
        "completed":    bool(fu_reminders) and all(r["status"] == "done" for r in fu_reminders)
    }

    # Persist report to PostgreSQL (safe — skips if already saved this session)
    try:
        from medical_chatbot.database.db_manager import save_report
        save_report(
            session_id=uid,
            structured_report=str({
                "symptoms": entities.get("symptoms", []),
                "severity": entities.get("severity"),
                "duration": entities.get("duration"),
                "medical_context": ctx.get("medical_context", "Not specified")
            }),
            ai_assessment=recommendation,
            risk_level=risk,
            symptoms_snapshot=entities.get("symptoms", [])
        )
        print("[DoctorRoute] Report saved to PostgreSQL.")
    except Exception as _e:
        print(f"[DoctorRoute] Could not save report: {_e}")

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
        "followup_plan": ctx.get("doctor_followup_plan", None),
        "followup_results": followup_results
    }

    return jsonify({"report": report})

@app.route('/api/doctor/followup/save', methods=['POST'])
def save_followup_plan():
    data = request.json
    uid = data.get('uid')
    if not uid:
        return jsonify({"error": "Missing session ID"}), 400

    state = session_store.get(uid)
    if not state:
        return jsonify({"error": "Session not found"}), 404

    medications = data.get("medications", [])
    next_scan = data.get("nextScan", "")
    visit_date = data.get("visitDate", "")
    lab_tests = data.get("labTests", "")
    notes = data.get("notes", "")

    # Save the follow-up plan to the session context
    state["session"].update_context("doctor_followup_plan", {
        "medications": medications,
        "next_scan": next_scan,
        "visit_date": visit_date,
        "lab_tests": lab_tests,
        "notes": notes,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M")
    })

    # --- Push the plan into the patient's Follow-Up reminders ---
    # The patient-side follow-up store is keyed by the same user_id (uid),
    # since both share Flask's session["user_id"].
    if visit_date:
        followup_store.add_reminder(
            uid, "appointment", "Doctor Follow-up Visit", visit_date, notes=notes
        )
    if next_scan:
        followup_store.add_reminder(
            uid, "scan", "Follow-up Scan", next_scan, notes=notes
        )
    if lab_tests:
        followup_store.add_reminder(
            uid, "lab", lab_tests, next_scan or visit_date, notes=notes
        )
    for med in medications:
        med_name = med.get("name", med) if isinstance(med, dict) else med
        med_time = med.get("date_time", "") if isinstance(med, dict) else ""
        followup_store.add_reminder(
            uid, "medication", med_name, med_time, notes=notes
        )

    return jsonify({"status": "success"})

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    # Disable reloader to prevent TensorFlow mutex lock errors on macOS
    app.run(debug=True, port=5001, use_reloader=False)

@app.route('/api/conversations', methods=['GET'])
@login_required
def list_conversations():
    patient_id = session.get('patient_id')
    try:
        convs = get_conversations(patient_id)
        for c in convs:
            c['created_at'] = c['created_at'].strftime('%Y-%m-%d %H:%M') if c.get('created_at') else ''
            c['updated_at'] = c['updated_at'].strftime('%Y-%m-%d %H:%M') if c.get('updated_at') else ''
        return jsonify({'conversations': convs})
    except Exception as e:
        return jsonify({'conversations': [], 'error': str(e)})


@app.route('/api/conversations/<conv_id>', methods=['GET'])
@login_required
def load_conversation(conv_id):
    patient_id = session.get('patient_id')
    messages = get_conversation_messages(conv_id, patient_id)
    if messages is None:
        return jsonify({'error': 'Conversation not found'}), 404
    if conv_id not in session_store:
        restored = Session(conv_id)
        for m in messages:
            restored.add_message(m['role'], m['content'], m.get('mode', 'patient'))
        session_store[conv_id] = {
            'session': restored,
            'patient_bot': PatientAgent(restored),
            'doctor_bot': DoctorAgent(restored),
            'mode': 'patient',
            'title': 'Restored Conversation'
        }
    session['user_id'] = conv_id
    formatted = [{'role': m['role'], 'content': m['content'], 'mode': m.get('mode', 'patient')} for m in messages]
    return jsonify({'history': formatted, 'session_id': conv_id})


@app.route('/api/conversations/<conv_id>/rename', methods=['POST'])
@login_required
def rename_conv(conv_id):
    patient_id = session.get('patient_id')
    data = request.json or {}
    new_title = data.get('title', '').strip()
    if not new_title:
        return jsonify({'error': 'Title cannot be empty'}), 400
    ok = rename_conversation(conv_id, patient_id, new_title)
    if conv_id in session_store:
        session_store[conv_id]['title'] = new_title
    return jsonify({'ok': ok})


@app.route('/api/conversations/<conv_id>/delete', methods=['POST'])
@login_required
def delete_conv(conv_id):
    patient_id = session.get('patient_id')
    ok = delete_conversation(conv_id, patient_id)
    if conv_id in session_store:
        del session_store[conv_id]
    if session.get('user_id') == conv_id:
        session.pop('user_id', None)
    return jsonify({'ok': ok})


@app.route('/api/conversations/new', methods=['POST'])
@login_required
def new_conversation():
    session.pop('user_id', None)
    state = get_agent_session()
    uid = session['user_id']
    return jsonify({'session_id': uid, 'message': 'New conversation started'})
