"""
Follow-Up Routes — Flask Blueprint
All API endpoints for the follow-up module.
"""
import os
import uuid
from flask import Blueprint, render_template, request, jsonify, session
from medical_chatbot.followup import store

followup_bp = Blueprint("followup", __name__)

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "followup_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _get_patient_id():
    """Get or create a patient ID from the Flask session."""
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())
    return session["user_id"]


# ---- PAGE ---- #
@followup_bp.route("/followup")
def followup_page():
    return render_template("followup.html")


# ---- REMINDERS ---- #
@followup_bp.route("/api/followup/reminders", methods=["GET"])
def get_reminders():
    pid = _get_patient_id()
    return jsonify({"reminders": store.get_reminders(pid)})


@followup_bp.route("/api/followup/reminders", methods=["POST"])
def add_reminder():
    pid = _get_patient_id()
    data = request.json
    r = store.add_reminder(
        pid,
        reminder_type=data.get("type", "medication"),
        title=data.get("title", ""),
        date_time=data.get("date_time", ""),
        notes=data.get("notes", ""),
    )
    return jsonify({"reminder": r})


@followup_bp.route("/api/followup/reminders/<rid>/done", methods=["POST"])
def mark_done(rid):
    pid = _get_patient_id()
    ok = store.mark_reminder_done(pid, rid)
    return jsonify({"success": ok})


# ---- UPLOADS ---- #
@followup_bp.route("/api/followup/uploads", methods=["GET"])
def get_uploads():
    pid = _get_patient_id()
    return jsonify({"uploads": store.get_uploads(pid)})


@followup_bp.route("/api/followup/upload", methods=["POST"])
def upload_file():
    pid = _get_patient_id()

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    file_type = request.form.get("file_type", "scan")
    safe_name = f"{uuid.uuid4().hex[:8]}_{f.filename}"
    filepath = os.path.join(UPLOAD_DIR, safe_name)
    f.save(filepath)

    upload = store.add_upload(pid, safe_name, filepath, file_type, f.filename)
    return jsonify({"upload": upload})


# ---- EVALUATE ---- #
@followup_bp.route("/api/followup/evaluate/<upload_id>", methods=["POST"])
def evaluate(upload_id):
    pid = _get_patient_id()
    result = store.evaluate_upload(pid, upload_id)
    if result is None:
        return jsonify({"error": "Upload not found"}), 404
    return jsonify({"evaluation": result})


# ---- SEND TO DOCTOR ---- #
@followup_bp.route("/api/followup/send/<upload_id>", methods=["POST"])
def send_to_doctor(upload_id):
    pid = _get_patient_id()
    report = store.send_to_doctor(pid, upload_id)
    if report is None:
        return jsonify({"error": "Upload not found"}), 404
    return jsonify({"report": report})


# ---- REPORTS ---- #
@followup_bp.route("/api/followup/reports", methods=["GET"])
def get_reports():
    pid = _get_patient_id()
    return jsonify({"reports": store.get_reports(pid)})


@followup_bp.route("/api/followup/all_reports", methods=["GET"])
def get_all_reports():
    return jsonify({"reports": store.get_all_reports()})
