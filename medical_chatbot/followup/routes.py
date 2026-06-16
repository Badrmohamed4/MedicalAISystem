"""
Follow-Up Routes — Flask Blueprint
All API endpoints for the follow-up module.
"""
import os
import uuid
from flask import Blueprint, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
from medical_chatbot.followup import store

followup_bp = Blueprint("followup", __name__)

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "followup_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "pdf", "dcm"}


def _allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


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
    return jsonify({"reminders": store.get_reminders(pid), "patient_id": pid})


@followup_bp.route("/api/followup/all_patients", methods=["GET"])
def get_all_patients():
    """Overview of all patients for the multi-patient Follow-Up view."""
    return jsonify({"patients": store.get_all_patients_reminders()})


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


@followup_bp.route("/api/followup/patient/<pid>/reminders", methods=["GET"])
def get_patient_reminders(pid):
    """Get all reminders for a specific patient (used by the all-patients overview)."""
    return jsonify({"reminders": store.get_reminders(pid), "patient_id": pid})
def mark_done(rid):
    pid = _get_patient_id()
    ok = store.mark_reminder_done(pid, rid)
    return jsonify({"success": ok})


@followup_bp.route("/api/followup/reminders/<rid>/link", methods=["POST"])
def set_reminder_link(rid):
    """Let the patient attach a link (e.g. to lab results) to a reminder,
    such as a 'Lab Tests Required' reminder."""
    pid = _get_patient_id()
    data = request.json or {}
    link = (data.get("link") or "").strip()
    if not link:
        return jsonify({"error": "Link is required"}), 400

    r = store.set_reminder_link(pid, rid, link)
    if r is None:
        return jsonify({"error": "Reminder not found"}), 404
    return jsonify({"reminder": r})


@followup_bp.route("/api/followup/reminders/<rid>/scan", methods=["POST"])
def upload_scan_for_reminder(rid):
    """Upload a scan and attach it to a 'Follow-up Scan' reminder.
    The upload is also automatically sent to the doctor."""
    pid = _get_patient_id()

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not _allowed_file(f.filename):
        return jsonify({"error": "File type not allowed"}), 400

    clean_name = secure_filename(f.filename)
    safe_name = f"{uuid.uuid4().hex[:8]}_{clean_name}"
    filepath = os.path.join(UPLOAD_DIR, safe_name)
    f.save(filepath)

    upload = store.add_upload(pid, safe_name, filepath, "scan", clean_name)

    r = store.attach_scan_to_reminder(pid, rid, upload["id"])
    if r is None:
        return jsonify({"error": "Reminder not found"}), 404

    # Auto-evaluate and send to the doctor right away.
    store.evaluate_upload(pid, upload["id"])
    report = store.send_to_doctor(pid, upload["id"])

    return jsonify({"reminder": r, "upload": upload, "report": report})


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

    if not _allowed_file(f.filename):
        return jsonify({"error": "File type not allowed"}), 400

    file_type = request.form.get("file_type", "scan")
    clean_name = secure_filename(f.filename)
    safe_name = f"{uuid.uuid4().hex[:8]}_{clean_name}"
    filepath = os.path.join(UPLOAD_DIR, safe_name)
    f.save(filepath)

    upload = store.add_upload(pid, safe_name, filepath, file_type, clean_name)
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
    # Lazy import avoids a circular import with web_app at module load time.
    from medical_chatbot.web_app import session_store

    uid = session.get("user_id")
    current = session_store.get(uid)
    if not current or current.get("mode") != "doctor":
        return jsonify({"error": "Doctor access required"}), 403

    return jsonify({"reports": store.get_all_reports()})