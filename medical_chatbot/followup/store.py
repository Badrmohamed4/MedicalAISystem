"""
Follow-Up Data Store
Simple in-memory store for reminders, uploads, and reports.
Linked by Patient ID.
"""
import time
import uuid


# In-memory stores (use DB in production)
_reminders = {}   # patient_id -> [reminder, ...]
_uploads = {}     # patient_id -> [upload, ...]
_reports = {}     # patient_id -> [report, ...]


# ------------------------------------------------------------------ #
#  REMINDERS                                                          #
# ------------------------------------------------------------------ #
def add_reminder(patient_id, reminder_type, title, date_time, notes=""):
    """Add a reminder for a patient. Types: medication, scan, lab"""
    if patient_id not in _reminders:
        _reminders[patient_id] = []

    reminder = {
        "id": str(uuid.uuid4())[:8],
        "type": reminder_type,
        "title": title,
        "date_time": date_time,
        "notes": notes,
        "status": "pending",
        "created_at": time.time()
    }
    _reminders[patient_id].append(reminder)
    return reminder


def get_reminders(patient_id):
    """Get all reminders for a patient."""
    return _reminders.get(patient_id, [])


def mark_reminder_done(patient_id, reminder_id):
    """Mark a reminder as completed."""
    for r in _reminders.get(patient_id, []):
        if r["id"] == reminder_id:
            r["status"] = "done"
            return True
    return False


# ------------------------------------------------------------------ #
#  UPLOADS                                                             #
# ------------------------------------------------------------------ #
def add_upload(patient_id, filename, filepath, file_type, original_name):
    """Record an uploaded file. file_type: scan, lab_report"""
    if patient_id not in _uploads:
        _uploads[patient_id] = []

    upload = {
        "id": str(uuid.uuid4())[:8],
        "filename": filename,
        "filepath": filepath,
        "file_type": file_type,
        "original_name": original_name,
        "evaluation": None,
        "sent_to_doctor": False,
        "uploaded_at": time.time()
    }
    _uploads[patient_id].append(upload)
    return upload


def get_uploads(patient_id):
    """Get all uploads for a patient."""
    return _uploads.get(patient_id, [])


def evaluate_upload(patient_id, upload_id):
    """Simple evaluation: labels result as 'Good' or 'Needs follow-up'."""
    for u in _uploads.get(patient_id, []):
        if u["id"] == upload_id:
            # Simple logic: PDFs/lab reports default to "Needs follow-up"
            # Scans check filename for known keywords
            name = u["original_name"].lower()
            if u["file_type"] == "lab_report":
                u["evaluation"] = "Needs follow-up"
            elif any(w in name for w in ["normal", "healthy", "clear", "negative"]):
                u["evaluation"] = "Good"
            else:
                u["evaluation"] = "Needs follow-up"
            return u["evaluation"]
    return None


# ------------------------------------------------------------------ #
#  DOCTOR REPORTS                                                      #
# ------------------------------------------------------------------ #
def send_to_doctor(patient_id, upload_id):
    """Send an upload to the doctor and generate a simple report."""
    uploads = _uploads.get(patient_id, [])
    target = None
    for u in uploads:
        if u["id"] == upload_id:
            target = u
            break

    if not target:
        return None

    # Auto-evaluate if not done yet
    if not target["evaluation"]:
        evaluate_upload(patient_id, upload_id)

    target["sent_to_doctor"] = True

    if patient_id not in _reports:
        _reports[patient_id] = []

    report = {
        "id": str(uuid.uuid4())[:8],
        "patient_id": patient_id,
        "upload_id": upload_id,
        "file_name": target["original_name"],
        "file_type": target["file_type"],
        "evaluation": target["evaluation"],
        "status": "Sent to Doctor",
        "created_at": time.time()
    }
    _reports[patient_id].append(report)
    return report


def get_reports(patient_id):
    """Get all doctor reports for a patient."""
    return _reports.get(patient_id, [])


def get_all_reports():
    """Get all reports across all patients (for doctor view)."""
    all_reports = []
    for pid, reports in _reports.items():
        for r in reports:
            all_reports.append(r)
    return all_reports
