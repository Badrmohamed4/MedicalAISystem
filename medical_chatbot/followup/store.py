"""
Follow-Up Data Store — In-memory
Keyed by patient_id (= Flask session user_id).
"""
import time
import uuid

_reminders = {}
_uploads   = {}
_reports   = {}


# ------------------------------------------------------------------ #
#  REMINDERS                                                          #
# ------------------------------------------------------------------ #
def add_reminder(patient_id, reminder_type, title, date_time, notes=""):
    _reminders.setdefault(patient_id, [])
    r = {
        "id": str(uuid.uuid4())[:8],
        "type": reminder_type,
        "title": title,
        "date_time": date_time,
        "notes": notes,
        "status": "pending",
        "link": None,
        "attached_upload": None,
        "created_at": time.time()
    }
    _reminders[patient_id].append(r)
    return r


def get_reminders(patient_id):
    reminders = _reminders.get(patient_id, [])
    now_str = time.strftime("%Y-%m-%dT%H:%M")
    for r in reminders:
        if r["status"] == "pending" and r.get("date_time") and r["date_time"] < now_str:
            r["status"] = "overdue"
    return sorted(reminders, key=lambda r: (r.get("date_time") == "", r.get("date_time", "")))


def get_all_patients_reminders():
    """Summary of every patient for the overview page."""
    result = []
    for pid, reminders in _reminders.items():
        # re-run overdue check
        now_str = time.strftime("%Y-%m-%dT%H:%M")
        for r in reminders:
            if r["status"] == "pending" and r.get("date_time") and r["date_time"] < now_str:
                r["status"] = "overdue"
        result.append({
            "patient_id":    pid,
            "total":         len(reminders),
            "pending_count": sum(1 for r in reminders if r["status"] == "pending"),
            "overdue_count": sum(1 for r in reminders if r["status"] == "overdue"),
            "done_count":    sum(1 for r in reminders if r["status"] == "done"),
            "last_activity": max(r["created_at"] for r in reminders)
        })
    return sorted(result, key=lambda p: p["last_activity"], reverse=True)


def mark_reminder_done(patient_id, reminder_id):
    for r in _reminders.get(patient_id, []):
        if r["id"] == reminder_id:
            r["status"] = "done"
            return True
    return False


def set_reminder_link(patient_id, reminder_id, link):
    for r in _reminders.get(patient_id, []):
        if r["id"] == reminder_id:
            r["link"] = link
            return r
    return None


def attach_scan_to_reminder(patient_id, reminder_id, upload_id):
    for r in _reminders.get(patient_id, []):
        if r["id"] == reminder_id:
            r["attached_upload"] = upload_id
            return r
    return None


# ------------------------------------------------------------------ #
#  UPLOADS                                                            #
# ------------------------------------------------------------------ #
def add_upload(patient_id, filename, filepath, file_type, original_name):
    _uploads.setdefault(patient_id, [])
    u = {
        "id": str(uuid.uuid4())[:8],
        "filename": filename,
        "filepath": filepath,
        "file_type": file_type,
        "original_name": original_name,
        "evaluation": None,
        "sent_to_doctor": False,
        "uploaded_at": time.time()
    }
    _uploads[patient_id].append(u)
    return u


def get_uploads(patient_id):
    return _uploads.get(patient_id, [])


def evaluate_upload(patient_id, upload_id):
    for u in _uploads.get(patient_id, []):
        if u["id"] == upload_id:
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
#  DOCTOR REPORTS                                                     #
# ------------------------------------------------------------------ #
def send_to_doctor(patient_id, upload_id):
    target = next((u for u in _uploads.get(patient_id, []) if u["id"] == upload_id), None)
    if not target:
        return None
    if not target["evaluation"]:
        evaluate_upload(patient_id, upload_id)
    target["sent_to_doctor"] = True
    _reports.setdefault(patient_id, [])
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
    return _reports.get(patient_id, [])


def get_all_reports():
    return [r for reports in _reports.values() for r in reports]