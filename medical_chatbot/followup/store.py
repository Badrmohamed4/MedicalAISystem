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
        "prediction": None,
        "confidence": None,
        "model_type": None,
        "sent_to_doctor": False,
        "uploaded_at": time.time()
    }
    _uploads[patient_id].append(u)
    return u


def get_uploads(patient_id):
    return _uploads.get(patient_id, [])


def evaluate_upload(patient_id, upload_id, model_type=None):
    for u in _uploads.get(patient_id, []):
        if u["id"] == upload_id:
            if u["file_type"] == "lab_report":
                u["evaluation"] = "Needs follow-up"
                u["prediction"] = None
                u["confidence"] = None
                u["model_type"] = None
                return u["evaluation"]

            # Run AI model inference using the model type from the patient conversation
            try:
                from medical_chatbot.utils.image_processor import ModelWrapper

                # model_type must come from the conversation context (brain/lung/skin)
                # fallback to brain if not provided
                resolved_type = model_type if model_type in ["brain", "lung", "skin"] else "brain"

                wrapper = ModelWrapper()
                label, conf = wrapper.predict(u["filepath"], model_type=resolved_type)

                u["prediction"] = label
                u["confidence"] = round(float(conf) * 100, 1)
                u["model_type"] = resolved_type
                u["evaluation"] = "Good" if any(w in label.lower() for w in ["normal", "no tumor"]) else "Needs follow-up"

            except Exception as e:
                print(f"[Store] Model inference error: {e}")
                u["prediction"] = "Error running model"
                u["confidence"] = None
                u["model_type"] = model_type
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
        "prediction": target.get("prediction"),
        "confidence": target.get("confidence"),
        "model_type": target.get("model_type"),
        "status": "Sent to Doctor",
        "created_at": time.time()
    }
    _reports[patient_id].append(report)
    return report


def get_reports(patient_id):
    return _reports.get(patient_id, [])


def get_all_reports():
    return [r for reports in _reports.values() for r in reports]