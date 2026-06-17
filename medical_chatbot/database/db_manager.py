"""
Database Manager — PostgreSQL persistence for Medical AI System
===============================================================
Handles all database operations using PostgreSQL.
Connection: localhost:5432, database: medical_ai_db
"""

import psycopg2
import psycopg2.extras
import json
import os
from datetime import datetime

# ── Connection Configuration ──────────────────────────────────────────
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "medical_ai_db",
    "user": "postgres",
    "password": "postgres"
}


def get_connection():
    """Get a PostgreSQL connection."""
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    return conn


# ══════════════════════════════════════════════════════════════════════
# SCHEMA CREATION
# ══════════════════════════════════════════════════════════════════════

def init_db():
    """Creates all tables if they don't exist. Safe to call multiple times."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id      TEXT PRIMARY KEY,
            created_at      TIMESTAMP NOT NULL DEFAULT NOW(),
            updated_at      TIMESTAMP NOT NULL DEFAULT NOW(),
            medical_context TEXT DEFAULT 'none',
            risk_level      TEXT DEFAULT 'Unknown',
            mode            TEXT DEFAULT 'patient',
            context_json    JSONB,
            is_active       BOOLEAN DEFAULT TRUE
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            message_id      SERIAL PRIMARY KEY,
            session_id      TEXT NOT NULL REFERENCES sessions(session_id),
            role            TEXT NOT NULL,
            content         TEXT NOT NULL,
            mode            TEXT DEFAULT 'patient',
            timestamp       TIMESTAMP NOT NULL DEFAULT NOW()
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS symptoms (
            symptom_id      SERIAL PRIMARY KEY,
            session_id      TEXT NOT NULL REFERENCES sessions(session_id),
            raw_term        TEXT NOT NULL,
            normalized_term TEXT,
            severity        TEXT,
            timestamp       TIMESTAMP NOT NULL DEFAULT NOW()
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            image_id        SERIAL PRIMARY KEY,
            session_id      TEXT NOT NULL REFERENCES sessions(session_id),
            filename        TEXT NOT NULL,
            filepath        TEXT NOT NULL,
            image_type      TEXT DEFAULT 'unknown',
            upload_time     TIMESTAMP NOT NULL DEFAULT NOW()
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id   SERIAL PRIMARY KEY,
            session_id      TEXT NOT NULL REFERENCES sessions(session_id),
            image_id        INTEGER REFERENCES images(image_id),
            model_used      TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            confidence      REAL NOT NULL,
            timestamp       TIMESTAMP NOT NULL DEFAULT NOW()
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            report_id           SERIAL PRIMARY KEY,
            session_id          TEXT NOT NULL REFERENCES sessions(session_id),
            structured_report   TEXT,
            ai_assessment       TEXT,
            risk_level          TEXT,
            symptoms_snapshot   JSONB,
            generated_at        TIMESTAMP NOT NULL DEFAULT NOW()
        )
    """)

    # Index for fast session lookups
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_session
        ON messages(session_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_symptoms_session
        ON symptoms(session_id)
    """)

    conn.commit()
    conn.close()
    print("[DB] PostgreSQL database initialized successfully.")


# ══════════════════════════════════════════════════════════════════════
# SESSION OPERATIONS
# ══════════════════════════════════════════════════════════════════════

def save_session(session_id, context_dict, medical_context=None,
                 risk_level=None, mode="patient"):
    """Saves or updates a session."""
    now = datetime.now()
    context_json = json.dumps(context_dict)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT session_id FROM sessions WHERE session_id = %s",
        (session_id,)
    )
    existing = cursor.fetchone()

    if existing:
        cursor.execute("""
            UPDATE sessions
            SET updated_at = %s, medical_context = %s,
                risk_level = %s, mode = %s, context_json = %s
            WHERE session_id = %s
        """, (now, medical_context or 'none',
              risk_level or 'Unknown', mode,
              context_json, session_id))
    else:
        cursor.execute("""
            INSERT INTO sessions
                (session_id, created_at, updated_at,
                 medical_context, risk_level, mode, context_json)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (session_id, now, now,
              medical_context or 'none',
              risk_level or 'Unknown',
              mode, context_json))

    conn.commit()
    conn.close()


def load_session(session_id):
    """Loads a session from the database."""
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cursor.execute(
        "SELECT * FROM sessions WHERE session_id = %s",
        (session_id,)
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    result = dict(row)
    if result.get("context_json"):
        result["context"] = result["context_json"]
    return result


def get_all_sessions():
    """Returns all active sessions for doctor console."""
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cursor.execute("""
        SELECT session_id, created_at, updated_at,
               medical_context, risk_level, mode
        FROM sessions
        WHERE is_active = TRUE
        ORDER BY updated_at DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


# ══════════════════════════════════════════════════════════════════════
# MESSAGE OPERATIONS
# ══════════════════════════════════════════════════════════════════════

def save_message(session_id, role, content, mode="patient"):
    """Saves a single message."""
    conn = get_connection()
    conn.cursor().execute("""
        INSERT INTO messages (session_id, role, content, mode)
        VALUES (%s, %s, %s, %s)
    """, (session_id, role, content, mode))
    conn.commit()
    conn.close()


def get_messages(session_id):
    """Returns all messages for a session."""
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cursor.execute("""
        SELECT role, content, mode, timestamp
        FROM messages
        WHERE session_id = %s
        ORDER BY timestamp ASC
    """, (session_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


# ══════════════════════════════════════════════════════════════════════
# SYMPTOM OPERATIONS
# ══════════════════════════════════════════════════════════════════════

def save_symptoms(session_id, symptoms_list, severity=None):
    """Saves new symptoms — avoids duplicates."""
    if not symptoms_list:
        return

    conn = get_connection()
    cursor = conn.cursor()

    # Get already-saved symptoms
    cursor.execute(
        "SELECT raw_term FROM symptoms WHERE session_id = %s",
        (session_id,)
    )
    existing = set(row[0] for row in cursor.fetchall())

    for symptom in symptoms_list:
        if symptom and symptom not in existing:
            cursor.execute("""
                INSERT INTO symptoms (session_id, raw_term, severity)
                VALUES (%s, %s, %s)
            """, (session_id, symptom, severity))
            existing.add(symptom)

    conn.commit()
    conn.close()


def get_symptoms(session_id):
    """Returns all symptoms for a session."""
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cursor.execute("""
        SELECT raw_term, normalized_term, severity, timestamp
        FROM symptoms WHERE session_id = %s
        ORDER BY timestamp ASC
    """, (session_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


# ══════════════════════════════════════════════════════════════════════
# IMAGE + PREDICTION OPERATIONS
# ══════════════════════════════════════════════════════════════════════

def save_image(session_id, filename, filepath, image_type="unknown"):
    """Saves an uploaded image record."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO images (session_id, filename, filepath, image_type)
        VALUES (%s, %s, %s, %s) RETURNING image_id
    """, (session_id, filename, filepath, image_type))

    image_id = cursor.fetchone()[0]
    conn.commit()
    conn.close()
    return image_id


def save_prediction(session_id, model_used, predicted_class,
                    confidence, image_id=None):
    """Saves a CNN prediction result."""
    conn = get_connection()
    conn.cursor().execute("""
        INSERT INTO predictions
            (session_id, image_id, model_used, predicted_class, confidence)
        VALUES (%s, %s, %s, %s, %s)
    """, (session_id, image_id, model_used, predicted_class, confidence))
    conn.commit()
    conn.close()


# ══════════════════════════════════════════════════════════════════════
# REPORT OPERATIONS
# ══════════════════════════════════════════════════════════════════════

def save_report(session_id, structured_report, ai_assessment,
                risk_level, symptoms_snapshot):
    """Saves a generated doctor report."""
    conn = get_connection()
    conn.cursor().execute("""
        INSERT INTO reports
            (session_id, structured_report, ai_assessment,
             risk_level, symptoms_snapshot)
        VALUES (%s, %s, %s, %s, %s)
    """, (session_id, structured_report, ai_assessment,
          risk_level, json.dumps(symptoms_snapshot)))
    conn.commit()
    conn.close()


def get_latest_report(session_id):
    """Returns the most recent report for a session."""
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cursor.execute("""
        SELECT * FROM reports
        WHERE session_id = %s
        ORDER BY generated_at DESC
        LIMIT 1
    """, (session_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


# ══════════════════════════════════════════════════════════════════════
# STATS
# ══════════════════════════════════════════════════════════════════════

def get_stats():
    """Returns aggregate statistics for the dashboard."""
    conn = get_connection()
    cursor = conn.cursor()

    total = cursor.execute(
        "SELECT COUNT(*) FROM sessions"
    )
    total = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COUNT(*) FROM sessions WHERE medical_context = 'brain'"
    )
    brain = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COUNT(*) FROM sessions WHERE medical_context = 'lung'"
    )
    lung = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COUNT(*) FROM sessions WHERE risk_level = 'High'"
    )
    high_risk = cursor.fetchone()[0]

    conn.close()
    return {
        "total_sessions": total,
        "brain_sessions": brain,
        "lung_sessions": lung,
        "high_risk_sessions": high_risk
    }
