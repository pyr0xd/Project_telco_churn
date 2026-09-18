# db.py
import json
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from config import config

logger = logging.getLogger(__name__)

def get_db_connection():
    """Establishes and returns a connection to the PostgreSQL/Supabase database."""
    try:
        conn = psycopg2.connect(config.DATABASE_URL)
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to the database: {e}")
        return None

def save_prediction_log(features: dict, prediction: str, probability: float, model_version: str = "v1.0") -> bool:
    """
    Saves a prediction record into the Supabase 'predictions_logs' table.
    """
    conn = get_db_connection()
    if not conn:
        return False

    sql = """
        INSERT INTO predictions_logs (features, prediction, probability, model_version)
        VALUES (%s, %s, %s, %s);
    """
    
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql, (json.dumps(features), str(prediction), float(probability), model_version))
            conn.commit()
            return True
    except Exception as e:
        logger.error(f"Error saving prediction to DB: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def fetch_recent_logs(limit: int = 10):
    """Fetches the most recent prediction logs from the database."""
    conn = get_db_connection()
    if not conn:
        return []

    sql = """
        SELECT id, created_at, features, prediction, probability, model_version
        FROM predictions_logs
        ORDER BY created_at DESC
        LIMIT %s;
    """
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(sql, (limit,))
            return cursor.fetchall()
    except Exception as e:
        logger.error(f"Error fetching logs: {e}")
        return []
    finally:
        conn.close()