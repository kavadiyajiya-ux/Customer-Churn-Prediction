import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime

# ----------------------------------
# 1. DATABASE CONFIGURATION
# ----------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DB_DIR = BASE_DIR / "data"
DB_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DB_DIR / "campaigns.db"


# ----------------------------------
# 2. DATABASE CONNECTION
# ----------------------------------

def get_connection():
    return sqlite3.connect(DB_PATH)


# ----------------------------------
# 3. INITIALIZE DATABASE
# ----------------------------------

def initialize_database():

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS campaigns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT NOT NULL,
            campaign_name TEXT NOT NULL,
            offer_details TEXT,
            campaign_date TEXT NOT NULL,
            response TEXT DEFAULT 'Pending',
            outcome TEXT DEFAULT 'Not Evaluated',
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


# ----------------------------------
# 4. ADD NEW CAMPAIGN
# ----------------------------------

def add_campaign(
    customer_id,
    campaign_name,
    offer_details,
    campaign_date
):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO campaigns (
            customer_id,
            campaign_name,
            offer_details,
            campaign_date,
            response,
            outcome,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        customer_id,
        campaign_name,
        offer_details,
        campaign_date,
        "Pending",
        "Not Evaluated",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()


# ----------------------------------
# 5. GET ALL CAMPAIGNS
# ----------------------------------

def get_all_campaigns():

    initialize_database()

    conn = get_connection()

    query = """
        SELECT *
        FROM campaigns
        ORDER BY id DESC
    """

    df = pd.read_sql_query(query, conn)

    conn.close()

    return df


# ----------------------------------
# 6. UPDATE CUSTOMER RESPONSE
# ----------------------------------

def update_response(campaign_id, response):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""
        UPDATE campaigns
        SET response = ?
        WHERE id = ?
    """, (response, campaign_id))

    conn.commit()
    conn.close()


# ----------------------------------
# 7. UPDATE RETENTION OUTCOME
# ----------------------------------

def update_outcome(campaign_id, outcome):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""
        UPDATE campaigns
        SET outcome = ?
        WHERE id = ?
    """, (outcome, campaign_id))

    conn.commit()
    conn.close()


# ----------------------------------
# 8. DELETE CAMPAIGN
# ----------------------------------

def delete_campaign(campaign_id):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""
        DELETE FROM campaigns
        WHERE id = ?
    """, (campaign_id,))

    conn.commit()
    conn.close()


# ----------------------------------
# 9. GET CUSTOMER CAMPAIGNS
# ----------------------------------

def get_customer_campaigns(customer_id):

    conn = get_connection()

    query = """
        SELECT *
        FROM campaigns
        WHERE customer_id = ?
        ORDER BY id DESC
    """

    df = pd.read_sql_query(
        query,
        conn,
        params=(customer_id,)
    )

    conn.close()

    return df


# ----------------------------------
# 10. INITIALIZE DATABASE
# ----------------------------------

initialize_database()