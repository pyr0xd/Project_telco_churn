import os
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

#Function to fetch the dataset from the database

#call "fetch_dataset" to use dataset

def fetch_dataset() -> pd.DataFrame:
    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    supabase = create_client(url, key)

    all_rows = []
    start = 0
    batch_size = 1000

    while True:
        response = supabase.table("telco_churn").select("*").range(start, start + batch_size - 1).execute()
        if not response.data:
            break
        all_rows.extend(response.data)
        start += batch_size

    return pd.DataFrame(all_rows)