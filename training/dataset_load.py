from pathlib import Path
import os
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

CLEANED_DATA_PATH = Path('artifacts/processed/cleaned_data.pkl')

def fetch_and_clean_data() -> pd.DataFrame:
    if CLEANED_DATA_PATH.exists():
        return pd.read_pickle(CLEANED_DATA_PATH)
    
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

    df = pd.DataFrame(all_rows)
    df = df.drop(columns=['customerID'], errors='ignore')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)

    CLEANED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(CLEANED_DATA_PATH)
    return df

def get_preprocessor(feature_columns):
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_cols = [c for c in feature_columns if c not in numeric_cols]

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    return ColumnTransformer([
        ('num', numeric_pipeline, numeric_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])

def get_data_splits():
    df = fetch_and_clean_data()
    train_full, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Churn'])
    train, val = train_test_split(train_full, test_size=0.25, random_state=42, stratify=train_full['Churn'])

    X_train, y_train = train.drop(columns=['Churn']), train['Churn']
    X_val, y_val = val.drop(columns=['Churn']), val['Churn']
    X_test, y_test = test.drop(columns=['Churn']), test['Churn']
    
    preprocessor = get_preprocessor(X_train.columns)
    return X_train, y_train, X_val, y_val, X_test, y_test, preprocessor