import os
import pandas as pd
import joblib
from dotenv import load_dotenv
from supabase import create_client
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

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

df = fetch_dataset()
df = df.drop(columns=['customerID'])  # bara en identifierare, ingen feature
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# de ~11 tomma raderna blir NaN här — hanteras i imputer-steget nedan
df['Churn'] = (df['Churn'] == 'Yes').astype(int)
from sklearn.model_selection import train_test_split

train_full, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Churn'])
train, val = train_test_split(train_full, test_size=0.25, random_state=42, stratify=train_full['Churn'])

X_train, y_train = train.drop(columns=['Churn']), train['Churn']
X_val, y_val = val.drop(columns=['Churn']), val['Churn']
X_test, y_test = test.drop(columns=['Churn']), test['Churn']

# Vilka kolumner är numeriska respektive kategoriska (allt annat)
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

# Bearbetning för numeriska kolumner: fyll saknade värden med median, skala sen till medelvärde 0 / std 1
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Bearbetning för kategoriska kolumner: one-hot-encoda till dummy-kolumner
categorical_pipeline = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Kör rätt bearbetning på rätt kolumner, allt i ett steg
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

# Slå ihop preprocessing + modell i en pipeline så allt körs i en enda .fit()
full_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('model', LogisticRegression(max_iter=1000))
])
full_pipeline.fit(X_train, y_train)

