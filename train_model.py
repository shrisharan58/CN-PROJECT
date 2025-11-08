import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import joblib
import os
import json
from datetime import datetime # <--- THIS IS THE FIX

# --- Configuration ---
DATASET_PATH = "dataset.csv" 
ATTACKS_LOOKUP = "attacks.csv" 
MODEL_OUT = "ids_model.pkl"
LABELER_OUT = "label_encoder.pkl"
SCALER_OUT = "feature_scaler.pkl"
FEATURES_OUT = "feature_names.pkl"
# --- NEW: Output file for model metrics ---
METRICS_OUT = "model_metrics.json"

META_COLUMNS = [
    'source ip', 'destination ip', 'source_ip', 'destination_ip', 'source port', 'destination port',
    'source_port', 'destination_port', 'timestamp', 'flow duration', 'flow_duration'
]

def load_dataset(path=DATASET_PATH):
    """Loads the dataset from a CSV file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Please ensure {path} is in the directory.")
    print(f"Loading dataset from {path}...")
    return pd.read_csv(path, low_memory=False)

def get_attack_info(attack_label):
    """Loads and looks up attack information from attacks.csv."""
    try:
        attacks_df = pd.read_csv(ATTACKS_LOOKUP)
        col_name = 'attack_name' if 'attack_name' in attacks_df.columns else 'Attack Type'
        info = attacks_df[attacks_df[col_name].str.lower() == attack_label.lower()]
        
        if not info.empty:
            return info.iloc[0].to_dict()
        return {"description": "No specific details found for this attack type.", "severity": "Medium"}
    except Exception as e:
        print(f"Warning: Could not load or parse {ATTACKS_LOOKUP}. Error: {e}")
        return {"description": "Detailed attack info unavailable.", "severity": "Unknown"}

def preprocess(df):
    """Cleans, encodes categorical features, and prepares data for training."""
    df = df.copy()
    
    # 1. Standardize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]
    cols_lower = {c.lower(): c for c in df.columns}
    
    # 2. Extract and encode Label
    label_candidates = ['attack type', 'attack_type', 'attack', 'class', 'label']
    label_col = None
    for cand in label_candidates:
        if cand in cols_lower:
            label_col = cols_lower[cand]
            break
    if label_col is None:
        raise ValueError("Label column not found. Ensure dataset has 'Attack Type' or 'class' or 'label'.")
    print(f"Found label column: '{label_col}'")

    le = LabelEncoder()
    df[label_col] = df[label_col].astype(str).str.lower().str.strip()
    
    # Ensure 'benign' is a known class
    if 'benign' not in df[label_col].unique():
        print("Warning: 'benign' class not found in labels. Adding a mock 'benign' entry for encoder.")
        temp_labels = pd.concat([df[label_col], pd.Series(['benign'])])
        y = le.fit_transform(temp_labels)
        y = y[:-1] # Remove the mock entry after fitting
    else:
        y = le.fit_transform(df[label_col])
    
    print(f"Target classes found: {le.classes_}")

    # 3. Drop metadata and label column
    features_df = df.drop(columns=[c for c in df.columns if c.lower() in META_COLUMNS or c == label_col], errors='ignore')
    
    # ---
    # --- START PREPROCESSING FIX ---
    # ---
    
    # 4. Identify true categorical columns BEFORE converting
    categorical_cols = features_df.select_dtypes(include=['object']).columns.tolist()

    # 5. Convert all other columns to numeric, replacing 'inf' and errors
    for col in features_df.columns:
        if col not in categorical_cols:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
    
    # 6. Replace Inf and NaN with 0 (this is now safe)
    features_df = features_df.replace([np.inf, -np.inf], 0).fillna(0)

    # 7. Now, run OHE on the TRUE categorical columns
    if categorical_cols:
        print(f"One-Hot Encoding categorical features: {categorical_cols}")
        features_df = pd.get_dummies(features_df, columns=categorical_cols, drop_first=True)
    else:
        print("No categorical features found to encode.")
    
    # ---
    # --- END PREPROCESSING FIX ---
    # ---
    
    X = features_df.values
    feature_names = features_df.columns.tolist()
    
    print(f"Pre-processing complete. {len(feature_names)} features generated.")
    return X, y, le, feature_names

def train_and_save(X, y, le, feature_names):
    """Trains the model, scales features, and saves all artifacts."""
    print("Splitting data and training model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 1. Feature Scaling (fit only on training data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. Model Training (Using XGBoost Classifier)
    print("Training XGBoost model...")
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(le.classes_),
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # 3. Evaluation
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    target_names = [str(c) for c in le.classes_]
    report_dict = classification_report(y_test, y_pred, target_names=target_names, zero_division=0, output_dict=True)
    report_str = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
    
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", report_str)
    
    # ---
    # --- NEW: Save metrics to JSON for the frontend
    # ---
    metrics_data = {
        "accuracy": accuracy,
        "classification_report": report_dict,
        "classes": target_names,
        "training_timestamp": datetime.now().isoformat()
    }
    with open(METRICS_OUT, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Model metrics saved to {METRICS_OUT}")
    
    # 4. Save artifacts
    print("Saving model artifacts...")
    joblib.dump(model, MODEL_OUT)
    joblib.dump(le, LABELER_OUT)
    joblib.dump(scaler, SCALER_OUT)
    
    with open(FEATURES_OUT, 'w') as f:
        json.dump(feature_names, f)
    
    print(f"Training complete. Artifacts saved: {MODEL_OUT}, {LABELER_OUT}, {SCALER_OUT}, {FEATURES_OUT}")
    
    return model, le, scaler, feature_names

if __name__ == "__main__":
    try:
        df_main = load_dataset(DATASET_PATH)
        X_main, y_main, le_main, feature_names_main = preprocess(df_main)
        train_and_save(X_main, y_main, le_main, feature_names_main)

    except FileNotFoundError as e:
        print(f"FATAL ERROR: {e}. Cannot run without the required files.")
    except ValueError as e:
        print(f"FATAL ERROR: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during training: {e}")
        import traceback
        traceback.print_exc()