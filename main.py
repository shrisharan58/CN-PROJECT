# main.py — CORRECTED & FAST
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import io
import os
import hashlib
import json
import math
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Any
import asyncio
from pydantic import BaseModel
from contextlib import asynccontextmanager 

# --- Initialization and Configuration ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    global model, le, scaler, feature_names
    print("Loading model artifacts...")
    try:
        # Load model
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(MODEL_PATH)
        model = joblib.load(MODEL_PATH)

        # Load LabelEncoder
        if not os.path.exists(LE_PATH):
            raise FileNotFoundError(LE_PATH)
        le = joblib.load(LE_PATH)

        # Load scaler
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(SCALER_PATH)
        scaler = joblib.load(SCALER_PATH)

        # Load feature names (try JSON then pickle)
        if not os.path.exists(FEATURE_NAMES_PATH):
            raise FileNotFoundError(FEATURE_NAMES_PATH)
        try:
            # try JSON
            with open(FEATURE_NAMES_PATH, 'r') as f:
                feature_names = json.load(f)
            if not isinstance(feature_names, list):
                raise ValueError("feature_names file did not contain a list.")
        except Exception:
            # fallback to joblib/pickle
            feature_names = joblib.load(FEATURE_NAMES_PATH)

        # Ensure feature_names is a list of strings
        feature_names = [str(x) for x in feature_names]

        # Load attacks lookup CSV (optional)
        load_attack_info()

        print("Model artifacts loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: Missing artifact {e.filename}. Please run training pipeline first.")
        raise HTTPException(status_code=500, detail=f"Missing model artifact: {e.filename}")
    except Exception as e:
        print(f"Unexpected error loading artifacts: {e}")
        raise HTTPException(status_code=500, detail="Model loading failed.")
    
    # --- This is where the application starts running ---
    yield
    # --- Code to run on shutdown (optional) ---
    print("Shutting down application...")


app = FastAPI(title="CyberShield AI - Advanced Threat Analysis", lifespan=lifespan)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Artifact Paths (env overrides allowed) ---
MODEL_PATH = os.environ.get("MODEL_PATH", "ids_model.pkl")
LE_PATH = os.environ.get("LE_PATH", "label_encoder.pkl")
SCALER_PATH = os.environ.get("SCALER_PATH", "feature_scaler.pkl")
FEATURE_NAMES_PATH = os.environ.get("FEATURE_NAMES_PATH", "feature_names.pkl")  # may be json or pkl
ATTACKS_LOOKUP = os.environ.get("ATTACKS_LOOKUP", "attacks.csv")
# --- NEW: Path to metrics file ---
METRICS_PATH = os.environ.get("METRICS_PATH", "model_metrics.json")

# Global variables for loaded artifacts
model = None
le = None
scaler = None
feature_names: List[str] = []
attack_info_lookup: Dict[str, Dict[str, Any]] = {}

# Columns that we consider metadata (keep in response)
# These are all lowercase
meta_columns = [
    'source ip', 'destination ip', 'source_ip', 'destination_ip',
    'source port', 'destination port', 'source_port', 'destination_port',
    'timestamp', 'flow duration', 'flow_duration'
]


# ---------------- Helper Functions ----------------
def load_attack_info():
    """Load attack descriptions and metadata from CSV into attack_info_lookup."""
    global attack_info_lookup
    try:
        if os.path.exists(ATTACKS_LOOKUP):
            df = pd.read_csv(ATTACKS_LOOKUP)
            # Standardize attack name column
            col_name = 'attack_name' if 'attack_name' in df.columns else 'Attack Type'
            df = df.where(pd.notnull(df), None)
            attack_info_lookup = df.set_index(col_name).to_dict('index')
            print(f"Successfully loaded {len(attack_info_lookup)} attack descriptions from {ATTACKS_LOOKUP}.")
        else:
            print(f"Warning: {ATTACKS_LOOKUP} not found. Using generic attack info.")
            attack_info_lookup = {}
    except Exception as e:
        print(f"Error loading {ATTACKS_LOOKUP}: {e}")
        attack_info_lookup = {}


def get_attack_details(label: str) -> Dict[str, Any]:
    """Return description, severity, mitigation for a label (fallbacks available)."""
    # Find details by exact match or case-insensitive
    details = attack_info_lookup.get(label)
    if details is None:
         for k, v in attack_info_lookup.items():
            if k.lower() == label.lower():
                details = v
                break
    if details is None:
        details = {}

    return {
        "description": details.get("description", "No description available for this threat type."),
        "severity": details.get("severity", "Medium" if label.lower() not in ['benign', 'normal'] else "Low"),
        "mitigation": details.get("mitigation", "General security best practices recommended.")
    }


def generate_risk_score(label: str, probability: float, severity: str) -> int:
    """Map model probability and severity to a 0-100 risk score."""
    # Note: probability is 0-1
    base_score = probability * 100
    if label.lower() in ['benign', 'normal', 'none']:
        # Low score for benign
        score = max(5 - (base_score / 20), 0)
    elif severity == 'Critical':
        score = min(90 + (base_score / 10), 100)
    elif severity == 'High':
        score = min(70 + (base_score / 3.3), 90)
    elif severity == 'Medium':
        score = min(40 + (base_score / 2), 70)
    elif severity == 'Low':
        score = min(10 + (base_score / 3.3), 40)
    else:
        score = min(40 + (base_score / 2), 70) # Default to medium
    return int(round(score))

# Vectorize for speed
v_generate_risk_score = np.vectorize(generate_risk_score)


def generate_id(row: pd.Series) -> str:
    """Create short unique id from row metadata fields. Expects a pandas Series."""
    ts = row.get('timestamp', row.get('time', ''))
    src = row.get('source ip', row.get('source_ip', ''))
    dst = row.get('destination ip', row.get('destination_ip', ''))
    dport = str(row.get('destination port', row.get('destination_port', '')))
    hash_input = f"{ts}-{src}-{dst}-{dport}"
    # stable short id
    return hashlib.md5(hash_input.encode()).hexdigest()[:8]


# ---------------- API Endpoints ----------------
@app.get("/")
def read_root():
    return {"status": "CyberShield AI API is running."}


# ---
# --- NEW ENDPOINT: Get model training details
# ---
@app.get("/api/model-details")
def get_model_details():
    if not os.path.exists(METRICS_PATH):
        raise HTTPException(status_code=404, detail="Model metrics file not found. Please train the model first.")
    try:
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
        return JSONResponse(content=metrics)
    except Exception as e:
        print(f"Error reading metrics file: {e}")
        raise HTTPException(status_code=500, detail="Could not read model metrics file.")


# Pydantic model for manual scan endpoint
class ManualScanRequest(BaseModel):
    src_ip: str
    dst_ip: str
    protocol: str
    src_port: str
    dst_port: str


@app.post("/api/scan-single")
async def scan_single_connection(request: ManualScanRequest):
    """
    Mocked endpoint for single-connection scan (limited inputs).
    Returns a simulated result based on dst_port or protocol.
    """
    try:
        payload = request.dict()
        # Mock rules (same as frontend mock)
        if payload.get('dst_port') == '22':
            result = {"label": "SSH Brute-force", "risk_score": 85, "severity": "High",
                      "description": "Detected potential brute-force attempt on SSH port."}
        elif payload.get('dst_port') in ('80', '8080', '443'):
            result = {"label": "Web Attack (XSS)", "risk_score": 70, "severity": "High",
                      "description": "Potential Cross-Site Scripting attack vector detected."}
        elif payload.get('protocol', '').lower() == 'icmp':
            result = {"label": "Ping Sweep", "risk_score": 40, "severity": "Medium",
                      "description": "ICMP echo request detected, part of a potential network scan."}
        else:
            result = {"label": "Benign", "risk_score": 5, "severity": "Low",
                      "description": "Traffic appears to be normal user activity."}
        return JSONResponse(content=result)
    except Exception as e:
        print(f"Manual scan failed: {e}")
        raise HTTPException(status_code=500, detail="Manual scan analysis failed.")


@app.post("/api/analyze")
async def analyze_traffic(file: UploadFile = File(...)):
    """
    Batch CSV analysis endpoint. Expects a CSV file. Uses vectorized operations.
    """
    # Ensure artifacts loaded
    if not model or not le or not scaler or not feature_names:
        raise HTTPException(status_code=503, detail="Model artifacts are not loaded.")

    # Validate file extension
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .csv file.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), low_memory=False) 
        
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty.")

        # Strip whitespace from columns before processing
        df.columns = [c.strip() for c in df.columns]

        # Keep original metadata for response (safe selection)
        metadata_cols_present = [c for c in df.columns if c.lower() in meta_columns]
        
        if not metadata_cols_present:
            metadata_df = pd.DataFrame(index=df.index)
        else:
            metadata_df = df[metadata_cols_present].copy()

        # ---
        # --- CRITICAL FIX: Find and remove the Label column, just like in training
        # ---
        cols_lower = {c.lower(): c for c in df.columns} # Use already-stripped names
        label_candidates = ['attack type', 'attack_type', 'attack', 'class', 'label']
        label_col = None
        for cand in label_candidates:
            if cand in cols_lower:
                label_col = cols_lower[cand]
                break
        
        # ---
        # --- Preprocessing: remove metadata AND label column
        # ---
        cols_to_drop = metadata_cols_present
        if label_col:
            print(f"Found and removing label column from prediction data: {label_col}")
            cols_to_drop.append(label_col)
        else:
            print("Warning: Could not find label column to drop for prediction.")

        df_processed = df.drop(columns=cols_to_drop, errors='ignore').copy()

        # ---
        # --- START PREPROCESSING (Mirrors train_model.py)
        # ---

        # 1. Identify true categorical columns BEFORE converting
        categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()

        # 2. Convert all other columns to numeric, replacing 'inf' and errors
        for col in df_processed.columns:
            if col not in categorical_cols:
                # Convert to numeric, errors will become NaN
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        # 3. Replace Inf and NaN with 0 (this is now safe)
        df_processed = df_processed.replace([np.inf, -np.inf], 0).fillna(0)

        # 4. Now, run OHE on the TRUE categorical columns
        if categorical_cols:
            print(f"One-Hot Encoding categorical features: {categorical_cols}")
            df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
        else:
            print("No categorical features found to encode.")

        # ---
        # --- END PREPROCESSING
        # ---

        # Ensure all expected model features are present
        df_processed = df_processed.reindex(columns=feature_names, fill_value=0)
        
        # Feature Scaling
        X_scaled = scaler.transform(df_processed.values)

        # --- Predictions ---
        probabilities = model.predict_proba(X_scaled)  # shape (n_samples, n_classes)
        predictions = model.predict(X_scaled)
        
        # ---
        # --- START: Vectorized Analysis (MUCH FASTER)
        # ---
        
        # 1. Create base results DataFrame
        results_df = pd.DataFrame({
            'label': le.inverse_transform(predictions),
            'probability': np.max(probabilities, axis=1) # Get max prob for each row
        })

        # 2. Get attack details (vectorized)
        unique_labels = results_df['label'].unique()
        details_map = {label: get_attack_details(label) for label in unique_labels}
        
        results_df['details'] = results_df['label'].map(details_map)
        results_df['severity'] = results_df['details'].apply(lambda x: x.get('severity', 'Medium'))
        results_df['description'] = results_df['details'].apply(lambda x: x.get('description', 'No description'))
        
        # 3. Generate risk scores (vectorized)
        results_df['risk_score'] = v_generate_risk_score(
            results_df['label'],
            results_df['probability'], # probability is 0-1
            results_df['severity']
        )
        
        # 4. Clean metadata and combine
        # Create a map of original (spaced) column names to clean ones
        meta_lookup = {c: c.lower().strip() for c in metadata_df.columns}
        clean_meta_df = metadata_df.rename(columns=meta_lookup)
        
        # Combine all data
        df_final = pd.concat([clean_meta_df, results_df], axis=1)

        # 5. Generate IDs (fastest way is .apply)
        df_final['id'] = df_final.apply(generate_id, axis=1)
        
        # 6. Prepare final JSON fields
        def safe_int(val):
            try: return int(float(val))
            except Exception: return 0
        
        # ---
        # --- This section is now robust and handles missing columns ---
        # ---
        if 'source ip' in df_final.columns:
            df_final['src_ip'] = df_final['source ip']
        elif 'source_ip' in df_final.columns:
            df_final['src_ip'] = df_final['source_ip']
        else:
            df_final['src_ip'] = None

        if 'destination ip' in df_final.columns:
            df_final['dst_ip'] = df_final['destination ip']
        elif 'destination_ip' in df_final.columns:
            df_final['dst_ip'] = df_final['destination_ip']
        else:
            df_final['dst_ip'] = None

        if 'source port' in df_final.columns:
            df_final['src_port'] = df_final['source port'].apply(safe_int)
        elif 'source_port' in df_final.columns:
            df_final['src_port'] = df_final['source_port'].apply(safe_int)
        else:
            df_final['src_port'] = 0

        if 'destination port' in df_final.columns:
            df_final['dst_port'] = df_final['destination port'].apply(safe_int)
        elif 'destination_port' in df_final.columns:
            df_final['dst_port'] = df_final['destination_port'].apply(safe_int)
        else:
            df_final['dst_port'] = 0
            
        if 'timestamp' in df_final.columns:
             df_final['timestamp'] = df_final['timestamp'].fillna(datetime.now(timezone.utc).isoformat())
        else:
            df_final['timestamp'] = datetime.now(timezone.utc).isoformat()
            
        df_final['probability'] = df_final['probability'] * 100 # Convert to percentage

        # 7. Select final columns and convert to list of dicts
        final_columns = [
            'id', 'label', 'probability', 'risk_score', 'description', 
            'severity', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'timestamp'
        ]
        # Ensure all columns exist, fill with None if not
        for col in final_columns:
            if col not in df_final:
                df_final[col] = None
                
        all_predictions = df_final[final_columns].to_dict('records')
        
        # ---
        # --- END: Vectorized Analysis
        # ---
        
        # --- Build response ---
        summary = {
            "total_connections": int(len(df_final)),
            "attack_count": 0,
            "active_threats": 0,
            "attack_types_distribution": [],
            "overall_health": 100.0,
            "traffic_series": []
        }

        # Calculate summary stats (fast)
        attack_df = df_final[df_final['label'].str.lower().isin(['benign', 'normal', 'none']) == False]
        summary["attack_count"] = int(len(attack_df))
        
        if not attack_df.empty:
            attack_counts = attack_df['label'].value_counts()
            summary["active_threats"] = int(len(attack_counts))
            
            # Get severity for each attack type for the pie chart
            attack_severities = attack_counts.index.to_series().map(details_map).apply(lambda x: x.get('severity', 'Medium'))
            
            summary["attack_types_distribution"] = [
                {"name": k, "value": int(v), "risk_score": attack_severities[k]}
                for k, v in attack_counts.items()
            ]
            
            # Find highest risk prediction
            highest_risk_prediction_series = attack_df.loc[attack_df['risk_score'].idxmax()]
            highest_risk_prediction = highest_risk_prediction_series.to_dict()
            # Convert numpy types to python types for JSON
            highest_risk_prediction = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in highest_risk_prediction.items() if k in final_columns}

        else:
            summary["active_threats"] = 0
            summary["attack_types_distribution"] = []
            highest_risk_prediction = None


        # Calculate overall health
        if summary["total_connections"] > 0:
            health_penalty = (summary["attack_count"] / summary["total_connections"]) * 100 * 2
            overall_health = max(100 - health_penalty, 0)
        else:
            overall_health = 100.0
        summary["overall_health"] = float(f"{overall_health:.2f}")

        # Mock Traffic Series (fast)
        total_attackers = summary["attack_count"]
        total_benign = summary["total_connections"] - total_attackers
        summary["traffic_series"] = [
            {"name": "Start", "benign": total_benign * 0.9, "attack": total_attackers * 0.1},
            {"name": "Mid-Hr", "benign": total_benign * 0.8, "attack": total_attackers * 0.2},
            {"name": "Now", "benign": total_benign, "attack": total_attackers},
        ]

        response_data = {
            "summary": summary,
            "predictions": all_predictions,
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_analyzed": int(len(df_final)),
            "highest_risk_prediction": highest_risk_prediction,
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        # Log the exception (avoid leaking internals to client)
        print(f"Analysis failed: {repr(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An error occurred during file processing.")

# ---
# --- REMOVED WebSocket and Live Feed code ---
# ---