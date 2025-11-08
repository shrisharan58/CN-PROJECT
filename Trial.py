from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
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
import re
from datetime import datetime, timezone
from typing import List, Dict, Any
import asyncio
from pydantic import BaseModel
import httpx # For async IP lookups
import xgboost as xgb # Required to unpickle the XGBoost model

# --- Initialization and Configuration ---\
app = FastAPI(title="CyberShield AI - Advanced Threat Analysis")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Artifact Paths (env overrides allowed) ---\
MODEL_PATH = os.environ.get("MODEL_PATH", "ids_model.pkl")
LE_PATH = os.environ.get("LE_PATH", "label_encoder.pkl")
SCALER_PATH = os.environ.get("SCALER_PATH", "feature_scaler.pkl")
FEATURE_NAMES_PATH = os.environ.get("FEATURE_NAMES_PATH", "feature_names.pkl")
ATTACKS_LOOKUP_PATH = os.environ.get("ATTACKS_LOOKUP", "attacks.csv")

# Global variables for loaded artifacts
model = None
le = None
scaler = None
feature_names = []
attack_info_map = {}

# --- Pydantic Models ---
class ManualInput(BaseModel):
    """Defines the structure for manual data entry."""
    data: Dict[str, Any]

class ThreatData(BaseModel):
    """Defines the structure of a threat object."""
    id: str
    timestamp: str
    ip: str
    port: str
    type: str
    severity: str
    confidence: str
    source: str
    geo: str
    raw_data: Dict[str, Any]


# --- Helper Functions ---

def sanitize_col_name(col_name: str) -> str:
    """Sanitizes column names to match training format."""
    if not isinstance(col_name, str):
        col_name = str(col_name)
    col_name = col_name.strip()
    col_name = re.sub(r'[^a-zA-Z0-9_]+', '_', col_name)
    if re.match(r'^\d', col_name):
        col_name = '_' + col_name
    return col_name

def load_artifacts():
    """Loads all ML artifacts into global variables."""
    global model, le, scaler, feature_names, attack_info_map
    print("Loading ML artifacts...")
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"{MODEL_PATH} not found. Please run train_model.py first.")
        
        model = joblib.load(MODEL_PATH)
        le = joblib.load(LE_PATH)
        scaler = joblib.load(SCALER_PATH)
        
        with open(FEATURE_NAMES_PATH, 'r') as f:
            feature_names = json.load(f)
            
        # Load attack descriptions
        if os.path.exists(ATTACKS_LOOKUP_PATH):
            attacks_df = pd.read_csv(ATTACKS_LOOKUP_PATH)
            # Standardize 'name' column for lookup
            if 'name' in attacks_df.columns:
                attacks_df['name_lookup'] = attacks_df['name'].str.strip()
                attack_info_map = attacks_df.set_index('name_lookup').to_dict('index')
            else:
                print(f"Warning: 'name' column not in {ATTACKS_LOOKUP_PATH}.")
        else:
            print(f"Warning: {ATTACKS_LOOKUP_PATH} not found. Attack descriptions will be generic.")
            
        print(f"Successfully loaded {len(feature_names)} features, model, scaler, and label encoder.")
        print(f"Loaded {len(attack_info_map)} attack definitions.")

    except FileNotFoundError as e:
        print(f"Fatal Error: {e}")
        print("Application cannot start without model artifacts. Exiting.")
        exit(1) # Exit if artifacts aren't found
    except Exception as e:
        print(f"Fatal Error loading artifacts: {e}")
        exit(1)

def get_attack_info(label: str) -> Dict[str, Any]:
    """Retrieves attack details from the loaded map."""
    # Find the label in our map
    info = attack_info_map.get(label, {})
    
    # Provide sensible defaults
    return {
        "name": info.get("name", label),
        "description": info.get("description", "No description available."),
        "severity": info.get("severity", "High" if label != "Benign" else "Normal"),
        "mitigation": info.get("mitigation", "Standard incident response procedures recommended.")
    }

async def get_ip_info(ip: str, client: httpx.AsyncClient) -> Dict[str, Any]:
    """Fetches geolocation data for an IP address."""
    if not ip or ip in ["localhost", "127.0.0.1"] or ip.startswith("192.168.") or ip.startswith("10."):
        return {"geo": "Internal/Local"}
    try:
        # Using a free, no-key-required API
        response = await client.get(f"https://ipapi.co/{ip}/json/", timeout=2.0)
        if response.status_code == 200:
            data = response.json()
            city = data.get('city', 'Unknown')
            country = data.get('country_name', 'Unknown')
            return {"geo": f"{city}, {country}", "lat": data.get('latitude'), "lon": data.get('longitude')}
        else:
            return {"geo": "Geo-lookup Failed"}
    except Exception as e:
        # print(f"IP info lookup failed for {ip}: {e}")
        return {"geo": "N/A"}

def create_error_entry(error_msg: str, index: int) -> Dict[str, Any]:
    """Creates a standardized error object for the frontend."""
    return {
        "id": hashlib.md5(f"error{index}{datetime.now(timezone.utc).isoformat()}".encode()).hexdigest()[:8],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ip": "N/A",
        "port": "N/A",
        "type": "Processing Error",
        "severity": "Critical",
        "confidence": "N/A",
        "source": "System",
        "geo": "N/A",
        "raw_data": {"error": error_msg}
    }

async def process_and_predict(df: pd.DataFrame, source: str) -> List[Dict[str, Any]]:
    """
    The core analysis pipeline.
    1. Cleans and aligns input data.
    2. Scales data.
    3. Predicts with XGBoost.
    4. Applies 80% confidence logic.
    5. Enriches with metadata.
    """
    start_time = datetime.now()
    processed_data = []
    
    if df.empty:
        return []

    # --- 1. Feature Alignment and Sanitization ---
    # Sanitize input columns
    df.columns = [sanitize_col_name(col) for col in df.columns]
    
    # Align columns with the model's features
    # Drop columns from input that are not in feature_names
    for col in df.columns:
        if col not in feature_names:
            df = df.drop(columns=[col])
            
    # Add missing columns (from feature_names) with default value 0
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
            
    # Ensure column order matches training
    try:
        aligned_df = df[feature_names]
    except KeyError as e:
        print(f"Fatal column mismatch: {e}")
        return [create_error_entry(f"Input data is missing critical features: {e}", 0)]

    # --- 2. Scale Data ---
    try:
        scaled_df = scaler.transform(aligned_df)
    except Exception as e:
        print(f"Error during scaling: {e}")
        return [create_error_entry(f"Data scaling error: {e}", i) for i in range(len(df))]

    # --- 3. Get Predictions & Probabilities (MODIFIED) ---
    try:
        predictions_encoded = model.predict(scaled_df)
        probabilities = model.predict_proba(scaled_df)
        predictions_str = le.inverse_transform(predictions_encoded)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return [create_error_entry(f"Prediction error: {e}", i) for i in range(len(df))]

    # --- 4. Process Each Row with New Logic ---
    async with httpx.AsyncClient() as client:
        for i, (index, row) in enumerate(df.iterrows()):
            try:
                prediction_label = predictions_str[i]
                proba_vector = probabilities[i]
                confidence = np.max(proba_vector) # Get the highest probability

                # Get base attack info from our map
                attack_info = get_attack_info(prediction_label)
                
                final_label = ""
                final_severity = ""

                # --- APPLYING USER'S 80% LOGIC ---
                if prediction_label == "Benign":
                    final_label = "Normal Network"
                    final_severity = "Normal"
                else:
                    # It's an attack prediction
                    if confidence > 0.80:
                        # High confidence attack
                        final_label = attack_info.get("name", prediction_label)
                        final_severity = attack_info.get("severity", "High")
                    else:
                        # Low confidence attack -> Medium Severity
                        final_label = f"Suspicious (Low Conf. {prediction_label})"
                        final_severity = "Medium"
                
                # --- 5. Enrich with Metadata ---
                # Try to find a valid IP field
                ip = str(row.get('source_ip', row.get('destination_ip', 'N/A')))
                port = str(row.get('destination_port', row.get('source_port', 'N/A')))
                
                ip_info = await get_ip_info(ip, client)
                
                threat_data = {
                    "id": hashlib.md5(f"{i}{source}{row.get('timestamp', '')}{port}".encode()).hexdigest()[:8],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "ip": ip,
                    "port": port,
                    "type": final_label,
                    "severity": final_severity,
                    "confidence": f"{confidence * 100:.2f}%", # Add confidence
                    "source": source,
                    "geo": ip_info.get("geo", "N/A"),
                    "lat": ip_info.get("lat"),
                    "lon": ip_info.get("lon"),
                    "raw_data": row.to_dict() # Add raw data for modal
                }
                
                processed_data.append(threat_data)
            
            except Exception as e:
                print(f"Error processing row {i}: {e}")
                processed_data.append(create_error_entry(f"Error processing row: {e}", i))

    end_time = datetime.now()
    print(f"Processed {len(df)} rows from {source} in {(end_time - start_time).total_seconds():.4f}s")
    
    # --- 6. Broadcast to Live Feed ---
    # We broadcast only high-severity threats to avoid clutter
    high_sev_threats = [t for t in processed_data if t["severity"] in ["High", "Critical"]]
    if high_sev_threats:
        # Run broadcast in background
        asyncio.create_task(broadcast_threats(high_sev_threats))

    return processed_data


# --- FastAPI Lifecycle Events ---
@app.on_event("startup")
def startup_event():
    """Loads artifacts on server startup."""
    load_artifacts()
    # To re-enable the demo feed, uncomment the line below
    # asyncio.create_task(simulate_threat_feed())

# --- API Endpoints ---
@app.get("/")
def read_root():
    """Root endpoint for health check."""
    return {"status": "CyberShield AI API is running."}

@app.post("/upload")
async def analyze_csv(file: UploadFile = File(...)):
    """Analyzes an uploaded CSV file."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .csv file.")
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Clean data: drop rows with NaN/Inf values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        
        results = await process_and_predict(df, source=file.filename)
        return JSONResponse(content=results)
        
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Error parsing CSV. Please check the file format.")
    except Exception as e:
        print(f"Error in /upload: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

@app.post("/analyze-manual")
async def analyze_manual_entry(payload: ManualInput):
    """Analyzes a single data entry provided as JSON."""
    try:
        # Convert the single JSON object into a DataFrame
        df = pd.DataFrame([payload.data])

        # Clean data: drop rows with NaN/Inf values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        
        results = await process_and_predict(df, source="Manual Input")
        # Return the first (and only) result
        if results:
            return JSONResponse(content=results[0])
        else:
            raise HTTPException(status_code=400, detail="Invalid manual input data.")
            
    except Exception as e:
        print(f"Error in /analyze-manual: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

# --- WebSocket Management ---
active_connections: List[WebSocket] = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handles persistent WebSocket connections for the live feed."""
    await websocket.accept()
    active_connections.append(websocket)
    print(f"Client connected. Total connections: {len(active_connections)}")
    try:
        while True:
            # Keep the connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        print("Client disconnected.")
        active_connections.remove(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)

async def broadcast_threats(threat_data_list: List[Dict[str, Any]]):
    """Pushes new threat data to all connected WebSocket clients."""
    if not active_connections:
        return
    
    message = json.dumps(threat_data_list)
    # Iterate over a copy to allow safe removals
    for connection in active_connections.copy():
        try:
            await connection.send_text(message)
        except Exception as e:
            print(f"Failed to send to client: {e}. Removing connection.")
            try:
                active_connections.remove(connection)
            except ValueError:
                pass


# ---------------- Demo: Simulated Threat Feed ----------------
async def simulate_threat_feed():
    """
    Simulates external threat feed and broadcasts to connected clients.
    NOTE: This is optional — to enable, uncomment the create_task call in startup_event above.
    """
    await asyncio.sleep(5)  # initial delay
    print("Starting threat simulation...")
    mock_threats = [
        {"ip": "185.12.33.9", "type": "DDoS", "severity": "Critical", "confidence": "99.8%", "lat": 48.8566, "lon": 2.3522},
        {"ip": "10.5.5.1", "type": "Port Scan", "severity": "Medium", "confidence": "75.2%", "lat": 34.0522, "lon": -118.2437},
        {"ip": "201.44.1.18", "type": "SQL Injection", "severity": "High", "confidence": "92.1%", "lat": -23.5505, "lon": -46.6333},
    ]

    rng = np.random.default_rng()
    while True:
        t = mock_threats[int(rng.integers(0, len(mock_threats)))]
        live_threat = {
            **t,
            "id": hashlib.md5(f"sim{datetime.now(timezone.utc).isoformat()}".encode()).hexdigest()[:8],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "port": str(rng.integers(1024, 65535)),
            "source": "Simulation",
            "geo": "Simulated Geo",
            "raw_data": {"simulated": True, "info": "This is a demo threat."}
        }
        await broadcast_threats([live_threat]) # Broadcast wants a list
        await asyncio.sleep(rng.uniform(3, 8)) # random delay

