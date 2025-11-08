"use client";

import React, { useState, useEffect, useRef } from 'react';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid,
  ResponsiveContainer, PieChart, Pie, Cell, AreaChart, Area
} from 'recharts';
import {
  Shield, Upload, Activity, Globe, AlertTriangle, AlertCircle,
  Server, Sliders, Database, Loader, XCircle, Terminal, ScanLine,
  Play, Pause, RefreshCw, Eye, EyeOff, BarChart3, Network,
  FileText, TrendingUp, Zap, CheckCircle, Home, ChevronRight, Clock,
  Cpu // --- NEW: Added icon for Model Details
} from 'lucide-react';

// --- API Configuration ---
// Make sure your FastAPI backend is running on this address
const API_BASE_URL = "http://localhost:8000"; 
// ---

// Add comprehensive styles
const styles: string = `
  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    color: #e2e8f0;
    overflow-x: hidden;
  }

  .dashboard-container {
    min-height: 100vh;
    background: linear-gradient(135deg, #020617 0%, #0f172a 50%, #1e293b 100%);
  }

  .sidebar {
    position: fixed;
    left: 0;
    top: 0;
    width: 256px;
    height: 100vh;
    background: rgba(15, 23, 42, 0.6);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(100, 116, 139, 0.3);
    z-index: 40;
    padding: 24px;
  }

  .sidebar-logo {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 32px;
  }

  .logo-icon {
    padding: 8px;
    background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%);
    border-radius: 12px;
  }

  .logo-text h1 {
    font-size: 20px;
    font-weight: 700;
    color: white;
  }

  .logo-text p {
    font-size: 12px;
    color: #94a3b8;
  }

  .nav-menu {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .nav-button {
    width: 100%;
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    border: none;
    border-radius: 12px;
    background: transparent;
    color: #94a3b8;
    cursor: pointer;
    transition: all 0.3s ease;
    font-size: 14px;
    font-weight: 500;
  }

  .nav-button:hover:not(:disabled) {
    background: rgba(51, 65, 85, 0.5);
    color: white;
  }

  .nav-button.active {
    background: linear-gradient(90deg, rgba(6, 182, 212, 0.2) 0%, rgba(59, 130, 246, 0.2) 100%);
    color: #06b6d4;
    border: 1px solid rgba(6, 182, 212, 0.3);
  }

  .nav-button:disabled {
    cursor: not-allowed;
    color: #475569;
  }

  .sidebar-status {
    position: absolute;
    bottom: 24px;
    left: 24px;
    right: 24px;
    background: rgba(30, 41, 59, 0.5);
    backdrop-filter: blur(10px);
    border-radius: 12px;
    padding: 16px;
    border: 1px solid rgba(71, 85, 105, 0.5);
  }

  .status-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
  }

  .status-dot {
    width: 8px;
    height: 8px;
    background: #4ade80;
    border-radius: 50%;
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  .main-content {
    margin-left: 256px;
    min-height: 100vh;
    padding: 32px;
  }

  .upload-section {
    max-width: 1200px;
    margin: 0 auto;
    padding: 48px 24px;
  }

  .hero-section {
    text-align: center;
    margin-bottom: 64px;
  }

  .hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 12px;
    padding: 8px 16px;
    background: rgba(6, 182, 212, 0.1);
    border: 1px solid rgba(6, 182, 212, 0.3);
    border-radius: 999px;
    margin-bottom: 24px;
  }

  .hero-title {
    font-size: 48px;
    font-weight: 700;
    margin-bottom: 24px;
    background: linear-gradient(90deg, #06b6d4 0%, #3b82f6 50%, #8b5cf6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  .hero-description {
    font-size: 20px;
    color: #94a3b8;
    max-width: 800px;
    margin: 0 auto;
    line-height: 1.6;
  }

  .upload-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 32px;
    margin-bottom: 32px;
  }

  .upload-card {
    background: rgba(15, 23, 42, 0.4);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(100, 116, 139, 0.3);
    border-radius: 24px;
    padding: 32px;
    transition: all 0.5s ease;
  }

  .upload-card:hover {
    border-color: rgba(6, 182, 212, 0.5);
    transform: translateY(-4px);
  }

  .card-header {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 24px;
  }

  .card-icon {
    padding: 16px;
    background: linear-gradient(135deg, rgba(6, 182, 212, 0.2) 0%, rgba(59, 130, 246, 0.2) 100%);
    border-radius: 16px;
  }

  .card-title h3 {
    font-size: 24px;
    font-weight: 700;
    color: white;
    margin-bottom: 4px;
  }

  .card-title p {
    color: #94a3b8;
    font-size: 14px;
  }

  .file-drop-zone {
    border: 2px dashed #475569;
    border-radius: 16px;
    padding: 48px 32px;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
    margin-bottom: 16px;
  }

  .file-drop-zone:hover {
    border-color: #06b6d4;
    background: rgba(6, 182, 212, 0.05);
  }

  .input-group {
    margin-bottom: 12px;
  }

  .input-label {
    display: block;
    font-size: 11px;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 4px;
    letter-spacing: 0.5px;
  }

  .input-field {
    width: 100%;
    padding: 12px 16px;
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid #475569;
    border-radius: 12px;
    color: white;
    font-size: 14px;
    transition: all 0.3s ease;
  }

  .input-field:focus {
    outline: none;
    border-color: #8b5cf6;
    background: rgba(30, 41, 59, 0.7);
  }

  .btn-primary {
    width: 100%;
    padding: 16px;
    background: linear-gradient(90deg, #06b6d4 0%, #3b82f6 100%);
    border: none;
    border-radius: 12px;
    color: white;
    font-weight: 600;
    font-size: 16px;
    cursor: pointer;
    transition: all 0.3s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
  }

  .btn-primary:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 10px 40px rgba(6, 182, 212, 0.4);
  }

  .btn-primary:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .btn-secondary {
    background: linear-gradient(90deg, #8b5cf6 0%, #ec4899 100%);
  }

  .metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 24px;
    margin-bottom: 32px;
  }

  .metric-card {
    background: rgba(15, 23, 42, 0.4);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(100, 116, 139, 0.3);
    border-radius: 16px;
    padding: 24px;
    transition: all 0.3s ease;
  }

  .metric-card:hover {
    border-color: rgba(6, 182, 212, 0.5);
    transform: translateY(-4px);
    box-shadow: 0 10px 30px rgba(6, 182, 212, 0.1);
  }

  .metric-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 16px;
  }

  .metric-icon {
    padding: 12px;
    background: linear-gradient(135deg, rgba(6, 182, 212, 0.2) 0%, rgba(59, 130, 246, 0.2) 100%);
    border-radius: 12px;
  }

  .metric-content h4 {
    font-size: 12px;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 8px;
    font-weight: 600;
    letter-spacing: 0.5px;
  }

  .metric-value {
    font-size: 32px;
    font-weight: 700;
    color: white;
  }
  
  .metric-value-small {
    font-size: 24px;
    font-weight: 700;
    color: white;
  }

  .chart-card {
    background: rgba(15, 23, 42, 0.4);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(100, 116, 139, 0.3);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 32px;
  }

  .chart-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 24px;
  }

  .chart-title {
    font-size: 18px;
    font-weight: 600;
    color: white;
  }

  .table-container {
    overflow-x: auto;
  }

  .data-table {
    width: 100%;
    border-collapse: collapse;
  }

  .data-table thead tr {
    border-bottom: 1px solid #334155;
  }

  .data-table th {
    text-align: left;
    padding: 12px 16px;
    color: #94a3b8;
    font-weight: 600;
    font-size: 14px;
  }

  .data-table tbody tr {
    border-bottom: 1px solid rgba(51, 65, 85, 0.5);
    transition: background 0.2s ease;
  }

  .data-table tbody tr:hover {
    background: rgba(51, 65, 85, 0.3);
    cursor: pointer;
  }

  .data-table td {
    padding: 12px 16px;
    color: #cbd5e1;
    font-size: 14px;
  }

  .badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
  }

  .badge-success {
    background: rgba(74, 222, 128, 0.2);
    color: #4ade80;
  }

  .badge-danger {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
  }

  .badge-warning {
    background: rgba(251, 191, 36, 0.2);
    color: #fbbf24;
  }

  .progress-bar {
    width: 80px;
    height: 8px;
    background: #334155;
    border-radius: 999px;
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #06b6d4 0%, #ef4444 100%);
    border-radius: 999px;
    transition: width 0.3s ease;
  }

  .alert {
    padding: 24px;
    border-radius: 16px;
    margin-bottom: 32px;
  }

  .alert-error {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
  }

  .alert-success {
    background: rgba(74, 222, 128, 0.1);
    border: 1px solid rgba(74, 222, 128, 0.3);
  }

  .grid-2 {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 32px;
  }

  .grid-3 {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 24px;
  }

  .modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.7);
    backdrop-filter: blur(10px);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }

  .modal-content {
    background: rgba(30, 41, 59, 0.9);
    border: 1px solid rgba(100, 116, 139, 0.3);
    border-radius: 16px;
    padding: 32px;
    width: 90%;
    max-width: 600px;
  }

  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 24px;
  }

  .modal-header h3 {
    font-size: 24px;
    color: white;
  }

  .modal-close-btn {
    background: transparent;
    border: none;
    color: #94a3b8;
    cursor: pointer;
  }

  .modal-close-btn:hover {
    color: white;
  }

  .modal-body .detail-grid {
    display: grid;
    grid-template-columns: 1fr 2fr;
    gap: 12px;
  }

  .modal-body .detail-grid > span:nth-child(odd) {
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    font-size: 12px;
  }

  .modal-body .detail-grid > span:nth-child(even) {
    color: #e2e8f0;
    font-family: monospace;
    font-size: 14px;
    word-break: break-all;
  }


  @media (max-width: 1024px) {
    .grid-2, .grid-3 {
      grid-template-columns: 1fr;
    }
    
    .sidebar {
      transform: translateX(-100%);
    }
    
    .main-content {
      margin-left: 0;
    }
  }

  @media (max-width: 600px) {
    .modal-body .detail-grid {
      grid-template-columns: 1fr;
    }
  }
`;

// Inject styles
if (typeof document !== 'undefined') {
  // Remove existing stylesheet if it exists
  const oldStyle = document.getElementById('dashboard-styles');
  if (oldStyle) {
    oldStyle.remove();
  }
  const styleSheet = document.createElement('style');
  styleSheet.id = 'dashboard-styles';
  styleSheet.textContent = styles;
  document.head.appendChild(styleSheet);
}

// Types
interface DashboardData {
  summary: {
    total_connections: number;
    attack_count: number;
    active_threats: number;
    attack_types_distribution: Array<{ name: string; value: number; risk_score: string }>;
    overall_health: number;
    traffic_series: Array<{ name: string; benign: number; attack: number }>;
  };
  predictions: Array<{
    id: string;
    label: string;
    probability: number;
    risk_score: number;
    severity: string;
    src_ip: string;
    dst_ip: string;
    dst_port: number;
    timestamp: string;
  }>;
  analysis_timestamp: string;
}

interface ManualInputs {
  src_ip: string;
  dst_ip: string;
  src_port: string;
  dst_port: string;
  protocol: string;
}

interface ScanResult {
  label: string;
  risk_score: number;
  severity: string;
  description: string;
}

// --- NEW: Type for Model Details ---
interface ModelDetails {
  accuracy: number;
  classification_report: {
    [key: string]: {
      precision: number;
      recall: number;
      'f1-score': number;
      support: number;
    };
  };
  classes: string[];
  training_timestamp: string;
}


// Add THREE.js types if they are available in window
declare const window: any;

const EMPTY_DATA: DashboardData = {
  summary: {
    total_connections: 0,
    attack_count: 0,
    active_threats: 0,
    attack_types_distribution: [],
    overall_health: 100,
    traffic_series: [],
  },
  predictions: [],
  analysis_timestamp: '',
};

const ATTACK_COLORS: string[] = ['#ef4444', '#f59e0b', '#06b6d4', '#8b5cf6', '#ec4899'];

// --- REMOVED loadScript utility ---

// Main Component
const CyberShieldDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('overview');
  const [data, setData] = useState<DashboardData>(EMPTY_DATA);
  const [isDataLoaded, setIsDataLoaded] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [file, setFile] = useState<File | null>(null);
  const [manualInputs, setManualInputs] = useState<ManualInputs>({
    src_ip: '172.16.1.50',
    dst_ip: '104.20.15.3',
    src_port: '15000',
    dst_port: '8080',
    protocol: 'tcp'
  });
  const [scanResult, setScanResult] = useState<ScanResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedLog, setSelectedLog] = useState<DashboardData['predictions'][0] | null>(null);
  
  // --- NEW: State for model details ---
  const [modelDetails, setModelDetails] = useState<ModelDetails | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  // --- REMOVED threeJsContainerRef ---

  // --- NEW: Fetch model details when app loads ---
  useEffect(() => {
    const fetchModelDetails = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/model-details`);
        if (!response.ok) {
          throw new Error("Could not load model details. Please train the model.");
        }
        const details: ModelDetails = await response.json();
        setModelDetails(details);
      } catch (err: any) {
        console.error(err.message);
        setError(err.message);
      }
    };
    fetchModelDetails();
  }, []); // Runs once on component mount

  const handleFileUpload = async (): Promise<void> => {
    if (!file) return;

    setIsLoading(true);
    setError(null);
    setScanResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_BASE_URL}/api/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Analysis failed. Please try again.");
      }

      const result: DashboardData = await response.json();

      setData(result);
      setIsDataLoaded(true);
      setError(null);
      setActiveTab('overview'); // Switch to overview after load
    } catch (err: any) {
      setError(err.message || "An unknown error occurred during analysis.");
      setIsDataLoaded(false);
    } finally {
      setIsLoading(false);
    }
  };

  const handleManualScan = async (): Promise<void> => {
    setIsLoading(true);
    setError(null);
    setScanResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/scan-single`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(manualInputs),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Manual scan failed.");
      }

      const result: ScanResult = await response.json();
      setScanResult(result);
      setError(null);

    } catch (err: any) {
      setError(err.message || "Manual scan failed.");
      setScanResult(null);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = (): void => {
    setData(EMPTY_DATA);
    setIsDataLoaded(false); 
    setFile(null);
    setScanResult(null);
    setError(null);
    setActiveTab('overview'); // Reset tab to default
    setSelectedLog(null); // Close modal
    
    // Clear file input
    if (fileInputRef.current) {
        fileInputRef.current.value = "";
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>): void => {
    setFile(e.target.files?.[0] || null);
    setError(null);
  };

  const handleManualInputChange = (e: React.ChangeEvent<HTMLInputElement>, key: keyof ManualInputs): void => {
    setManualInputs(prev => ({ ...prev, [key]: e.target.value }));
  };

  // --- REMOVED Live Feed and Threat Map useEffect hooks ---

  const renderUpload = (): JSX.Element => (
    <div className="upload-section">
      <div className="hero-section">
        <div className="hero-badge">
          <div className="status-dot" />
          <span style={{ color: '#06b6d4', fontSize: '14px', fontWeight: 600 }}>
            AI-Powered Security Analysis
          </span>
        </div>
        <h1 className="hero-title">Advanced Threat Detection</h1>
        <p className="hero-description">
          Upload network traffic data or perform real-time analysis to identify security threats with cutting-edge AI technology
        </p>
      </div>

      <div className="upload-grid">
        <div className="upload-card">
          <div className="card-header">
            <div className="card-icon">
              <Database style={{ width: 32, height: 32, color: '#06b6d4' }} />
            </div>
            <div className="card-title">
              <h3>Batch Analysis</h3>
              <p>Upload CSV for comprehensive scan</p>
            </div>
          </div>

          <div className="file-drop-zone" onClick={() => fileInputRef.current?.click()}>
            <Upload style={{ width: 48, height: 48, color: '#475569', margin: '0 auto 16px' }} />
            <p style={{ color: '#94a3b8', marginBottom: '8px' }}>
              {file ? file.name : 'Click to upload or drag and drop'}
            </p>
            <p style={{ fontSize: '12px', color: '#64748b' }}>CSV files only (Max 50MB)</p>
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              style={{ display: 'none' }}
            />
          </div>

          <button onClick={handleFileUpload} disabled={!file || isLoading} className="btn-primary">
            {isLoading ? (
              <>
                <Loader style={{ width: 20, height: 20, animation: 'spin 1s linear infinite' }} />
                Analyzing...
              </>
            ) : (
              <>
                <ScanLine style={{ width: 20, height: 20 }} />
                Start Analysis
              </>
            )}
          </button>
        </div>

        <div className="upload-card">
          <div className="card-header">
            <div className="card-icon" style={{ background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.2) 0%, rgba(236, 72, 153, 0.2) 100%)' }}>
              <Sliders style={{ width: 32, height: 32, color: '#8b5cf6' }} />
            </div>
            <div className="card-title">
              <h3>Manual Scan</h3>
              <p>Real-time connection analysis</p>
            </div>
          </div>

          {(Object.keys(manualInputs) as Array<keyof ManualInputs>).map((key) => (
            <div key={key} className="input-group">
              <label className="input-label">{key.replace('_', ' ')}</label>
              <input
                type="text"
                value={manualInputs[key]}
                onChange={(e) => handleManualInputChange(e, key)}
                className="input-field"
                placeholder={key === 'src_ip' ? 'e.g., 192.168.1.10' : ''}
              />
            </div>
          ))}

          <button onClick={handleManualScan} disabled={isLoading} className="btn-primary btn-secondary">
            {isLoading ? (
              <>
                <Loader style={{ width: 20, height: 20, animation: 'spin 1s linear infinite' }} />
                Scanning...
              </>
            ) : (
              <>
                <Zap style={{ width: 20, height: 20 }} />
                Scan Connection
              </>
            )}
          </button>
        </div>
      </div>

      {/* --- Display Scan Result --- */}
      {scanResult && (
        <div className={`alert ${scanResult.label === 'Benign' ? 'alert-success' : 'alert-error'}`}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
            <CheckCircle style={{ width: 24, height: 24, color: scanResult.label === 'Benign' ? '#4ade80' : '#ef4444' }} />
            <h3 style={{ fontSize: '18px', fontWeight: 600, color: 'white' }}>Scan Complete</h3>
          </div>
          <div className="grid-2" style={{ gap: '16px' }}>
            <div>
              <p style={{ color: '#94a3b8', fontSize: '14px' }}>Threat Type</p>
              <p style={{ fontSize: '18px', fontWeight: 700, color: scanResult.label === 'Benign' ? '#4ade80' : '#ef4444' }}>
                {scanResult.label}
              </p>
            </div>
            <div>
              <p style={{ color: '#94a3b8', fontSize: '14px' }}>Risk Score</p>
              <p style={{ fontSize: '18px', fontWeight: 700, color: '#f59e0b' }}>{scanResult.risk_score}</p>
            </div>
          </div>
          <p style={{ color: '#cbd5e1', fontSize: '14px', marginTop: '16px' }}>
            {scanResult.description}
          </p>
        </div>
      )}

      {/* --- Display Main Error --- */}
      {error && (
          <div className="alert alert-error">
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
              <XCircle style={{ width: 24, height: 24, color: '#ef4444' }} />
              <h3 style={{ fontSize: '18px', fontWeight: 600, color: 'white' }}>Analysis Failed</h3>
            </div>
            <p style={{ color: '#cbd5e1', fontSize: '14px', marginTop: '16px', fontFamily: 'monospace' }}>
              {error}
            </p>
          </div>
        )}
    </div>
  );

  const renderDashboard = (): JSX.Element => (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '32px' }}>
        <div>
          <h2 style={{ fontSize: '32px', fontWeight: 700, color: 'white', marginBottom: '8px' }}>
            Security Overview
          </h2>
          <p style={{ color: '#94a3b8' }}>
            Last updated: {new Date(data.analysis_timestamp).toLocaleString()}
          </p>
        </div>
        <button onClick={handleReset} className="btn-primary" style={{ width: 'auto', padding: '12px 24px', background: 'linear-gradient(90deg, #ef4444 0%, #dc2626 100%)' }}>
          <RefreshCw style={{ width: 20, height: 20 }} />
          New Analysis
        </button>
      </div>

      <div className="metric-grid">
        <div className="metric-card">
          <div className="metric-header">
            <div className="metric-icon">
              <Shield style={{ width: 24, height: 24, color: '#8b5cf6' }} />
            </div>
          </div>
          <div className="metric-content">
            <h4>System Health</h4>
            <div className="metric-value">{data.summary.overall_health.toFixed(1)}%</div>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-header">
            <div className="metric-icon">
              <Network style={{ width: 24, height: 24, color: '#06b6d4' }} />
            </div>
          </div>
          <div className="metric-content">
            <h4>Total Connections</h4>
            <div className="metric-value">{data.summary.total_connections.toLocaleString()}</div>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-header">
            <div className="metric-icon">
              <AlertTriangle style={{ width: 24, height: 24, color: '#ef4444' }} />
            </div>
          </div>
          <div className="metric-content">
            <h4>Attack Detections</h4>
            <div className="metric-value">{data.summary.attack_count.toLocaleString()}</div>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-header">
            <div className="metric-icon">
              <Zap style={{ width: 24, height: 24, color: '#f59e0b' }} />
            </div>
          </div>
          <div className="metric-content">
            <h4>Active Threats</h4>
            <div className="metric-value">{data.summary.active_threats.toLocaleString()}</div>
          </div>
        </div>
      </div>

      <div>
        <div className="chart-card">
          <div className="chart-header">
            <BarChart3 style={{ width: 20, height: 20, color: '#06b6d4' }} />
            <h3 className="chart-title">Traffic Analysis</h3>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={data.summary.traffic_series}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="name" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(30, 41, 59, 0.8)',
                  borderColor: '#475569',
                  borderRadius: '12px'
                }}
                labelStyle={{ color: 'white' }}
                itemStyle={{ color: '#e2e8f0' }}
              />
              <Area type="monotone" dataKey="benign" stackId="1" stroke="#06b6d4" fill="#06b6d4" fillOpacity={0.6} />
              <Area type="monotone" dataKey="attack" stackId="1" stroke="#ef4444" fill="#ef4444" fillOpacity={0.6} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="chart-card">
        <div className="chart-header">
          <Server style={{ width: 20, height: 20, color: '#06b6d4' }} />
          <h3 className="chart-title">Recent Detections (All Traffic)</h3>
        </div>
        <div className="table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Threat</th>
                <th>Source IP</th>
                <th>Destination</th>
                <th>Risk Score</th>
                <th>Severity</th>
              </tr>
            </thead>
            <tbody>
              {data.predictions.slice(0, 10).map((pred) => (
                <tr key={pred.id} onClick={() => setSelectedLog(pred)}>
                  <td>
                    <span className={pred.label.toLowerCase() === 'benign' ? 'badge badge-success' : 'badge badge-danger'}>
                      {pred.label}
                    </span>
                  </td>
                  <td style={{ fontFamily: 'monospace' }}>{pred.src_ip}</td>
                  <td style={{ fontFamily: 'monospace' }}>{pred.dst_ip}:{pred.dst_port}</td>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <div className="progress-bar">
                        <div className="progress-fill" style={{ width: `${pred.risk_score}%` }} />
                      </div>
                      <span>{pred.risk_score.toFixed(0)}</span>
                    </div>
                  </td>
                  <td>
                    <span className={`badge ${
                        pred.severity === 'Critical' ? 'badge-danger' :
                        pred.severity === 'High' ? 'badge-danger' :
                        pred.severity === 'Medium' ? 'badge-warning' :
                        'badge-success' // Benign and Low will be green
                      }`}>
                      {pred.severity}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );

  // --- New Render Functions ---

  const renderAnalytics = (): JSX.Element => (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '32px' }}>
        <h2 style={{ fontSize: '32px', fontWeight: 700, color: 'white', marginBottom: '8px' }}>
          Detailed Analytics
        </h2>
      </div>
      
      <div className="grid-2">
        <div className="chart-card">
          <div className="chart-header">
            <BarChart3 style={{ width: 20, height: 20, color: '#06b6d4' }} />
            <h3 className="chart-title">Traffic Analysis (Benign vs. Attack)</h3>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={data.summary.traffic_series}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="name" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(30, 41, 59, 0.8)',
                  borderColor: '#475569',
                  borderRadius: '12px'
                }}
                labelStyle={{ color: 'white' }}
                itemStyle={{ color: '#e2e8f0' }}
              />
              <Area type="monotone" dataKey="benign" stackId="1" stroke="#06b6d4" fill="#06b6d4" fillOpacity={0.6} />
              <Area type="monotone" dataKey="attack" stackId="1" stroke="#ef4444" fill="#ef4444" fillOpacity={0.6} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card">
          <div className="chart-header">
            <Activity style={{ width: 20, height: 20, color: '#8b5cf6' }} />
            <h3 className="chart-title">Attack Type Distribution</h3>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={data.summary.attack_types_distribution}
                cx="50%"
                cy="50%"
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
                label={({ name, percent }: { name: string, percent: number }) => `${name} ${(percent * 100).toFixed(0)}%`}
              >
                {data.summary.attack_types_distribution.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={ATTACK_COLORS[index % ATTACK_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(30, 41, 59, 0.8)',
                  borderColor: '#475569',
                  borderRadius: '12px'
                }}
                labelStyle={{ color: 'white' }}
                itemStyle={{ color: '#e2e8f0' }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="chart-card">
          <div className="chart-header">
            <TrendingUp style={{ width: 20, height: 20, color: '#f59e0b' }} />
            <h3 className="chart-title">Risk Score Over Time (Sampled)</h3>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={data.predictions.slice(0, 15).map(p => ({ name: p.timestamp ? p.timestamp.slice(11, 16) : 'N/A', risk: p.risk_score })).reverse()}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="name" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" domain={[0, 100]}/>
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(30, 41, 59, 0.8)',
                  borderColor: '#475569',
                  borderRadius: '12px'
                }}
                labelStyle={{ color: 'white' }}
                itemStyle={{ color: '#e2e8f0' }}
              />
              <Line type="monotone" dataKey="risk" stroke="#f59e0b" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
    </div>
  );

  // --- REMOVED renderThreatMap ---

  const renderEventLogs = (): JSX.Element => (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '32px' }}>
        <h2 style={{ fontSize: '32px', fontWeight: 700, color: 'white', marginBottom: '8px' }}>
          Event Logs (All Traffic)
        </h2>
      </div>
      <div className="chart-card">
        <div className="chart-header">
          <FileText style={{ width: 20, height: 20, color: '#06b6d4' }} />
          <h3 className="chart-title">All Detections ({data.predictions.length} items)</h3>
        </div>
        <div className="table-container" style={{ maxHeight: '600px', overflowY: 'auto' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>Threat</th>
                <th>Source IP</th>
                <th>Destination</th>
                <th>Risk Score</th>
                <th>Severity</th>
              </tr>
            </thead>
            <tbody>
              {data.predictions.map((pred) => (
                <tr key={pred.id} onClick={() => setSelectedLog(pred)}>
                  <td>{pred.timestamp ? new Date(pred.timestamp).toLocaleTimeString() : 'N/A'}</td>
                  <td>
                    <span className={pred.label.toLowerCase() === 'benign' ? 'badge badge-success' : 'badge badge-danger'}>
                      {pred.label}
                    </span>
                  </td>
                  <td style={{ fontFamily: 'monospace' }}>{pred.src_ip || 'N/A'}</td>
                  <td style={{ fontFamily: 'monospace' }}>{pred.dst_ip || 'N/A'}:{pred.dst_port}</td>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <div className="progress-bar">
                        <div className="progress-fill" style={{ width: `${pred.risk_score}%` }} />
                      </div>
                      <span>{pred.risk_score.toFixed(0)}</span>
                    </div>
                  </td>
                  <td>
                    <span className={`badge ${
                        pred.severity === 'Critical' ? 'badge-danger' :
                        pred.severity === 'High' ? 'badge-danger' :
                        pred.severity === 'Medium' ? 'badge-warning' :
                        'badge-success' // Benign and Low will be green
                      }`}>
                      {pred.severity}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );

  // ---
  // --- ADDED New Feature: Threat Log
  // ---
  const renderThreatLog = (): JSX.Element => {
    // Filter predictions to only show attacks
    const threats = data.predictions.filter(pred => pred.label.toLowerCase() !== 'benign');
    
    return (
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '32px' }}>
          <h2 style={{ fontSize: '32px', fontWeight: 700, color: 'white', marginBottom: '8px' }}>
            Threat Log (Attacks Only)
          </h2>
        </div>
        <div className="chart-card">
          <div className="chart-header">
            <AlertTriangle style={{ width: 20, height: 20, color: '#ef4444' }} />
            <h3 className="chart-title">Detected Threats ({threats.length} items)</h3>
          </div>
          <div className="table-container" style={{ maxHeight: '600px', overflowY: 'auto' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Threat</th>
                  <th>Source IP</th>
                  <th>Destination</th>
                  <th>Risk Score</th>
                  <th>Severity</th>
                </tr>
              </thead>
              <tbody>
                {threats.length === 0 && (
                  <tr>
                    <td colSpan={6} style={{ textAlign: 'center', color: '#94a3b8', padding: '32px' }}>
                      No threats detected in this file.
                    </td>
                  </tr>
                )}
                {threats.map((pred) => (
                  <tr key={pred.id} onClick={() => setSelectedLog(pred)}>
                    <td>{pred.timestamp ? new Date(pred.timestamp).toLocaleTimeString() : 'N/A'}</td>
                    <td>
                      <span className={pred.label.toLowerCase() === 'benign' ? 'badge badge-success' : 'badge badge-danger'}>
                        {pred.label}
                      </span>
                    </td>
                    <td style={{ fontFamily: 'monospace' }}>{pred.src_ip || 'N/A'}</td>
                    <td style={{ fontFamily: 'monospace' }}>{pred.dst_ip || 'N/A'}:{pred.dst_port}</td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <div className="progress-bar">
                          <div className="progress-fill" style={{ width: `${pred.risk_score}%` }} />
                        </div>
                        <span>{pred.risk_score.toFixed(0)}</span>
                      </div>
                    </td>
                    <td>
                      <span className={`badge ${
                          pred.severity === 'Critical' ? 'badge-danger' :
                          pred.severity === 'High' ? 'badge-danger' :
                          pred.severity === 'Medium' ? 'badge-warning' :
                          'badge-success' // Benign and Low will be green
                        }`}>
                        {pred.severity}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    );
  };

  // ---
  // --- NEW: Render function for Model Details
  // ---
  const renderModelDetails = (): JSX.Element => {
    if (!modelDetails) {
      return (
        <div>
          <h2 style={{ fontSize: '32px', fontWeight: 700, color: 'white', marginBottom: '32px' }}>
            Model Training Details
          </h2>
          <div className="chart-card">
            <p style={{ color: '#94a3b8', textAlign: 'center' }}>Loading model details or metrics file not found...</p>
          </div>
        </div>
      );
    }
    
    const report = modelDetails.classification_report;
    const classes = Object.keys(report).filter(key => !key.includes('accuracy') && key !== 'macro avg' && key !== 'weighted avg');

    return (
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '32px' }}>
          <h2 style={{ fontSize: '32px', fontWeight: 700, color: 'white', marginBottom: '8px' }}>
            Model Training Details
          </h2>
          <p style={{ color: '#94a3b8' }}>
            Last Trained: {new Date(modelDetails.training_timestamp).toLocaleString()}
          </p>
        </div>

        <div className="metric-grid">
          <div className="metric-card">
            <div className="metric-header">
              <div className="metric-icon">
                <CheckCircle style={{ width: 24, height: 24, color: '#4ade80' }} />
              </div>
            </div>
            <div className="metric-content">
              <h4>Overall Accuracy</h4>
              <div className="metric-value">{(modelDetails.accuracy * 100).toFixed(2)}%</div>
            </div>
          </div>
          <div className="metric-card">
            <div className="metric-header">
              <div className="metric-icon">
                <Zap style={{ width: 24, height: 24, color: '#f59e0b' }} />
              </div>
            </div>
            <div className="metric-content">
              <h4>Macro Avg F1-Score</h4>
              <div className="metric-value-small">{(report['macro avg']['f1-score']).toFixed(3)}</div>
            </div>
          </div>
          <div className="metric-card">
            <div className="metric-header">
              <div className="metric-icon">
                <Sliders style={{ width: 24, height: 24, color: '#8b5cf6' }} />
              </div>
            </div>
            <div className="metric-content">
              <h4>Weighted Avg F1-Score</h4>
              <div className="metric-value-small">{(report['weighted avg']['f1-score']).toFixed(3)}</div>
            </div>
          </div>
        </div>

        <div className="chart-card">
          <div className="chart-header">
            <Cpu style={{ width: 20, height: 20, color: '#06b6d4' }} />
            <h3 className="chart-title">Classification Report (Per-Class Performance)</h3>
          </div>
          <div className="table-container">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Class (Attack Type)</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1-Score</th>
                  <th>Support (Test Samples)</th>
                </tr>
              </thead>
              <tbody>
                {classes.map(className => (
                  <tr key={className}>
                    <td style={{ fontWeight: 600 }}>{className}</td>
                    <td>{report[className].precision.toFixed(3)}</td>
                    <td>{report[className].recall.toFixed(3)}</td>
                    <td>{report[className]['f1-score'].toFixed(3)}</td>
                    <td>{report[className].support.toLocaleString()}</td>
                  </tr>
                ))}
                <tr style={{ background: 'rgba(51, 65, 85, 0.3)', fontWeight: 'bold' }}>
                  <td>Macro Avg</td>
                  <td>{report['macro avg'].precision.toFixed(3)}</td>
                  <td>{report['macro avg'].recall.toFixed(3)}</td>
                  <td>{report['macro avg']['f1-score'].toFixed(3)}</td>
                  <td>{report['macro avg'].support.toLocaleString()}</td>
                </tr>
                <tr style={{ background: 'rgba(51, 65, 85, 0.3)', fontWeight: 'bold' }}>
                  <td>Weighted Avg</td>
                  <td>{report['weighted avg'].precision.toFixed(3)}</td>
                  <td>{report['weighted avg'].recall.toFixed(3)}</td>
                  <td>{report['weighted avg']['f1-score'].toFixed(3)}</td>
                  <td>{report['weighted avg'].support.toLocaleString()}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    );
  };

  // --- REMOVED renderLiveFeed ---

  // --- Modal Renderer ---
  const renderLogDetailModal = (): JSX.Element | null => {
    if (!selectedLog) return null;

    return (
      <div className="modal-overlay" onClick={() => setSelectedLog(null)}>
        <div className="modal-content" onClick={(e) => e.stopPropagation()}>
          <div className="modal-header">
            <h3>Detection Details</h3>
            <button className="modal-close-btn" onClick={() => setSelectedLog(null)}>
              <XCircle style={{ width: 24, height: 24 }} />
            </button>
          </div>
          <div className="modal-body">
            <div className="detail-grid">
              <span>Threat ID</span>
              <span>{selectedLog.id}</span>
              
              <span>Timestamp</span>
              <span>{selectedLog.timestamp ? new Date(selectedLog.timestamp).toLocaleString() : 'N/A'}</span>
              
              <span>Threat Label</span>
              <span>{selectedLog.label}</span>
              
              <span>Severity</span>
              <span>{selectedLog.severity}</span>
              
              <span>Risk Score</span>
              <span>{selectedLog.risk_score.toFixed(2)}</span>
              
              <span>Probability</span>
              <span>{selectedLog.probability.toFixed(2)}%</span>
              
              <span>Source IP</span>
              <span>{selectedLog.src_ip || 'N/A'}</span>
              
              <span>Destination IP</span>
              <span>{selectedLog.dst_ip || 'N/A'}</span>
              
              <span>Destination Port</span>
              <span>{selectedLog.dst_port}</span>
            </div>
          </div>
        </div>
      </div>
    );
  };

  // --- Main Content Renderer ---
  const renderMainContent = (): JSX.Element => {
    switch (activeTab) {
      case 'overview':
        return renderDashboard();
      case 'analytics':
        return renderAnalytics();
      // --- ADDED 'threat-log' case ---
      case 'threat-log':
        return renderThreatLog();
      case 'logs':
        return renderEventLogs();
      // --- ADDED 'model-details' case ---
      case 'model-details':
        return renderModelDetails();
      // --- REMOVED 'threats' and 'live-feed' ---
      default:
        return renderDashboard();
    }
  };

// File: CyberShieldDashboard.tsx
  return (
    // --- FIXED: Added suppressHydrationWarning to the main div ---
    <div className="dashboard-container" suppressHydrationWarning={true}>
      <div className="sidebar">
        <div className="sidebar-logo">
          <div className="logo-icon">
            <Shield style={{ width: 24, height: 24, color: 'white' }} />
          </div>
          <div className="logo-text">
            <h1>CyberShield</h1>
            <p>AI Security</p>
          </div>
        </div>

        <nav className="nav-menu">
          <button
            className={`nav-button ${activeTab === 'overview' ? 'active' : ''}`}
            onClick={() => setActiveTab('overview')}
            disabled={!isDataLoaded}
          >
            <Home style={{ width: 20, height: 20 }} />
            <span>Overview</span>
            {activeTab === 'overview' && <ChevronRight style={{ width: 16, height: 16, marginLeft: 'auto' }} />}
          </button>

          <button
            className={`nav-button ${activeTab === 'analytics' ? 'active' : ''}`}
            onClick={() => setActiveTab('analytics')}
            disabled={!isDataLoaded}
          >
            <BarChart3 style={{ width: 20, height: 20 }} />
            <span>Analytics</span>
            {activeTab === 'analytics' && <ChevronRight style={{ width: 16, height: 16, marginLeft: 'auto' }} />}
          </button>

          {/* ---
          --- ADDED New "Threat Log" Button
          --- */}
          <button
            className={`nav-button ${activeTab === 'threat-log' ? 'active' : ''}`}
            onClick={() => setActiveTab('threat-log')}
            disabled={!isDataLoaded}
          >
            <AlertTriangle style={{ width: 20, height: 20 }} />
            <span>Threat Log</span>
            {activeTab === 'threat-log' && <ChevronRight style={{ width: 16, height: 16, marginLeft: 'auto' }} />}
          </button>

          <button
            className={`nav-button ${activeTab === 'logs' ? 'active' : ''}`}
            onClick={() => setActiveTab('logs')}
            disabled={!isDataLoaded}
          >
            <FileText style={{ width: 20, height: 20 }} />
            <span>Event Logs (All)</span>
            {activeTab === 'logs' && <ChevronRight style={{ width: 16, height: 16, marginLeft: 'auto' }} />}
          </button>

          {/* ---
          --- ADDED New "Model Details" Button
          --- */}
          <button
            className={`nav-button ${activeTab === 'model-details' ? 'active' : ''}`}
            onClick={() => setActiveTab('model-details')}
          >
            <Cpu style={{ width: 20, height: 20 }} />
            <span>Model Details</span>
            {activeTab === 'model-details' && <ChevronRight style={{ width: 16, height: 16, marginLeft: 'auto' }} />}
          </button>

          {/* ---
          --- REMOVED "Threat Map" and "Live Feed" buttons
          --- */}

        </nav>

        <div className="sidebar-status">
          <div className="status-header">
            <div className="status-dot" />
            <span style={{ fontSize: '14px', color: '#cbd5e1', fontWeight: 500 }}>System Status</span>
          </div>
          <p style={{ fontSize: '12px', color: '#64748b' }}>All systems operational</p>
        </div>
      </div>

      <div className="main-content">
        {/* --- UPDATED: Model Details tab is always available --- */}
        {!isDataLoaded && activeTab !== 'model-details' ? renderUpload() : renderMainContent()}
      </div>
      
      {renderLogDetailModal()}
    </div>
  );
};

export default CyberShieldDashboard;