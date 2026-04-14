import React, { useState, useRef } from 'react';
import { UploadCloud, Play, AlertCircle, RotateCcw, CheckCircle2 } from 'lucide-react';

function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    e.currentTarget.classList.add('dragover');
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      checkAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      checkAndSetFile(e.target.files[0]);
    }
  };

  const checkAndSetFile = (selectedFile) => {
    const validExtensions = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska'];
    // For simplicity, we also allow standard .mp4, .avi parsing
    if (selectedFile.type.startsWith('video/') || selectedFile.name.match(/\.(mp4|avi|mov|mkv)$/i)) {
      setFile(selectedFile);
      setError(null);
    } else {
      setError('Please select a valid video file (.mp4, .avi, .mov, .mkv)');
      setFile(null);
    }
  };

  const handleSubmit = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('video', file);

    try {
      // Use relative path to utilize the Vite proxy which handles mapping to Flask
      // (whether running natively or via host.docker.internal in Docker)
      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to analyze video');
      }

      setResult(data.data);
    } catch (err) {
      console.error(err);
      setError(err.message || 'An error occurred during analysis');
    } finally {
      setLoading(false);
    }
  };

  const resetAll = () => {
    setFile(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>Depression Detection Engine</h1>
        <p>Multimodal Fusion AI for Clinical Analysis</p>
      </header>

      <main>
        <div className="glass-card">
          {error && (
            <div className="error-message">
              <AlertCircle style={{display: 'inline', marginRight: '8px', verticalAlign: 'middle'}} size={20} />
              {error}
            </div>
          )}

          {!loading && !result && (
            <div className="upload-container">
              <div 
                className="file-drop-area"
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  onChange={handleFileChange}
                  accept="video/mp4,video/avi,video/quicktime,video/x-matroska"
                />
                
                {file ? (
                  <>
                    <CheckCircle2 className="upload-icon" style={{color: 'var(--success)'}} />
                    <h3>Video Selected</h3>
                    <p className="file-info">{file.name} ({(file.size / (1024 * 1024)).toFixed(2)} MB)</p>
                  </>
                ) : (
                  <>
                    <UploadCloud className="upload-icon" />
                    <h3>Drag & Drop Video Here</h3>
                    <p style={{color: 'var(--text-muted)', marginTop: '0.5rem'}}>or click to browse files</p>
                  </>
                )}
              </div>

              <button 
                className="btn" 
                onClick={handleSubmit} 
                disabled={!file}
              >
                <Play size={20} />
                Analyze Patient Video
              </button>
            </div>
          )}

          {loading && (
            <div className="loader-container">
              <div className="spinner"></div>
              <h2 className="loader-text">Analyzing Facial, Audio & Semantic Features...</h2>
              <p style={{color: 'var(--text-muted)', marginTop: '1rem'}}>
                Extracting OpenFace, Whisper, and Parselmouth features. This may take a few minutes.
              </p>
            </div>
          )}

          {result && (
            <div className="results-container">
              <div className="result-header">
                <h2>Diagnostic Overview</h2>
                <div className={`risk-badge risk-${result.depression_level}`}>
                  {result.depression_level.toUpperCase()} RISK
                </div>
              </div>

              <div className="stats-grid">
                <div className="stat-box">
                  <div className="stat-value">{(result.median_prob * 100).toFixed(1)}%</div>
                  <div className="stat-label">Baseline Probability</div>
                </div>
                <div className="stat-box">
                  <div className="stat-value">{(result.positive_pct * 100).toFixed(1)}%</div>
                  <div className="stat-label">Sequences &ge; 50%</div>
                </div>
                <div className="stat-box">
                  <div className="stat-value">{(result.high_pct * 100).toFixed(1)}%</div>
                  <div className="stat-label">High Risk Sequences</div>
                </div>
                <div className="stat-box">
                  <div className="stat-value">{result.mean_prob > 0.5 ? 'Positive' : 'Negative'}</div>
                  <div className="stat-label">Mean Prediction Class</div>
                </div>
              </div>
              
              <div style={{marginTop: '1rem'}}>
                <h3 style={{marginBottom: '1rem', borderBottom: '1px solid var(--glass-border)', paddingBottom: '0.5rem'}}>Segment Analysis</h3>
                <div style={{display: 'flex', flexDirection: 'column', gap: '0.5rem'}}>
                  <div style={{display: 'flex', justifyContent: 'space-between'}}>
                    <span>Severe Symptoms (&ge; 70%)</span>
                    <span>{result.counts['>= 0.70']} sequences</span>
                  </div>
                  <div style={{display: 'flex', justifyContent: 'space-between'}}>
                    <span>Moderate Alert (50% - 70%)</span>
                    <span>{result.counts['0.50 - 0.70']} sequences</span>
                  </div>
                  <div style={{display: 'flex', justifyContent: 'space-between'}}>
                    <span>Mild Indicators (30% - 50%)</span>
                    <span>{result.counts['0.30 - 0.50']} sequences</span>
                  </div>
                  <div style={{display: 'flex', justifyContent: 'space-between'}}>
                    <span>Normal Range (&lt; 30%)</span>
                    <span>{result.counts['< 0.30']} sequences</span>
                  </div>
                </div>
              </div>

              <div style={{display: 'flex', justifyContent: 'center', marginTop: '2rem'}}>
                <button className="btn" onClick={resetAll} style={{background: 'rgba(255,255,255,0.1)'}}>
                  <RotateCcw size={20} />
                  Analyze Another Patient
                </button>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
