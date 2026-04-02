# -------------------------------------------------
# setup.ps1 – Windows PowerShell version (NO OCTAVE)
# -------------------------------------------------

# 0. Enable TLS 1.2 for web requests
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# 1. Install system dependency (FFmpeg only)
Write-Host "Installing FFmpeg..."

try {
    # Use winget if available
    Write-Host "Attempting installation via winget..."
    winget install Gyan.FFmpeg --accept-package-agreements --accept-source-agreements
    Write-Host "FFmpeg installed via winget."
} catch {
    # Fallback to manual download
    Write-Host "Winget failed or not found. Falling back to direct download..."
    
    $ffmpegUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    $ffmpegZip = "$env:TEMP\ffmpeg.zip"

    Write-Host "Downloading FFmpeg..."
    Invoke-WebRequest -Uri $ffmpegUrl -OutFile $ffmpegZip

    Write-Host "Extracting FFmpeg..."
    Expand-Archive -Path $ffmpegZip -DestinationPath "$env:ProgramFiles\ffmpeg" -Force

    $ffmpegBin = "$env:ProgramFiles\ffmpeg\ffmpeg-*-essentials_build\bin"
    [Environment]::SetEnvironmentVariable(
        "Path",
        $env:Path + ";$ffmpegBin",
        [EnvironmentVariableTarget]::Machine
    )
}

# 2. Clone COVAREP repo (optional if you still need it)
if (-Not (Test-Path "./covarep")) {
    Write-Host "Cloning COVAREP repository..."
    git clone https://github.com/covarep/covarep.git
} else {
    Write-Host "COVAREP already present."
}

# 3. Pull OpenFace Docker image
Write-Host "Pulling OpenFace Docker image..."
docker pull algebr/openface:latest

# 4. Install Python dependencies (inside venv)
Write-Host "Installing Python packages..."
python -m pip install --upgrade "pip>=24,<25"
pip install praat-parselmouth openai-whisper ffmpeg-python pandas librosa

Write-Host "Setup complete! You can now run the Python pipelines."