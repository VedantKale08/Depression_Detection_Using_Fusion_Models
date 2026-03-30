# -------------------------------------------------
# setup.ps1 – Windows PowerShell version of setup.sh
# -------------------------------------------------

# 0. Enable TLS 1.2 for web requests (Crucial to prevent Invoke-WebRequest errors)
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# 1. Install system dependencies (Octave & FFmpeg)
Write-Host "Installing Octave and FFmpeg..."

try {
    # Attempt winget first since it's cleaner
    Write-Host "Attempting installation via winget..."
    winget install GNU.Octave --accept-package-agreements --accept-source-agreements
    winget install Gyan.FFmpeg --accept-package-agreements --accept-source-agreements
    Write-Host "Octave and FFmpeg installed via winget."
} catch {
    # Fallback to direct download
    Write-Host "Winget failed or not found. Falling back to direct downloads..."
    
    $octaveUrl = "https://ftpmirror.gnu.org/octave/windows/octave-8.4.0-w64.msi"
    $octaveInstaller = "$env:TEMP\octave.msi"
    Write-Host "Downloading Octave from $octaveUrl..."
    Invoke-WebRequest -Uri $octaveUrl -OutFile $octaveInstaller
    Write-Host "Installing Octave..."
    Start-Process msiexec.exe -ArgumentList "/i `"$octaveInstaller`" /quiet /norestart" -Wait
    
    $ffmpegUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    $ffmpegZip = "$env:TEMP\ffmpeg.zip"
    Write-Host "Downloading FFmpeg from $ffmpegUrl..."
    Invoke-WebRequest -Uri $ffmpegUrl -OutFile $ffmpegZip
    Write-Host "Extracting FFmpeg..."
    Expand-Archive -Path $ffmpegZip -DestinationPath "$env:ProgramFiles\ffmpeg" -Force
    $ffmpegBin = "$env:ProgramFiles\ffmpeg\ffmpeg-*-essentials_build\bin"
    [Environment]::SetEnvironmentVariable("Path", $env:Path + ";$ffmpegBin", [EnvironmentVariableTarget]::Machine)
}

# 2. Clone COVAREP repo (Git for Windows must be installed)
if (-Not (Test-Path "./covarep")) {
    Write-Host "Cloning COVAREP repository..."
    git clone https://github.com/covarep/covarep.git
} else {
    Write-Host "COVAREP already present."
}

# 3. Pull OpenFace Docker image (Docker Desktop must be installed & running)
Write-Host "Pulling OpenFace Docker image..."
docker pull algebr/openface:latest

# 4. Install Python dependencies (use the venv you created earlier)
Write-Host "Installing Python packages..."
python -m pip install --upgrade pip
pip install praat-parselmouth openai-whisper ffmpeg-python pandas librosa

Write-Host "Setup complete! You can now run the Python pipelines."
