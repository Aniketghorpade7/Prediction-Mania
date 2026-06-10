$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================"
Write-Host "      Prediction Mania Installer"
Write-Host "========================================"
Write-Host ""

Set-Location $PSScriptRoot

# --------------------------------------------------
# Verify Python exists
# --------------------------------------------------

try {
    $pythonVersion = py -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
}
catch {
    Write-Host ""
    Write-Host "ERROR: Python is not installed."
    Write-Host "Install Python 3.10+ and try again."
    pause
    exit 1
}

$versionParts = $pythonVersion.Split('.')

if (
    [int]$versionParts[0] -lt 3 -or
    ([int]$versionParts[0] -eq 3 -and [int]$versionParts[1] -lt 10)
) {
    Write-Host ""
    Write-Host "ERROR: Python 3.10 or newer is required."
    Write-Host "Detected Python $pythonVersion"
    pause
    exit 1
}

Write-Host "Python $pythonVersion detected."
Write-Host ""

# --------------------------------------------------
# Create virtual environment
# --------------------------------------------------

if (!(Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    py -m venv .venv

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to create virtual environment."
        pause
        exit 1
    }
}
else {
    Write-Host "Using existing .venv"
}

Write-Host ""

# --------------------------------------------------
# Activate virtual environment
# --------------------------------------------------

& ".\.venv\Scripts\Activate.ps1"

# --------------------------------------------------
# Upgrade pip tools
# --------------------------------------------------

Write-Host "Upgrading pip..."

python -m pip install --upgrade `
    pip `
    setuptools `
    wheel

# --------------------------------------------------
# Detect NVIDIA GPU
# --------------------------------------------------

$HasNvidia = $false

try {
    $null = Get-Command nvidia-smi -ErrorAction Stop

    $gpuName = nvidia-smi --query-gpu=name --format=csv,noheader

    if ($gpuName) {
        $HasNvidia = $true
    }
}
catch {
    $HasNvidia = $false
}

# --------------------------------------------------
# Install PyTorch
# --------------------------------------------------

Write-Host ""
Write-Host "Installing PyTorch..."

if ($HasNvidia) {

    Write-Host ""
    Write-Host "NVIDIA GPU detected:"
    Write-Host $gpuName
    Write-Host ""
    Write-Host "Installing CUDA-enabled PyTorch..."
    Write-Host ""

    pip install `
        torch `
        torchvision `
        torchaudio `
        --index-url https://download.pytorch.org/whl/cu121
}
else {

    Write-Host ""
    Write-Host "No NVIDIA GPU detected."
    Write-Host "Installing CPU-only PyTorch..."
    Write-Host ""

    pip install `
        torch `
        torchvision `
        torchaudio `
        --index-url https://download.pytorch.org/whl/cpu
}

# --------------------------------------------------
# Install local TimeSFM package
# --------------------------------------------------

if (Test-Path ".\timesfm\pyproject.toml") {

    Write-Host ""
    Write-Host "Installing local TimeSFM package..."
    Write-Host ""

    pip install -e .\timesfm
}
else {

    Write-Host ""
    Write-Host "WARNING: timesfm folder not found."
}

# --------------------------------------------------
# Install dashboard requirements
# --------------------------------------------------

if (Test-Path ".\dashboard\requirements.txt") {

    Write-Host ""
    Write-Host "Installing dashboard requirements..."
    Write-Host ""

    pip install -r .\dashboard\requirements.txt
}
else {

    Write-Host ""
    Write-Host "WARNING: dashboard\requirements.txt not found."
}

# --------------------------------------------------
# Verify installation
# --------------------------------------------------

Write-Host ""
Write-Host "Verifying installation..."
Write-Host ""

python -c "import torch; print('Torch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available())"

# --------------------------------------------------
# Launch Streamlit
# --------------------------------------------------

Write-Host ""
Write-Host "Starting Prediction Mania..."
Write-Host ""

streamlit run .\dashboard\app.py

pause