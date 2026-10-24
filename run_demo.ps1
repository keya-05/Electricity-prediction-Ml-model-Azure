# Starts the full local pipeline in three windows: edge module, smart-meter simulator, dashboard.
# Usage (from the project folder):  .\run_demo.ps1            (model runs locally at the edge)
#                                   .\run_demo.ps1 -Endpoint  (model runs on the Azure ML endpoint)
param([switch]$Endpoint)
$root = $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Error "Create the venv first: python -m venv .venv; .\.venv\Scripts\pip install -r requirements.txt"; exit 1 }
if (-not (Test-Path (Join-Path $root "energy_data.csv"))) { & $py (Join-Path $root "generate_dataset.py") }
if (-not (Test-Path (Join-Path $root "energy_model.pkl"))) { & $py (Join-Path $root "train_model.py") }
$predictor = if ($Endpoint) { "endpoint" } else { "local" }

Start-Process powershell -WorkingDirectory $root -ArgumentList "-NoExit", "-Command", "`$host.UI.RawUI.WindowTitle='Edge module'; & '$py' edge_module.py --predictor $predictor"
Start-Sleep -Seconds 3
Start-Process powershell -WorkingDirectory $root -ArgumentList "-NoExit", "-Command", "`$host.UI.RawUI.WindowTitle='Smart-meter simulator'; & '$py' energy_simulator.py"
Start-Process powershell -WorkingDirectory $root -ArgumentList "-NoExit", "-Command", "`$host.UI.RawUI.WindowTitle='Dashboard'; & '$py' -m streamlit run dashboard.py"
Write-Host "Started ($predictor predictor). Dashboard: http://localhost:8501  (close the three windows to stop)"
