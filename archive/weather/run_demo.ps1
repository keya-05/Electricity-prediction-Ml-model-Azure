# Starts the full local pipeline in three windows: edge module, simulator, dashboard.
# Usage (from the project folder):  .\run_demo.ps1
$root = $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Error "Create the venv first: python -m venv .venv; .\.venv\Scripts\pip install -r requirements.txt"; exit 1 }
if (-not (Test-Path (Join-Path $root "weather_model.pkl"))) { & $py (Join-Path $root "train_model.py") }

Start-Process powershell -WorkingDirectory $root -ArgumentList "-NoExit", "-Command", "`$host.UI.RawUI.WindowTitle='Edge module'; & '$py' edge_module.py"
Start-Sleep -Seconds 3
Start-Process powershell -WorkingDirectory $root -ArgumentList "-NoExit", "-Command", "`$host.UI.RawUI.WindowTitle='Weather simulator'; & '$py' weather_simulator.py"
Start-Process powershell -WorkingDirectory $root -ArgumentList "-NoExit", "-Command", "`$host.UI.RawUI.WindowTitle='Dashboard'; & '$py' -m streamlit run dashboard.py"
Write-Host "Started. Dashboard: http://localhost:8501  (close the three windows to stop)"
