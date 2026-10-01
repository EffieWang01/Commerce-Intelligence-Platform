$ErrorActionPreference = 'Stop'
$env:COMMERCE_PROFILE = 'local'
python -m pip install -e '.[test]'
python -m pytest -q
python -m commerce_ai.cli evaluate
Write-Host 'API:  uvicorn commerce_ai.api:app --port 8008'
Write-Host 'Demo: streamlit run app.py --server.port 8503'

