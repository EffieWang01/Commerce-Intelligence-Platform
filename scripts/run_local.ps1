param(
    [ValidateSet('Auto', 'Rules', 'LambdaMART')]
    [string]$Ranker = 'Auto',
    [int]$ApiPort = 8008,
    [int]$UiPort = 8503
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = (Get-Command python -ErrorAction Stop).Source
$runtimeDir = Join-Path $projectRoot 'artifacts/runtime'
$modelPath = Join-Path $projectRoot 'artifacts/lambdamart/model.txt'
New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
$env:PYTHONPATH = (Join-Path $projectRoot 'src') + [IO.Path]::PathSeparator + $projectRoot
$env:COMMERCE_PROFILE = 'local'
$env:COMMERCE_DATA_DIR = Join-Path $projectRoot 'data/demo'
$env:COMMERCE_ARTIFACTS_DIR = Join-Path $projectRoot 'artifacts'
$env:OMP_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'

& $pythonPath -c 'import uvicorn, streamlit, commerce_ai'
if ($LASTEXITCODE -ne 0) { throw 'Missing dependencies. Run: python -m pip install -e ".[ranking,test]"' }
if ($Ranker -eq 'LambdaMART' -or ($Ranker -eq 'Auto' -and (Test-Path -LiteralPath $modelPath))) {
    if (-not (Test-Path -LiteralPath $modelPath)) { throw "Missing model: $modelPath" }
    $env:COMMERCE_RANKING_MODEL_PATH = $modelPath
    & $pythonPath -c 'import os; from pathlib import Path; from commerce_ai.ranking import load_ranking_model; load_ranking_model(Path(os.environ["COMMERCE_RANKING_MODEL_PATH"]))'
    if ($LASTEXITCODE -ne 0) { throw 'Cannot load LambdaMART. Check model and ranking dependencies.' }
    $expectedRanker = 'lightgbm-lambdarank'
} else {
    Remove-Item Env:COMMERCE_RANKING_MODEL_PATH -ErrorAction SilentlyContinue
    $expectedRanker = 'weighted-rules'
}

function Find-FreePort([int]$Preferred, [int]$Excluded = 0) {
    for ($port = $Preferred; $port -lt $Preferred + 100; $port++) {
        if ($port -eq $Excluded) { continue }
        $listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, $port)
        try { $listener.Start(); return $port }
        catch [Net.Sockets.SocketException] { continue }
        finally { $listener.Stop() }
    }
    throw 'No available port in the requested range.'
}

function Wait-Listener([int]$Port, $Process) {
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        $Process.Refresh()
        if ($Process.HasExited) { throw "Process exited; see $runtimeDir" }
        $client = [Net.Sockets.TcpClient]::new()
        try { $client.Connect('127.0.0.1', $Port); return }
        catch [Net.Sockets.SocketException] { Start-Sleep -Milliseconds 500 }
        finally { $client.Dispose() }
    }
    throw "Service did not listen on port $Port; see $runtimeDir"
}

$ApiPort = Find-FreePort $ApiPort
$UiPort = Find-FreePort $UiPort $ApiPort
$apiUrl = "http://127.0.0.1:$ApiPort"
$uiUrl = "http://127.0.0.1:$UiPort"
$env:COMMERCE_API_URL = $apiUrl
$apiProcess = $null
$uiProcess = $null
try {
    $apiProcess = Start-Process -FilePath $pythonPath -ArgumentList @('-m', 'uvicorn', 'commerce_ai.api:app', '--host', '127.0.0.1', '--port', $ApiPort) -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput "$runtimeDir/api-$ApiPort.log" -RedirectStandardError "$runtimeDir/api-$ApiPort.err.log"
    Wait-Listener $ApiPort $apiProcess
    $health = Invoke-RestMethod "$apiUrl/health" -TimeoutSec 180
    if ($health.providers.ranking -ne $expectedRanker) { throw "Unexpected ranking backend: $($health.providers.ranking)" }
    $uiProcess = Start-Process -FilePath $pythonPath -ArgumentList @('-m', 'streamlit', 'run', 'app.py', '--server.address', '127.0.0.1', '--server.port', $UiPort, '--server.headless', 'true', '--browser.gatherUsageStats', 'false') -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput "$runtimeDir/ui-$UiPort.log" -RedirectStandardError "$runtimeDir/ui-$UiPort.err.log"
    Wait-Listener $UiPort $uiProcess
    $null = Invoke-RestMethod "$uiUrl/_stcore/health" -TimeoutSec 10
} catch {
    foreach ($process in @($apiProcess, $uiProcess)) {
        if ($null -ne $process -and -not $process.HasExited) { Stop-Process -Id $process.Id }
    }
    throw
}
$state = @{ api_url = $apiUrl; ui_url = $uiUrl; api_pid = $apiProcess.Id; ui_pid = $uiProcess.Id; ranking = $health.providers.ranking; products = $health.catalog.products }
$state | ConvertTo-Json | Set-Content -LiteralPath "$runtimeDir/local-services.json" -Encoding utf8
Write-Host ($state | ConvertTo-Json)

