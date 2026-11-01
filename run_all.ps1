# Full pipeline for Windows PowerShell. Quick smoke test:  .\run_all.ps1 --trees 300 --jobs -1
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Extra)
Set-Location $PSScriptRoot
$steps = @(
  @("scripts/01_check_data.py"),
  @("scripts/02_replicate.py", "--seeds", "3"),
  @("scripts/03_serial_fraud.py", "--seeds", "3"),
  @("scripts/04_extension_gbm.py", "--seeds", "3", "--common-sample"),
  @("scripts/05_tune_rusboost.py", "--seeds", "3"),
  @("scripts/06_interpret.py", "--model", "gbm")
)
foreach ($s in $steps) {
  Write-Host "`n===== $($s[0]) =====" -ForegroundColor Cyan
  python @s @Extra
  if ($LASTEXITCODE -ne 0) { Write-Host "Stopped at $($s[0])" -ForegroundColor Red; exit 1 }
}
python scripts/07_collect_results.py
