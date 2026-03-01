# pick_place_bimanual/run.ps1
param(
  [Parameter(Mandatory=$true, Position=0)]
  [string]$Script,

  [Parameter(ValueFromRemainingArguments=$true)]
  $ScriptArgs
)

$isaacRoot = $env:ISAAC_SIM_ROOT
if (-not $isaacRoot) {
  throw "ISAAC_SIM_ROOT is not set. Set it."
}

$pythonBat = Join-Path $isaacRoot "python.bat"
if (-not (Test-Path $pythonBat)) {
  throw "Could not find python.bat at: $pythonBat (ISAAC_SIM_ROOT=$isaacRoot)"
}

$scriptPath = Resolve-Path $Script

Push-Location $isaacRoot
& $pythonBat $scriptPath @ScriptArgs
Pop-Location
