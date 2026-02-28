param([Parameter(Mandatory=$true)][string]$Script)

$isaacRoot = $env:ISAAC_SIM_ROOT

if (-not $isaacRoot) {
  throw "ISAAC_SIM_ROOT is not set. Set it to your local IsaacSim install folder."}

$pythonBat = Join-Path $isaacRoot "python.bat"
if (-not (Test-Path $pythonBat)) {
  throw "Could not find python.bat at: $pythonBat (ISAAC_SIM_ROOT=$isaacRoot)"}

$scriptPath = Resolve-Path $Script

Push-Location $isaacRoot
& $pythonBat $scriptPath
Pop-Location
