param(
    [switch]$UseVenv = $true
)

if ($UseVenv) {
    python -m venv .venv
    . .venv\\Scripts\\Activate.ps1
}

python -m pip install -r requirements-dev.txt
