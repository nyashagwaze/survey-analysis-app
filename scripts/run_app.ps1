param(
    [switch]$InstallDeps,
    [string]$Host = "localhost",
    [int]$Port = 8501
)

if ($InstallDeps) {
    python -m pip install -r requirements.txt
}

streamlit run streamlit_app.py --server.address $Host --server.port $Port
