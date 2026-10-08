@echo off

cd /d "%~dp0"

set PYTHONUTF8=1

set LABEL_STUDIO_BASE_DATA_DIR=%~dp0data

set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true

set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=<KAMP-harness>\data\images


set CSRF_TRUSTED_ORIGINS=https://*.trycloudflare.com,http://localhost:8080

set SSRF_PROTECTION_ENABLED=false

set USE_DEFAULT_BANNED_SUBNETS=false

echo [1/2] Starting click-to-box helper (port 9090) in a second window...

start "fixed_box_backend" venv_ml\Scripts\python fixed_box_backend.py

echo [2/2] Starting Label Studio at http://localhost:8080  (keep both windows open)

venv\Scripts\label-studio start --no-browser

pause

