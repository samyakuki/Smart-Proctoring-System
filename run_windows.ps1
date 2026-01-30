# Run helper for Windows PowerShell
# Usage: open PowerShell in project folder and run this file (you may need to set execution policy)

# 1) Create a virtual environment (if not exists)
python -m venv venv

# 2) Activate the venv in PowerShell
# If execution is blocked, run PowerShell as Administrator and: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1

# 3) Install dependencies (Windows-friendly)
pip install -r requirements_windows.txt

# 4) Run the app
python .\code1.py
