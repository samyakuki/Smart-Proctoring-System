Quick run instructions (Windows PowerShell)

1) Open PowerShell in the project folder `c:\Users\DELL\Desktop\5th Sem\FYP`.

2) Create and activate a venv, install deps, and run (copy each block):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements_windows.txt
python .\code1.py
```

Notes:
- The script uses your webcam (index 0). Press `q` in the video window to quit.
- `code1.py` tries to load weights from `best_model.keras`. If it's missing you'll see a warning and the model will run with random weights (less accurate).
- If PowerShell blocks script activation, allow execution with:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

- If you want GPU support on Windows, follow TensorFlow's official GPU setup guide for CUDA/cuDNN and ensure installed `tensorflow` matches the required CUDA version.
