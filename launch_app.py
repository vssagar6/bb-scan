import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
REQ_FILE = ROOT / "requirements.txt"
APP_FILE = ROOT / "app.py"


def find_python():
    candidates = []
    if os.name == "nt":
        candidates.extend([shutil.which("py"), shutil.which("python"), shutil.which("python3")])
    else:
        candidates.extend([shutil.which("python3"), shutil.which("python"), shutil.which("py")])
    return next((c for c in candidates if c), None)


def create_or_use_venv():
    if not VENV_DIR.exists():
        python = find_python()
        if python is None:
            raise RuntimeError(
                "Python was not found on this computer. Install Python 3.11 and then run this script again."
            )
        subprocess.run([python, "-m", "venv", str(VENV_DIR)], check=True)

    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def install_requirements(python_bin):
    subprocess.run([str(python_bin), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([str(python_bin), "-m", "pip", "install", "-r", str(REQ_FILE)], check=True)


def run_streamlit(python_bin):
    subprocess.run([str(python_bin), "-m", "streamlit", "run", str(APP_FILE)], cwd=str(ROOT), check=True)


def main():
    print("Setting up the BB Exhaustion Scanner…")
    python_bin = create_or_use_venv()
    install_requirements(python_bin)
    print("\nStarting Streamlit…")
    run_streamlit(python_bin)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as exc:
        print(f"\nError: {exc}")
        input("Press Enter to close…")
