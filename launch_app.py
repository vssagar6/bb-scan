import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
REQ_FILE = ROOT / "requirements.txt"
APP_FILE = ROOT / "app.py"


def run_version_check(command):
    try:
        result = subprocess.run(
            [*command, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def find_python_311():
    # Explicitly choose Python 3.11: newer Python releases may not yet have
    # prebuilt pandas wheels, causing pip to require Visual Studio C++ tools.
    candidates = [["py", "-3.11"]] if os.name == "nt" else [["python3.11"]]
    for command in candidates:
        if run_version_check(command) == "3.11":
            return command
    return None


def get_venv_python():
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def create_or_use_venv():
    python_311 = find_python_311()
    if python_311 is None:
        raise RuntimeError(
            "Python 3.11 (64-bit) was not found. Install it from https://www.python.org/downloads/ "
            "and enable the Python Launcher option, then double-click start_app.bat again."
        )

    venv_python = get_venv_python()
    if VENV_DIR.exists() and run_version_check([str(venv_python)]) != "3.11":
        print("Replacing the old virtual environment with one using Python 3.11…")
        shutil.rmtree(VENV_DIR)

    if not VENV_DIR.exists():
        subprocess.run([*python_311, "-m", "venv", str(VENV_DIR)], check=True)
    return venv_python


def install_requirements(python_bin):
    subprocess.run([str(python_bin), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([str(python_bin), "-m", "pip", "install", "--only-binary=pandas,numpy", "-r", str(REQ_FILE)], check=True)


def run_streamlit(python_bin):
    subprocess.run([str(python_bin), "-m", "streamlit", "run", str(APP_FILE)], cwd=str(ROOT), check=True)


def main():
    print("Setting up BB Exhaustion Scanner with Python 3.11…")
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
        print(f"\nSetup could not finish: {exc}")
        input("Press Enter to close…")
