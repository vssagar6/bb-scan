# BB Exhaustion Scanner

A Streamlit dashboard for scanning NSE stocks, reviewing Bollinger Band signals, tracking a local portfolio, and backtesting the strategy.

## Windows: easiest start

1. Install **Python 3.11 (64-bit)** from https://www.python.org/downloads/. In the installer, enable **Python Launcher** (and optionally **Add Python to PATH**).
2. Download the project ZIP: https://github.com/vssagar6/bb-scan/archive/refs/heads/main.zip
3. Extract the ZIP, open the extracted folder, and double-click `start_app.bat`.

The launcher specifically uses Python 3.11 and installs prebuilt pandas and NumPy packages, avoiding the Visual Studio build-tools error that can occur with newer Python versions. It creates a `.venv` folder and then starts the app in your browser. The first setup needs an internet connection.

If setup failed before, close the command window and run `start_app.bat` again. The launcher will replace an incompatible `.venv` automatically.

## macOS/Linux

Install Python 3.11, extract the ZIP, then double-click `start_app.sh` or run `bash start_app.sh` from the extracted folder.

## Run online

You can deploy from Streamlit Community Cloud. Since this repository is private, authorize Streamlit to access it through your GitHub account. Configure the app with branch `main` and entry file `app.py`.

## Limitations

The app stores portfolio entries in a local `portfolio.json`; hosted files may be temporary and shared by users of the same deployment. Do not store sensitive or private portfolio information in a shared hosted app. This app is informational only, not financial advice; backtests omit brokerage and slippage and do not predict future results.
