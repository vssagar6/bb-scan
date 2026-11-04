# BB Exhaustion Scanner

A Streamlit dashboard for scanning NSE stocks, reviewing Bollinger Band signals, tracking a local portfolio, and backtesting the strategy.

## Easiest way to start

1. Download the project ZIP:
   https://github.com/vssagar6/bb-scan/archive/refs/heads/main.zip
2. Unzip it.
3. Open the extracted folder.
4. Double-click one of these files:
   - Windows: `start_app.bat`
   - Mac/Linux: `start_app.sh`

This creates a local `.venv`, installs Python dependencies, and then opens the Streamlit app automatically.

## If double-click does not work

Use a terminal in the project folder:

- Windows: `start_app.bat`
- macOS/Linux: `bash start_app.sh`

Or run directly:

```bash
python launch_app.py
```

## Other notes

- The app uses local portfolio storage in `portfolio.json` and market data validation for more stable runs.
- Historical and live market data can fail for some symbols; the app surfaces those cleanly instead of crashing.
- This app is for informational and educational purposes only, not financial advice.
