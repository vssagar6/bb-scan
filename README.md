# BB Exhaustion Scanner

A Streamlit dashboard for scanning the included NSE stock lists, reviewing Bollinger Band signals, tracking a local portfolio, and testing the strategy against historical prices.

## Open the app

**Download the complete project as one ZIP:** [bb-scan-main.zip](https://github.com/vssagar6/bb-scan/archive/refs/heads/main.zip)

After downloading, unzip the folder. If you want it online instead of on your computer, open [Streamlit Community Cloud](https://share.streamlit.io/), sign in with GitHub, choose **Create app**, then select this repository, the `main` branch, and `app.py`. The repository is private, so Streamlit must be authorized to access it. The download link above also requires signing in to GitHub while the repository remains private.

## Run on your computer without command-line instructions

Install Python 3.11, then open the project folder and use the operating system's terminal to start Streamlit. If you prefer not to use a terminal, deploy through Streamlit Community Cloud using the steps above.

## Reliability and limitations

- Dependency versions are bounded in `requirements.txt`; Python 3.11 is selected in `.python-version`.
- Market responses are validated and malformed rows are discarded. Failed symbols are reported rather than silently treated as valid data. Network requests have timeouts and cached market data is refreshed periodically.
- Local mode uses `data.json` only; it does not unexpectedly make live requests for missing symbols.
- Portfolio writes use an atomic file replacement and invalid/corrupt entries are skipped safely.
- On hosted services local files can be ephemeral. The portfolio is a local JSON file, shared by viewers of the same deployment, and is not suitable for multiple users' private holdings without authentication and persistent per-user storage.
- This app is for informational and educational use only, not financial advice. Backtests exclude brokerage, slippage, and other real-world costs; past performance does not predict future results.
