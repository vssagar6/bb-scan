# BB Exhaustion Scanner

A Streamlit app that scans selected NSE stocks for Bollinger Band exhaustion signals, tracks a local portfolio, and runs a historical backtest.

## Run locally

1. Install Python 3.10 or newer.
2. Clone this repository and open a terminal in the project directory.
3. Install dependencies and start Streamlit:

   ```bash
   python -m pip install -r requirements.txt
   streamlit run app.py
   ```

4. Open the local URL printed by Streamlit (usually http://localhost:8501).

The app downloads market data from Yahoo Finance and can fall back to NSE via `jugaad-data`. Internet access is required for live data. The optional `data.json` file is used by the app's local-data mode.

## Deploy on Streamlit Community Cloud

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub and grant it access to this repository. For a private repository, make sure the GitHub account used by Streamlit has access.
2. Choose **Create app** and select this repository, the `main` branch, and `app.py` as the entry point.
3. Deploy. Streamlit installs dependencies from `requirements.txt`.

## Important notes

- This app provides indicators and backtests for informational purposes only; it is not financial advice. Historical results do not guarantee future performance.
- `portfolio.json` is created when you save holdings. It is intentionally excluded from Git. On hosted services, local files may be ephemeral and can be lost when the app restarts or redeploys.
- The current portfolio file is stored on the app server, not separately for each viewer. Do not deploy this app for multiple users with private holdings unless you add authentication and user-specific persistent storage.
- The strategy scans closed candles. Review signals and data quality yourself before making any investment decisions.
