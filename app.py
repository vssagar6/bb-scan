"""BB Exhaustion Scanner — Streamlit app with multi-timeframe backtests."""
import json
import os
import tempfile
from datetime import date, timedelta
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="BB Exhaustion Scanner", page_icon="📈", layout="wide")
ROOT = os.path.dirname(os.path.abspath(__file__))
PORTFOLIO_FILE = os.path.join(ROOT, "portfolio.json")
DATA_FILE = os.path.join(ROOT, "data.json")

# Fallback constituents; use the sidebar refresh control to get the latest lists.
UNIVERSE = {
    "Nifty 50": "ADANIENT ADANIPORTS APOLLOHOSP ASIANPAINT AXISBANK BAJAJ-AUTO BAJAJFINSV BAJFINANCE BEL BHARTIARTL CIPLA COALINDIA DRREDDY EICHERMOT ETERNAL GRASIM HCLTECH HDFCBANK HDFCLIFE HEROMOTOCO HINDALCO HINDUNILVR ICICIBANK INDUSINDBK INFY ITC JIOFIN JSWSTEEL KOTAKBANK LT M&M MARUTI NESTLEIND NTPC ONGC POWERGRID RELIANCE SBILIFE SBIN SHRIRAMFIN SUNPHARMA TATACONSUM TATASTEEL TCS TECHM TITAN TRENT ULTRACEMCO WIPRO",
    "Nifty Next 50": "ABB ADANIENSOL ADANIGREEN AMBUJACEM ATGL AUROPHARMA BANKBARODA BPCL BRITANNIA BOSCHL CANBK CHOLAFIN DLF DIVISLAB GAIL GODREJCP HAVELLS HAL HINDPETRO INDHOTEL IOC INDIGO INDUSTOWER IRCTC JINDALSTEL JSWENERGY LTIM LUPIN MARICO MOTHERSON NAUKRI PFC PIDILITIND PNB REC SBICARD SHREECEM SIEMENS TATAPOWER TORNTPHARM TVSMOTOR UNIONBANK VEDL ZOMATO ZYDUSLIFE".split(),
    "Nifty Alpha 50": "ABCAPITAL ADANIPOWER AUROPHARMA BDL BEL BHARATFORG BSE CGPOWER COFORGE DIXON ETERNAL GMRAIRPORT HAL HDFCAMC HINDZINC INDHOTEL INDIGO IRFC JINDALSTEL JSWENERGY LAURUSLABS LUPIN MAZDOCK MCX MUTHOOTFIN NATIONALUM NAUKRI OFSS PERSISTENT PFC PRESTIGE RECLTD SOLARINDS SRF SUZLON TATAELXSI TORNTPOWER TRENT TVSMOTOR VEDL VOLTAS".split(),
    "Midcap High Beta": "ADANIENT ADANIPORTS ADANIPOWER ASHOKLEY BANKBARODA BHARATFORG BHEL CANBK DLF FEDERALBNK GMRAIRPORT HINDALCO HUDCO IDEA IDFCFIRSTB IREDA IRFC JINDALSTEL JSWSTEEL LICHSGFIN MOTHERSON NATIONALUM NBCC NMDC OIL PFC PNB RECLTD RVNL SAIL SUZLON TATAPOWER TATASTEEL UNIONBANK VEDL YESBANK".split(),
}
UNIVERSE = {name: sorted(set(symbols)) for name, symbols in UNIVERSE.items()}
INDEXES = list(UNIVERSE)
SECTIONS = INDEXES + ["My Portfolio", "Backtest"]
NSE_URLS = {
    "Nifty 50": "https://niftyindices.com/IndexConstituent/ind_nifty50list.csv",
    "Nifty Next 50": "https://niftyindices.com/IndexConstituent/ind_niftynext50list.csv",
    "Nifty Alpha 50": "https://niftyindices.com/IndexConstituent/ind_niftyalpha50list.csv",
    "Midcap High Beta": "https://niftyindices.com/IndexConstituent/ind_niftyhighbeta50list.csv",
}
INTERVALS = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}


def get_symbols(name):
    return st.session_state.get("lists", {}).get(name, UNIVERSE[name])


def refresh_lists():
    updated, failed = {}, []
    for name, url in NSE_URLS.items():
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            response.raise_for_status()
            frame = pd.read_csv(StringIO(response.text))
            if "Symbol" not in frame or frame["Symbol"].dropna().empty:
                raise ValueError("NSE list has no symbols")
            updated[name] = sorted(set(frame["Symbol"].dropna().astype(str).str.strip()))
        except Exception:
            failed.append(name)
    st.session_state["lists"] = {**st.session_state.get("lists", {}), **updated}
    return failed


def clean_prices(frame):
    if frame is None or frame.empty:
        return None
    if isinstance(frame.columns, pd.MultiIndex):
        for level in range(frame.columns.nlevels):
            if {"Open", "High", "Low", "Close"}.issubset(set(map(str, frame.columns.get_level_values(level)))):
                frame = frame.copy()
                frame.columns = frame.columns.get_level_values(level)
                break
    required = ["Open", "High", "Low", "Close"]
    if not set(required).issubset(frame.columns):
        return None
    frame = frame.loc[:, required].copy()
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    index = pd.to_datetime(frame.index, errors="coerce", utc=True).tz_convert(None).normalize()
    frame.index = index
    frame = frame[~frame.index.isna()].sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame = frame[(frame.High >= frame.Low) & (frame.Open > 0) & (frame.Close > 0)]
    return frame if not frame.empty else None


@st.cache_data(ttl=3600, show_spinner=False)
def local_json():
    try:
        with open(DATA_FILE, encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def resample_prices(frame, interval):
    rule = {"1wk": "W", "1mo": "ME"}.get(interval)
    if not rule:
        return clean_prices(frame)
    return clean_prices(frame.resample(rule).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna())


def local_prices(symbol, years, interval):
    record = local_json().get(symbol)
    if not isinstance(record, dict):
        return None
    try:
        frame = pd.DataFrame({"Open": record["o"], "High": record["h"], "Low": record["l"], "Close": record["c"]}, index=pd.to_datetime(record["d"], errors="coerce"))
        frame = frame[frame.index >= pd.Timestamp(date.today() - timedelta(days=365 * int(years)))]
        return resample_prices(frame, interval)
    except (KeyError, TypeError, ValueError):
        return None


def nse_fallback(symbol, start, interval):
    from jugaad_data.nse import stock_df
    frame = stock_df(symbol=symbol, from_date=start, to_date=date.today(), series="EQ")
    if frame is None or frame.empty:
        return None
    frame = frame.rename(columns={"DATE": "Date", "OPEN": "Open", "HIGH": "High", "LOW": "Low", "CLOSE": "Close"})
    if "Date" in frame.columns:
        frame = frame.set_index("Date")
    return resample_prices(frame, interval)


@st.cache_data(ttl=900, show_spinner=False)
def load_many(symbols, years, interval, source):
    symbols = tuple(dict.fromkeys(symbols))
    result, pending = {}, []
    for symbol in symbols:
        local = local_prices(symbol, years, interval) if source == "local" else None
        if local is not None and len(local) >= 30:
            result[symbol] = local
        elif source == "live":
            pending.append(symbol)
    if not pending:
        return result
    start = date.today() - timedelta(days=365 * int(years))
    tickers = [f"{symbol}.NS" for symbol in pending]
    try:
        downloaded = yf.download(tickers=tickers, start=start, interval=interval, group_by="ticker", auto_adjust=True, progress=False, threads=True, timeout=20)
    except Exception:
        downloaded = pd.DataFrame()
    for symbol, ticker in zip(pending, tickers):
        candidate = None
        try:
            if not downloaded.empty:
                if isinstance(downloaded.columns, pd.MultiIndex):
                    for level in range(downloaded.columns.nlevels):
                        if ticker in downloaded.columns.get_level_values(level):
                            candidate = downloaded.xs(ticker, axis=1, level=level)
                            break
                elif len(pending) == 1:
                    candidate = downloaded
            candidate = clean_prices(candidate)
        except (KeyError, ValueError, TypeError, AttributeError):
            candidate = None
        if candidate is None or len(candidate) < 30:
            try:
                candidate = clean_prices(nse_fallback(symbol, start, interval))
            except Exception:
                candidate = None
        if candidate is not None and len(candidate) >= 30:
            result[symbol] = candidate
    return result


def add_indicators(frame, length, multiplier):
    data = frame.copy()
    basis = data.Close.rolling(int(length), min_periods=int(length)).mean()
    deviation = data.Close.rolling(int(length), min_periods=int(length)).std(ddof=0) * float(multiplier)
    data["basis"], data["upper"], data["lower"] = basis, basis + deviation, basis - deviation
    buy = np.zeros(len(data), dtype=bool)
    sell = np.zeros(len(data), dtype=bool)
    green_count = red_count = 0
    for i, (close, open_, upper, lower) in enumerate(zip(data.Close, data.Open, data.upper, data.lower)):
        if pd.isna(upper) or pd.isna(lower):
            continue
        green, red = close > open_, close < open_
        if green and close > upper:
            green_count += 1
        if red and close < lower:
            red_count += 1
        sell[i] = red and close < upper and green_count > 0
        buy[i] = green and close > lower and red_count > 0
        if sell[i]:
            green_count = 0
        if buy[i]:
            red_count = 0
    data["buy"], data["sell"] = buy, sell
    return data


def find_exit(data, start):
    """Long-only exits: upper Bollinger Band touch or Pine sell signal; no stop-loss."""
    for i in range(max(0, start), len(data)):
        row = data.iloc[i]
        if pd.notna(row.upper) and row.High >= row.upper:
            return i, max(float(row.Open), float(row.upper)), "Target (Upper BB)"
        if bool(row.sell):
            return i, float(row.Close), "Sell signal"
    return None


def backtest(data, symbol, capital):
    """Enter at next bar open after each buy signal to avoid same-bar lookahead."""
    trades, occupied_until = [], 0
    signals = np.flatnonzero(data.buy.to_numpy())
    for signal in signals:
        entry_index = int(signal) + 1
        if entry_index >= len(data) or int(signal) < occupied_until:
            continue
        entry = float(data.Open.iloc[entry_index])
        if entry <= 0:
            continue
        exit_info = find_exit(data, entry_index,)
        exit_index, price, reason = exit_info if exit_info else (len(data) - 1, float(data.Close.iloc[-1]), "Still open")
        quantity = max(int(capital // entry), 1)
        trades.append({"Symbol": symbol, "Signal Date": data.index[signal].date(), "Entry Date": data.index[entry_index].date(),
                       "Entry": round(entry, 2), "Exit Date": data.index[exit_index].date(), "Exit": round(price, 2),
                       "Exit Reason": reason, "Return %": round((price / entry - 1) * 100, 2), "Qty": quantity,
                       "P&L ₹": round((price - entry) * quantity, 2), "Bars Held": int(exit_index - entry_index)})
        occupied_until = exit_index + 1 if exit_info else len(data)
    return trades


def pending_buy(data, symbol, lookback):
    signals = np.flatnonzero(data.buy.to_numpy())
    if not len(signals):
        return None
    signal = int(signals[-1])
    if len(data) - 1 - signal > lookback:
        return None
    row = data.iloc[signal]
    return {"Symbol": symbol, "Signal Date": data.index[signal].date(), "Close": round(float(row.Close), 2),
            "Status": "🟢 BUY SIGNAL", "Buy on next candle": data.index[signal + 1].date() if signal + 1 < len(data) else "Next candle"}


def read_portfolio():
    try:
        with open(PORTFOLIO_FILE, encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def write_portfolio(holdings):
    fd, temp_path = tempfile.mkstemp(prefix="portfolio-", suffix=".json", dir=ROOT)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(holdings, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, PORTFOLIO_FILE)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def show_candles(symbol, data, trade_rows=None):
    chart = go.Figure()
    chart.add_trace(go.Candlestick(x=data.index, open=data.Open, high=data.High, low=data.Low, close=data.Close, name=symbol))
    for column, label, color in [("basis", "BB Basis", "orange"), ("upper", "Upper BB", "royalblue"), ("lower", "Lower BB", "royalblue")]:
        chart.add_trace(go.Scatter(x=data.index, y=data[column], mode="lines", name=label, line={"color": color, "width": 1}))
    buys = data[data.buy]
    sells = data[data.sell]
    if not buys.empty:
        chart.add_trace(go.Scatter(x=buys.index, y=buys.Low, mode="markers", name="Buy signal", marker={"symbol": "triangle-up", "size": 11, "color": "green"}))
    if not sells.empty:
        chart.add_trace(go.Scatter(x=sells.index, y=sells.High, mode="markers", name="Sell signal (exit)", marker={"symbol": "triangle-down", "size": 11, "color": "red"}))
    if trade_rows is not None and not trade_rows.empty:
        entries = pd.to_datetime(trade_rows["Entry Date"])
        exits = pd.to_datetime(trade_rows["Exit Date"])
        chart.add_trace(go.Scatter(x=entries, y=trade_rows["Entry"], mode="markers", name="Backtest entry", marker={"symbol": "circle", "size": 10, "color": "limegreen"}))
        chart.add_trace(go.Scatter(x=exits, y=trade_rows["Exit"], mode="markers", name="Backtest exit", marker={"symbol": "x", "size": 10, "color": "red"}))
    chart.update_layout(title=f"{symbol} — candlesticks and Bollinger Bands", xaxis_title="Date", yaxis_title="Price (₹)", xaxis_rangeslider_visible=False, height=650, legend_orientation="h")
    st.plotly_chart(chart, use_container_width=True)


# Sidebar controls
st.sidebar.title("📈 BB Exhaustion Scanner")
section = st.sidebar.radio("Section", SECTIONS)
with st.sidebar.expander("⚙️ Strategy settings"):
    bb_length = st.number_input("BB length", 5, 100, 22)
    bb_mult = st.number_input("BB multiplier", 0.5, 4.0, 1.5, 0.1)
    timeframe_label = st.selectbox("🕒 Timeframe", list(INTERVALS), index=0)
    interval = INTERVALS[timeframe_label]
    years = st.slider("History to load (years)", 1, 10, 5)
    lookback = st.slider("Show buy signals from last N candles", 1, 30, 10)
    source = st.selectbox("Data source", ["live", "local"], format_func=lambda value: {"live": "Live (Yahoo / NSE)", "local": "Local data.json only"}[value])

with st.sidebar.expander("➕ Add a bought stock"):
    index_name = st.selectbox("Index", INDEXES, key="holding_index")
    stock_choices = get_symbols(index_name) or UNIVERSE[index_name]
    holding_symbol = st.selectbox("Stock", stock_choices, key="holding_symbol")
    holding_quantity = st.number_input("No. of shares", 1, 10_000_000, 10)
    holding_price = st.number_input("Buy price ₹", 0.01, 1_000_000.0, 100.0, 0.05)
    holding_date = st.date_input("Buy date", date.today())
    if st.button("Save holding", type="primary"):
        portfolio = read_portfolio()
        portfolio.append({"index": index_name, "symbol": holding_symbol, "shares": int(holding_quantity), "price": float(holding_price), "entry_date": str(holding_date)})
        write_portfolio(portfolio)
        st.success(f"Saved {holding_symbol}")

if st.sidebar.button("🔄 Refresh lists from NSE"):
    failures = refresh_lists()
    if failures:
        st.sidebar.warning("Could not refresh: " + ", ".join(failures) + ". Existing lists are kept.")
    else:
        st.sidebar.success("Lists updated")
st.sidebar.caption("Buy alerts are shown in-app when a closed candle triggers the Pine Script buy condition. No stop-loss is used.")


def load_indicators(symbols, years=years, selected_interval=interval):
    loaded = load_many(tuple(symbols), int(years), selected_interval, source)
    return {symbol: add_indicators(frame, bb_length, bb_mult) for symbol, frame in loaded.items() if len(frame) >= bb_length}


def show_holdings(portfolio, data):
    rows = []
    for item in portfolio:
        frame = data.get(item.get("symbol"))
        if frame is None or frame.empty:
            continue
        entry_index = int(frame.index.searchsorted(pd.Timestamp(item["entry_date"])))
        latest = float(frame.Close.iloc[-1])
        pnl = (latest - float(item["price"])) * int(item["shares"])
        rows.append({"Symbol": item["symbol"], "Shares": item["shares"], "Entry Price": item["price"], "Entry Date": item["entry_date"],
                     "LTP": round(latest, 2), "P&L ₹": round(pnl, 2), "P&L %": round((latest / item["price"] - 1) * 100, 2)})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No holdings with available data.")


if section in INDEXES:
    st.title(section)
    symbols = get_symbols(section)
    with st.spinner(f"Loading market data for {len(symbols)} stocks…"):
        indicators = load_indicators(symbols)
    latest = max((frame.index[-1].date() for frame in indicators.values()), default="n/a")
    st.caption(f"{len(indicators)}/{len(symbols)} stocks loaded • timeframe: {timeframe_label} • latest candle: {latest}")
    rows = [signal for symbol, frame in indicators.items() if (signal := pending_buy(frame, symbol, lookback))]
    st.subheader("🟢 Buy signals")
    if rows:
        signals_frame = pd.DataFrame(rows).sort_values(["Signal Date", "Symbol"], ascending=[False, True])
        st.success(f"🔔 {len(rows)} buy signal(s) detected on the selected timeframe.")
        st.dataframe(signals_frame, use_container_width=True, hide_index=True)
    else:
        st.info("No recent buy signals on the selected timeframe.")
    st.subheader("🔴 Portfolio holdings")
    holdings = [item for item in read_portfolio() if item.get("index") == section]
    show_holdings(holdings, indicators)
elif section == "My Portfolio":
    st.title("My Portfolio")
    portfolio = read_portfolio()
    if portfolio:
        with st.spinner("Loading holding prices…"):
            indicators = load_indicators(sorted({item["symbol"] for item in portfolio}))
        show_holdings(portfolio, indicators)
        labels = [f"{i + 1}. {item['symbol']} – {item['shares']} shares @ ₹{item['price']} ({item['entry_date']})" for i, item in enumerate(portfolio)]
        selected = st.selectbox("Remove a holding (sold)", labels)
        if st.button("Remove holding"):
            portfolio.pop(labels.index(selected))
            write_portfolio(portfolio)
            st.rerun()
    else:
        st.info("No holdings yet. Add one using the sidebar.")
else:
    st.title("Backtest")
    col1, col2, col3 = st.columns(3)
    choice = col1.selectbox("Stocks", INDEXES + ["All four indices"])
    backtest_years = col2.slider("Years of history", 1, 10, 5)
    capital = col3.number_input("Capital per trade ₹", 1000, 10_000_000, 50_000, 1000)
    st.caption(f"Timeframe: {timeframe_label}. Long-only; buy at the next candle open after a buy signal; exit at upper BB or sell signal. Stop-loss is disabled.")
    if st.button("Run backtest", type="primary"):
        symbols = sorted({symbol for name in INDEXES for symbol in get_symbols(name)}) if choice == "All four indices" else get_symbols(choice)
        with st.spinner(f"Downloading {timeframe_label.lower()} data for {len(symbols)} stocks…"):
            raw = load_many(tuple(symbols), int(backtest_years), interval, source)
        backtest_frames = {symbol: add_indicators(frame, bb_length, bb_mult) for symbol, frame in raw.items() if len(frame) >= bb_length}
        trades = [trade for symbol, frame in backtest_frames.items() for trade in backtest(frame, symbol, capital)]
        st.session_state["backtest_frames"] = backtest_frames
        st.session_state["backtest_trades"] = pd.DataFrame(trades).sort_values("Exit Date").reset_index(drop=True) if trades else pd.DataFrame()
        st.session_state["backtest_scope"] = f"{choice}|{backtest_years}|{interval}|{bb_length}|{bb_mult}|{capital}"
        st.session_state.pop("backtest_trade_table", None)
    trades = st.session_state.get("backtest_trades", pd.DataFrame())
    if not trades.empty:
        wins, losses = trades[trades["P&L ₹"] > 0], trades[trades["P&L ₹"] <= 0]
        metrics = st.columns(5)
        metrics[0].metric("Trades", len(trades))
        metrics[1].metric("Win rate", f"{len(wins) / len(trades) * 100:.1f}%")
        metrics[2].metric("Avg return/trade", f"{trades['Return %'].mean():.2f}%")
        metrics[3].metric("Net P&L", f"₹{trades['P&L ₹'].sum():,.0f}")
        loss_total = abs(losses["P&L ₹"].sum())
        metrics[4].metric("Profit factor", f"{wins['P&L ₹'].sum() / loss_total:.2f}" if loss_total else "∞")
        equity = trades.assign(Equity=trades["P&L ₹"].cumsum()).set_index(pd.to_datetime(trades["Exit Date"]))["Equity"]
        st.subheader("Equity curve (cumulative P&L ₹)")
        st.line_chart(equity)
        st.subheader("By exit reason")
        st.dataframe(trades.groupby("Exit Reason").agg(Trades=("Symbol", "count"), AvgReturnPct=("Return %", "mean"), NetPnL=("P&L ₹", "sum")).round(2), use_container_width=True)
        st.subheader("All trades — click a row to display its stock candles")
        event = st.dataframe(trades, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", key="backtest_trade_table")
        st.download_button("Download trades CSV", trades.to_csv(index=False), "backtest_trades.csv", "text/csv")
        selected_rows = event.selection.rows if event and hasattr(event, "selection") else []
        if selected_rows:
            trade = trades.iloc[selected_rows[0]]
            symbol = trade["Symbol"]
            frame = st.session_state.get("backtest_frames", {}).get(symbol)
            if frame is not None:
                st.subheader(f"📊 {symbol} — selected trade chart")
                st.caption(f"Trade: {trade['Entry Date']} entry ₹{trade['Entry']:.2f} → {trade['Exit Date']} exit ₹{trade['Exit']:.2f} ({trade['Exit Reason']})")
                show_candles(symbol, frame, trades[trades.Symbol == symbol])
            else:
                st.warning(f"Chart data for {symbol} is no longer available. Run the backtest again.")
        st.caption("No brokerage or slippage included. Historical results do not guarantee future performance.")
    elif "backtest_trades" in st.session_state:
        st.warning("No trades generated. Check data availability and settings.")
