"""BB Exhaustion Scanner — Streamlit app."""
import json
import os
import re
import tempfile
from datetime import date, timedelta
from io import StringIO

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="BB Exhaustion Scanner", page_icon="📈", layout="wide")
ROOT = os.path.dirname(os.path.abspath(__file__))
PORTFOLIO_FILE = os.path.join(ROOT, "portfolio.json")
DATA_FILE = os.path.join(ROOT, "data.json")

UNIVERSE = {
    "Nifty 50": sorted(set("ADANIENT ADANIPORTS APOLLOHOSP ASIANPAINT AXISBANK BAJAJ-AUTO BAJAJFINSV BAJFINANCE BEL BHARTIARTL CIPLA COALINDIA DRREDDY EICHERMOT ETERNAL GRASIM HCLTECH HDFCBANK HDFCLIFE HEROMOTOCO HINDALCO HINDUNILVR ICICIBANK INDUSINDBK INFY ITC JIOFIN JSWSTEEL KOTAKBANK LT M&M MARUTI NESTLEIND NTPC ONGC POWERGRID RELIANCE SBILIFE SBIN SHRIRAMFIN SUNPHARMA TATACONSUM TATASTEEL TCS TECHM TITAN TRENT ULTRACEMCO WIPRO".split())),
    "Nifty 500": [],
    "Nifty Midcap 150": [],
}
INDEXES = list(UNIVERSE)
SECTIONS = INDEXES + ["My Portfolio", "Backtest"]
NSE_URLS = {
    "Nifty 50": "https://niftyindices.com/IndexConstituent/ind_nifty50list.csv",
    "Nifty 500": "https://niftyindices.com/IndexConstituent/ind_nifty500list.csv",
    "Nifty Midcap 150": "https://niftyindices.com/IndexConstituent/ind_niftymidcap150list.csv",
}


def fetch_index_symbols(name):
    try:
        response = requests.get(NSE_URLS[name], headers={"User-Agent": "Mozilla/5.0"}, timeout=12)
        response.raise_for_status()
        frame = pd.read_csv(StringIO(response.text))
        return sorted(set(frame["Symbol"].dropna().astype(str).str.strip().str.upper())) if "Symbol" in frame else []
    except (requests.RequestException, ValueError, pd.errors.ParserError):
        return []


def get_symbols(name):
    lists = st.session_state.get("lists", {})
    if name in lists:
        return lists[name]
    symbols = UNIVERSE.get(name, []) or fetch_index_symbols(name)
    if symbols:
        st.session_state.setdefault("lists", {})[name] = symbols
    return symbols


def refresh_lists():
    updated, failed = {}, []
    for name in INDEXES:
        symbols = fetch_index_symbols(name)
        if symbols:
            updated[name] = symbols
        else:
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
    for col in required:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    try:
        idx = pd.to_datetime(frame.index, errors="coerce", utc=True).tz_convert(None).normalize()
    except (AttributeError, TypeError, ValueError):
        idx = pd.to_datetime(frame.index, errors="coerce")
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        idx = idx.normalize()
    frame.index = idx
    frame = frame[~frame.index.isna()].sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame = frame[(frame.High >= frame.Low) & (frame.Open > 0) & (frame.Close > 0)]
    return frame if not frame.empty else None


@st.cache_data(ttl=3600, show_spinner=False)
def local_json():
    try:
        with open(DATA_FILE, encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def local_prices(symbol, years, interval):
    record = local_json().get(symbol)
    if not isinstance(record, dict):
        return None
    try:
        frame = pd.DataFrame({"Open": record["o"], "High": record["h"], "Low": record["l"], "Close": record["c"]}, index=pd.to_datetime(record["d"], errors="coerce"))
        frame = frame[frame.index >= pd.Timestamp(date.today() - timedelta(days=365 * int(years)))]
        if interval == "1wk":
            frame = frame.resample("W").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
        return clean_prices(frame)
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
    frame = clean_prices(frame)
    if frame is not None and interval == "1wk":
        frame = clean_prices(frame.resample("W").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna())
    return frame


@st.cache_data(ttl=900, show_spinner=False)
def load_many(symbols, years, interval, source):
    symbols = tuple(dict.fromkeys(symbols))
    results, pending = {}, []
    for symbol in symbols:
        local = local_prices(symbol, years, interval) if source == "local" else None
        if local is not None and len(local) >= 30:
            results[symbol] = local
        elif source == "live":
            pending.append(symbol)
    if not pending:
        return results
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
            results[symbol] = candidate
    return results


def normalize_nse_symbol(value):
    value = str(value).strip().upper().replace(" ", "")
    if ":" in value:
        exchange, value = value.split(":", 1)
        if exchange != "NSE":
            raise ValueError("Only NSE symbols are supported (for example, NSE:MCX).")
    if value.endswith(".NS"):
        value = value[:-3]
    if not re.fullmatch(r"[A-Z0-9&.-]+", value):
        raise ValueError("Enter a valid NSE symbol, such as NSE:MCX or RELIANCE.")
    return value


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).fillna(0)
    loss = (-delta).clip(lower=0).fillna(0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def has_bullish_rsi_divergence(data, bar_index, lookback=25):
    if bar_index < 2 or bar_index >= len(data):
        return False
    start = max(0, bar_index - lookback)
    frame = data.iloc[start:bar_index + 1].copy()
    if len(frame) < 5:
        return False
    recent_close = frame["Close"].iloc[:-1]
    recent_rsi = frame["rsi"].iloc[:-1]
    if recent_close.empty or recent_rsi.empty:
        return False
    price_low_idx = recent_close.idxmin()
    rsi_low_idx = recent_rsi.idxmin()
    if price_low_idx == rsi_low_idx:
        return False
    current_close = float(frame["Close"].iloc[-1])
    current_rsi = float(frame["rsi"].iloc[-1])
    low_price = float(frame.loc[price_low_idx, "Close"])
    low_rsi = float(frame.loc[rsi_low_idx, "rsi"])
    return current_close > low_price and current_rsi > low_rsi


def add_indicators(frame, length, multiplier):
    data = frame.copy()
    middle = data["Close"].rolling(int(length), min_periods=int(length)).mean()
    deviation = data["Close"].rolling(int(length), min_periods=int(length)).std(ddof=0) * float(multiplier)
    data["basis"], data["upper"], data["lower"] = middle, middle + deviation, middle - deviation
    data["rsi"] = calculate_rsi(data["Close"], 14)
    buy = np.zeros(len(data), dtype=bool)
    sell = np.zeros(len(data), dtype=bool)
    green_streak = red_streak = 0
    for i, (close, open_, upper, lower) in enumerate(zip(data.Close, data.Open, data.upper, data.lower)):
        if pd.isna(upper) or pd.isna(lower):
            continue
        green, red = close > open_, close < open_
        if green and close > upper:
            green_streak += 1
        if red and close < lower:
            red_streak += 1
        sell[i] = red and close < upper and green_streak > 0
        buy[i] = green and close > lower and red_streak > 0
        if sell[i]:
            green_streak = 0
        if buy[i]:
            red_streak = 0
    data["buy"], data["sell"] = buy, sell
    return data


def confirmed_buys(data, valid):
    """Return confirmed buy entries and stop-loss per entry."""
    entries = []
    for signal in np.flatnonzero(data.buy.to_numpy()):
        trigger = float(data.High.iloc[signal])
        for confirmation in range(int(signal) + 1, min(len(data), int(signal) + 1 + int(valid))):
            if data.High.iloc[confirmation] > trigger:
                price = max(float(data.Open.iloc[confirmation]), trigger)
                stop_loss = min(float(data.Low.iloc[signal]), float(data.Low.iloc[confirmation]))
                if price > 0:
                    entries.append((int(signal), confirmation, price, stop_loss))
                break
    return entries


def exit_on_bar(data, index, average_entry, stop_loss):
    """Stop-out first. If no stop, exit only at +5% profit or sell signal."""
    row = data.iloc[index]
    if stop_loss > 0 and float(row.Low) <= stop_loss:
        return index, min(float(row.Open), float(stop_loss)), "Stop-loss"
    profit_target = float(average_entry) * 1.05
    if pd.notna(row.upper) and row.High >= row.upper:
        target_price = max(float(row.Open), float(row.upper))
        if target_price >= profit_target:
            return index, target_price, "Target (Upper BB, +5%)"
    if bool(row.sell) and float(row.Close) >= profit_target:
        return index, float(row.Close), "Sell signal (+5%)"
    return None


def find_exit(data, start, average_entry, stop_loss):
    for i in range(max(0, start), len(data)):
        result = exit_on_bar(data, i, average_entry, stop_loss)
        if result:
            return result
    return None


def pending_buy(data, symbol, lookback, valid, use_rsi_divergence, rsi_lookback, partial_exit_pct):
    signals = np.flatnonzero(data.buy.to_numpy())
    if not len(signals):
        return None
    signal = int(signals[-1])
    if len(data) - 1 - signal > lookback:
        return None
    if use_rsi_divergence and not has_bullish_rsi_divergence(data, signal, rsi_lookback):
        return None
    trigger = float(data.High.iloc[signal])
    stop = float(data.Low.iloc[signal])
    confirmed = None
    status = "⏳ WAITING – buy above trigger"
    for i in range(signal + 1, min(len(data), signal + 1 + valid)):
        if data.High.iloc[i] > trigger:
            confirmed = i
            status = "✅ BUY CONFIRMED"
            break
        if data.Low.iloc[i] < stop:
            status = "❌ Failed (signal low broken)"
            break
    if confirmed is None and status.startswith("⏳") and len(data) - 1 - signal >= valid:
        status = "⌛ Expired"
    if confirmed is not None:
        entry = max(float(data.Open.iloc[confirmed]), trigger)
        stop = min(stop, float(data.Low.iloc[confirmed]))
        if find_exit(data, confirmed + 1, entry, stop):
            status = "🏁 Already exited"
    current_upper = data.upper.iloc[-1]
    entry = max(float(data.Open.iloc[confirmed]), trigger) if confirmed is not None else trigger
    return {
        "Symbol": symbol,
        "Signal Date": data.index[signal].date(),
        "Buy Above (Trigger)": round(trigger, 2),
        "Stop-loss": round(stop, 2),
        "Target (Upper BB)": round(float(current_upper), 2) if pd.notna(current_upper) else None,
        "LTP": round(float(data.Close.iloc[-1]), 2),
        "Status": status,
        "Confirmed On": data.index[confirmed].date() if confirmed is not None else None,
        "Partial Exit %": partial_exit_pct,
        "RSI Divergence": "Yes" if use_rsi_divergence else "No",
        "Reward:Risk": round((current_upper - entry) / (entry - stop), 2) if pd.notna(current_upper) and entry > stop else None,
    }


def read_portfolio():
    try:
        with open(PORTFOLIO_FILE, encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, list):
            return []
        valid = []
        for item in value:
            if not isinstance(item, dict):
                continue
            try:
                symbol, shares = str(item["symbol"]), int(item["shares"])
                price, entry_date = float(item["price"]), date.fromisoformat(str(item["entry_date"]))
                if symbol and shares > 0 and price > 0:
                    valid.append({**item, "symbol": symbol, "shares": shares, "price": price, "entry_date": str(entry_date), "sl": float(item.get("sl", 0) or 0)})
            except (KeyError, TypeError, ValueError):
                continue
        return valid
    except (OSError, json.JSONDecodeError):
        return []


def write_portfolio(holdings):
    """Atomic replacement prevents partial/corrupt portfolio files."""
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


def evaluate_holding(holding, data, valid):
    if data.empty:
        return None
    entry_index = int(data.index.searchsorted(pd.Timestamp(holding["entry_date"])))
    quantity = int(holding["shares"])
    invested = float(holding["shares"] * holding["price"])
    stop_loss = holding.get("sl", 0) or 0
    if stop_loss == 0 and entry_index < len(data):
        signal_low = float(data.Low.iloc[max(entry_index - 1, 0)])
        entry_low = float(data.Low.iloc[entry_index])
        stop_loss = min(signal_low, entry_low)

    setups = [setup for setup in confirmed_buys(data, valid) if setup[0] > entry_index and setup[1] > entry_index]
    setup_cursor = 0
    exit_info = None
    for i in range(max(0, entry_index + 1), len(data)):
        while setup_cursor < len(setups) and setups[setup_cursor][1] == i:
            _, _, add_price, add_stop = setups[setup_cursor]
            add_qty = max(int(capital_for_add_on(holding, add_price)), 1)
            quantity += add_qty
            invested += add_qty * add_price
            stop_loss = min(stop_loss, add_stop)
            setup_cursor += 1
        average_entry = invested / quantity
        exit_info = exit_on_bar(data, i, average_entry, stop_loss)
        if exit_info:
            break

    average_entry = invested / quantity
    latest = float(data.Close.iloc[-1])
    pnl = latest * quantity - invested
    if exit_info is None:
        status = "🟢 HOLD"
    elif exit_info[0] == len(data) - 1:
        status = f"🚨 EXIT NOW – {exit_info[2]}"
    else:
        status = f"🔴 EXIT DUE – {exit_info[2]} on {data.index[exit_info[0]].date()}"
    upper = data.upper.iloc[-1]
    return {
        "Symbol": holding["symbol"],
        "Shares (incl. modeled adds)": quantity,
        "Original Entry": round(float(holding["price"]), 2),
        "Average Entry": round(average_entry, 2),
        "Stop-loss": round(stop_loss, 2),
        "Entry Date": holding["entry_date"],
        "LTP": round(latest, 2),
        "P&L ₹ (incl. modeled adds)": round(pnl, 2),
        "P&L %": round((latest / average_entry - 1) * 100, 2),
        "Target (Upper BB)": round(float(upper), 2) if pd.notna(upper) else None,
        "Status": status,
    }


def capital_for_add_on(holding, add_price):
    return max(float(holding["price"]) * int(holding["shares"]), 0.0) / max(float(add_price), 1e-6)


def run_backtest(data, symbol, valid, use_stop, capital, use_rsi_divergence=False, rsi_lookback=25, partial_exit_pct=0.5):
    trades, occupied_until = [], 0
    for signal in np.flatnonzero(data.buy.to_numpy()):
        if signal < occupied_until:
            continue
        if use_rsi_divergence and not has_bullish_rsi_divergence(data, int(signal), rsi_lookback):
            continue
        trigger = float(data.High.iloc[signal])
        stop = float(data.Low.iloc[signal])
        confirmation = None
        for i in range(signal + 1, min(len(data), signal + 1 + valid)):
            if data.High.iloc[i] > trigger:
                confirmation = i
                break
            if data.Low.iloc[i] < stop:
                break
        if confirmation is None:
            continue
        entry = max(float(data.Open.iloc[confirmation]), trigger)
        if entry <= 0:
            continue

        quantity = max(int(capital // entry), 1)
        partial_qty = max(int(quantity * float(partial_exit_pct)), 1) if quantity > 1 and partial_exit_pct > 0 else quantity
        runner_qty = max(quantity - partial_qty, 0)
        partial_pnl = 0.0
        runner_pnl = 0.0
        partial_exit_index = None
        partial_exit_price = None
        final_index = len(data) - 1
        final_price = float(data.Close.iloc[-1])
        final_reason = "Still open"

        stop = min(stop, float(data.Low.iloc[confirmation]))
        for i in range(confirmation + 1, len(data)):
            row = data.iloc[i]
            if stop > 0 and row.Low <= stop:
                final_index = i
                final_price = min(float(row.Open), float(stop))
                final_reason = "Stop-loss"
                break
            if partial_exit_index is None and pd.notna(row.upper) and row.High >= row.upper:
                partial_exit_index = i
                partial_exit_price = max(float(row.Open), float(row.upper))
                partial_pnl = (partial_exit_price - entry) * partial_qty
                if runner_qty <= 0:
                    final_index = i
                    final_price = partial_exit_price
                    final_reason = "Partial exit (Upper BB)"
                    break
                continue
            if partial_exit_index is not None:
                if bool(row.sell):
                    final_index = i
                    final_price = float(row.Close)
                    final_reason = "Sell signal"
                    runner_pnl = (final_price - entry) * runner_qty
                    break
                if pd.notna(row.basis) and row.Close < row.basis:
                    final_index = i
                    final_price = float(row.Close)
                    final_reason = "Runner exit (below basis)"
                    runner_pnl = (final_price - entry) * runner_qty
                    break

        if partial_exit_index is None:
            final_index, final_price, final_reason = find_exit(data, confirmation + 1, entry, stop) or (len(data) - 1, float(data.Close.iloc[-1]), "Still open")
            total_pnl = (final_price - entry) * quantity
        else:
            if partial_exit_index is not None and runner_qty > 0 and final_reason == "Still open":
                final_price = float(data.Close.iloc[-1])
                runner_pnl = (final_price - entry) * runner_qty
            total_pnl = partial_pnl + runner_pnl

        if partial_exit_index is not None and runner_qty <= 0:
            final_reason = "Partial exit (Upper BB)"

        trades.append({
            "Symbol": symbol,
            "Signal Date": data.index[signal].date(),
            "Entry Date": data.index[confirmation].date(),
            "Entry": round(entry, 2),
            "Exit Date": data.index[final_index].date(),
            "Exit": round(final_price, 2),
            "Exit Reason": final_reason,
            "Qty": quantity,
            "Partial Qty": partial_qty,
            "Runner Qty": runner_qty,
            "Return %": round((final_price / entry - 1) * 100, 2),
            "P&L ₹": round(total_pnl, 2),
            "Bars Held": int(final_index - confirmation),
        })
        occupied_until = final_index + 1 if (final_reason != "Still open") else len(data)
    return trades


# Sidebar controls
st.sidebar.title("📈 BB Exhaustion Scanner")
section = st.sidebar.radio("Section", SECTIONS)
with st.sidebar.expander("⚙️ Strategy settings"):
    bb_length = st.number_input("BB length", 5, 100, 22)
    bb_mult = st.number_input("BB multiplier", 0.5, 4.0, 1.5, 0.1)
    interval = st.selectbox("Timeframe", ["1d", "1wk"], format_func=lambda value: {"1d": "Daily", "1wk": "Weekly"}[value])
    years = st.slider("History to load (years)", 1, 10, 3)
    lookback = st.slider("Show buy signals from last N candles", 1, 30, 10)
    valid = st.slider("Signal valid for N candles", 1, 15, 5)
    use_stop = st.checkbox("Use stop-loss if price hits the lower of signal and entry candle lows", value=True)
    use_rsi_divergence = st.checkbox("Use RSI bullish divergence filter", value=False)
    rsi_lookback = st.slider("RSI divergence lookback", 10, 60, 25)
    partial_exit_pct = st.slider("Partial exit at upper BB %", 0.0, 1.0, 0.5, 0.05)
    source = st.selectbox("Data source", ["live", "local"], format_func=lambda value: {"live": "Live (Yahoo / NSE)", "local": "Local data.json only"}[value])

with st.sidebar.expander("➕ Add a bought stock"):
    index_name = st.selectbox("Index", INDEXES, key="holding_index")
    choices = get_symbols(index_name) or UNIVERSE[index_name]
    symbol = st.selectbox("Stock", choices, key="holding_symbol") if choices else st.text_input("Stock symbol", "MCX")
    quantity = st.number_input("No. of shares", 1, 10_000_000, 10)
    price = st.number_input("Buy price ₹", 0.01, 1_000_000.0, 100.0, 0.05)
    buy_date = st.date_input("Buy date", date.today())
    stop_price = st.number_input("Stop-loss ₹ (0 = automatic)", 0.0, 1_000_000.0, 0.0, 0.05)
    if st.button("Save holding", type="primary"):
        holdings = read_portfolio()
        holdings.append({"index": index_name, "symbol": symbol, "shares": int(quantity), "price": float(price), "entry_date": str(buy_date), "sl": float(stop_price)})
        try:
            write_portfolio(holdings)
            st.success(f"Saved {symbol}")
        except OSError:
            st.error("Could not save your portfolio. Check that the app folder is writable.")

if st.sidebar.button("🔄 Refresh lists from NSE"):
    failures = refresh_lists()
    if failures:
        st.sidebar.warning("Could not refresh: " + ", ".join(failures) + ". Existing lists are kept.")
    else:
        st.sidebar.success("Lists updated")
st.sidebar.caption("Stop-loss is the lower of the signal low and the entry candle low. Upper-BB exits still require at least 5% profit over average entry before they trigger.")


def get_indicators(symbols):
    loaded = load_many(tuple(symbols), int(years), interval, source)
    return {key: add_indicators(frame, bb_length, bb_mult) for key, frame in loaded.items() if len(frame) >= bb_length}


def show_holdings(holdings, prices):
    rows = [evaluate_holding(item, prices[item["symbol"]], valid) for item in holdings if item["symbol"] in prices]
    missing = sorted({item["symbol"] for item in holdings if item["symbol"] not in prices})
    if missing:
        st.warning("No usable price data for: " + ", ".join(missing))
    rows = [row for row in rows if row]
    if rows:
        frame = pd.DataFrame(rows)
        urgent = frame["Status"].str.contains("EXIT", na=False)
        if urgent.any():
            st.error(f"{int(urgent.sum())} holding(s) need attention: {', '.join(frame.loc[urgent, 'Symbol'])}")
        st.dataframe(frame, use_container_width=True, hide_index=True)
        st.caption(f"Total P&L: ₹{frame['P&L ₹ (incl. modeled adds)'].sum():,.2f}")
    elif not missing:
        st.info("No holdings yet. Add one using the sidebar.")


if section in INDEXES:
    st.title(section)
    symbols = get_symbols(section)
    with st.spinner(f"Loading market data for {len(symbols)} stocks…"):
        indicators = get_indicators(symbols)
    latest = max((frame.index[-1].date() for frame in indicators.values()), default="n/a")
    st.caption(f"{len(indicators)}/{len(symbols)} stocks loaded • latest candle: {latest}")
    missing = sorted(set(symbols) - set(indicators))
    if missing:
        st.warning("Some stocks did not return enough valid data: " + ", ".join(missing))
    st.subheader("🟢 Buy signals")
    rows = [result for name, frame in indicators.items() if (result := pending_buy(frame, name, lookback, valid, use_rsi_divergence, rsi_lookback, partial_exit_pct))]
    show_all = st.checkbox("Also show failed / expired / exited signals", False, key=f"show_all_{section}")
    if not show_all:
        rows = [row for row in rows if row["Status"].startswith(("✅", "⏳"))]
    if rows:
        st.dataframe(pd.DataFrame(rows).sort_values(["Status", "Signal Date"], ascending=[False, False]), use_container_width=True, hide_index=True)
    else:
        st.info("No matching buy signals in the selected lookback period.")
    st.subheader("🔴 Exit alerts – your holdings in this index")
    show_holdings([item for item in read_portfolio() if item.get("index") == section], indicators)

elif section == "My Portfolio":
    st.title("My Portfolio")
    holdings = read_portfolio()
    if holdings:
        with st.spinner("Loading holding prices…"):
            indicators = get_indicators(sorted({item["symbol"] for item in holdings}))
        show_holdings(holdings, indicators)
        labels = [f"{i + 1}. {item['symbol']} – {item['shares']} shares @ ₹{item['price']} ({item['entry_date']})" for i, item in enumerate(holdings)]
        selected = st.selectbox("Remove a holding (sold)", labels)
        if st.button("Remove holding"):
            holdings.pop(labels.index(selected))
            try:
                write_portfolio(holdings)
                st.rerun()
            except OSError:
                st.error("Could not update the portfolio file.")
    else:
        st.info("No holdings yet. Add one using the sidebar.")

else:
    st.title("Backtest")
    st.caption("Choose an index or enter an individual NSE symbol, such as NSE:MCX. Confirmed buys add the same capital to open trades. Stop uses the lower of signal and entry candle lows; profit exits need at least 5% over average entry.")
    col1, col2, col3 = st.columns(3)
    choice = col1.selectbox("Stocks", INDEXES + ["All listed indices", "Custom stock"])
    test_years = col2.slider("Years of history", 1, 10, 5)
    capital = col3.number_input("Capital per trade ₹", 1000, 10_000_000, 50_000, 1000)
    custom_symbol = st.text_input("NSE stock symbol", "NSE:MCX", help="Examples: NSE:MCX, MCX, or RELIANCE") if choice == "Custom stock" else ""
    if st.button("Run backtest", type="primary"):
        try:
            if choice == "Custom stock":
                symbols = [normalize_nse_symbol(custom_symbol)]
            elif choice == "All listed indices":
                symbols = sorted({item for name in INDEXES for item in get_symbols(name)})
            else:
                symbols = get_symbols(choice)
        except ValueError as exc:
            st.error(str(exc))
            symbols = []
        if not symbols:
            st.warning("No stocks to backtest. Check the symbol or refresh NSE lists.")
        else:
            with st.spinner(f"Downloading data for {len(symbols)} stock(s)…"):
                raw = load_many(tuple(symbols), int(test_years), interval, source)
            missing = sorted(set(symbols) - set(raw))
            if missing:
                st.warning("No usable price data for: " + ", ".join(missing) + ". Check symbol and data source.")
            trades = []
            usable = [(name, frame) for name, frame in raw.items() if len(frame) >= bb_length]
            progress = st.progress(0.0)
            for i, (name, frame) in enumerate(usable):
                trades.extend(run_backtest(add_indicators(frame, bb_length, bb_mult), name, valid, use_stop, capital, use_rsi_divergence, rsi_lookback, partial_exit_pct))
                progress.progress((i + 1) / max(len(usable), 1))
            if not trades:
                st.warning("No trades generated. Check data availability and settings.")
            else:
                trade_frame = pd.DataFrame(trades).sort_values("Exit Date")
                winners = trade_frame[trade_frame["P&L ₹"] > 0]
                losers = trade_frame[trade_frame["P&L ₹"] <= 0]
                metrics = st.columns(5)
                metrics[0].metric("Trades", len(trade_frame))
                metrics[1].metric("Win rate", f"{len(winners) / len(trade_frame) * 100:.1f}%")
                metrics[2].metric("Avg return/trade", f"{trade_frame['Return %'].mean():.2f}%")
                metrics[3].metric("Net P&L", f"₹{trade_frame['P&L ₹'].sum():,.0f}")
                loss_total = abs(losers["P&L ₹"].sum())
                profit_factor = winners["P&L ₹"].sum() / loss_total if loss_total else float("inf")
                metrics[4].metric("Profit factor", f"{profit_factor:.2f}")
                equity = trade_frame.assign(Equity=trade_frame["P&L ₹"].cumsum()).set_index(pd.to_datetime(trade_frame["Exit Date"]))["Equity"]
                st.subheader("Equity curve (cumulative P&L ₹)")
                st.line_chart(equity)
                st.subheader("By exit reason")
                summary = trade_frame.groupby("Exit Reason").agg(Trades=("Symbol", "count"), AvgReturnPct=("Return %", "mean"), NetPnL=("P&L ₹", "sum")).round(2)
                st.dataframe(summary, use_container_width=True)
                st.subheader("All trades")
                st.dataframe(trade_frame, use_container_width=True, hide_index=True)
                st.download_button("Download trades CSV", trade_frame.to_csv(index=False), "backtest_trades.csv", "text/csv")
                st.caption("No brokerage or slippage included. Historical results do not guarantee future performance.")
