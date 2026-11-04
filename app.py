"""
BB Exhaustion Scanner  -  Streamlit app
Run:  streamlit run app.py

Strategy (from your Pine script, BB length 22, mult 1.5):
  BUY signal  : green candle closes back INSIDE lower BB after >=1 red candle closed below lower BB
  SELL signal : red candle closes back INSIDE upper BB after >=1 green candle closed above upper BB
  BUY CONFIRMED : a later candle's HIGH crosses the HIGH of the buy-signal candle
  EXIT          : target = Upper BB touched, OR Sell signal, OR (optional) stop-loss = LOW of signal candle
Data: Yahoo Finance (yfinance, symbol + ".NS"), fallback NSE via jugaad-data.
"""
import json
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="BB Exhaustion Scanner", page_icon="📈", layout="wide")
PORTFOLIO_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "portfolio.json")

# ----------------------------------------------------------------------------
# STOCK UNIVERSE (sorted dictionary). Press "Refresh lists from NSE" in the
# sidebar to replace these with the latest official constituents.
# ----------------------------------------------------------------------------
UNIVERSE = {
    "Nifty 50": [
        "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BAJAJFINSV",
        "BAJFINANCE", "BEL", "BHARTIARTL", "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "ETERNAL",
        "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDUSINDBK",
        "INFY", "ITC", "JIOFIN", "JSWSTEEL", "KOTAKBANK", "LT", "M&M", "MARUTI", "NESTLEIND", "NTPC",
        "ONGC", "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SHRIRAMFIN", "SUNPHARMA", "TATACONSUM",
        "TATASTEEL", "TCS", "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO",
    ],
    "Nifty Alpha 50": [
        "ABCAPITAL", "ADANIPOWER", "AUROPHARMA", "BDL", "BEL", "BHARATFORG", "BSE", "CGPOWER", "COFORGE",
        "DIXON", "ETERNAL", "GMRAIRPORT", "HAL", "HDFCAMC", "HINDZINC", "INDHOTEL", "INDIGO", "IRFC",
        "JINDALSTEL", "JSWENERGY", "LAURUSLABS", "LUPIN", "MAZDOCK", "MCX", "MUTHOOTFIN", "NATIONALUM",
        "NAUKRI", "OFSS", "PERSISTENT", "PFC", "PRESTIGE", "RECLTD", "SOLARINDS", "SRF", "SUZLON",
        "TATAELXSI", "TORNTPOWER", "TRENT", "TVSMOTOR", "VEDL", "VOLTAS",
    ],
    "Midcap High Beta": [
        "ADANIENT", "ADANIPORTS", "ADANIPOWER", "ASHOKLEY", "BANKBARODA", "BHARATFORG", "BHEL", "CANBK",
        "DLF", "FEDERALBNK", "GMRAIRPORT", "HINDALCO", "HUDCO", "IDEA", "IDFCFIRSTB", "IREDA", "IRFC",
        "JINDALSTEL", "JSWSTEEL", "LICHSGFIN", "MOTHERSON", "NATIONALUM", "NBCC", "NMDC", "OIL", "PFC",
        "PNB", "RECLTD", "RVNL", "SAIL", "SUZLON", "TATAPOWER", "TATASTEEL", "UNIONBANK", "VEDL", "YESBANK",
    ],
}
UNIVERSE = {k: sorted(set(v)) for k, v in sorted(UNIVERSE.items())}
SECTIONS = ["Nifty 50", "Nifty Alpha 50", "Midcap High Beta", "My Portfolio", "Backtest"]
INDEX_SECTIONS = ["Nifty 50", "Nifty Alpha 50", "Midcap High Beta"]

NSE_URLS = {
    "Nifty 50": "https://niftyindices.com/IndexConstituent/ind_nifty50list.csv",
    "Nifty Alpha 50": "https://niftyindices.com/IndexConstituent/ind_niftyalpha50list.csv",
    "Midcap High Beta": "https://niftyindices.com/IndexConstituent/ind_niftyhighbeta50list.csv",
}


def get_symbols(name):
    return st.session_state.get("lists", {}).get(name, UNIVERSE[name])


def refresh_lists():
    new, failed = {}, []
    for name, url in NSE_URLS.items():
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            r.raise_for_status()
            from io import StringIO
            syms = pd.read_csv(StringIO(r.text))["Symbol"].dropna().astype(str).str.strip().tolist()
            new[name] = sorted(set(syms))
        except Exception:
            failed.append(name)
    st.session_state["lists"] = {**st.session_state.get("lists", {}), **new}
    return failed


# ----------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------
def _clean(df):
    if df is None or df.empty:
        return None
    df = df[["Open", "High", "Low", "Close"]].dropna()
    idx = pd.to_datetime(df.index)
    df.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
    return df.sort_index() if len(df) else None


def _jugaad(symbol, start, interval):
    from jugaad_data.nse import stock_df  # pip install jugaad-data
    d = stock_df(symbol=symbol, from_date=start, to_date=date.today(), series="EQ")
    d = d.rename(columns={"DATE": "Date", "OPEN": "Open", "HIGH": "High", "LOW": "Low", "CLOSE": "Close"})
    d = d.set_index("Date").sort_index()[["Open", "High", "Low", "Close"]]
    if interval == "1wk":
        d.index = pd.to_datetime(d.index)
        d = d.resample("W").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    return d


DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")


@st.cache_resource
def _local_json():
    try:
        with open(DATA_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def _local_df(symbol, years, interval):
    """Daily OHLC from data.json (format: {SYMBOL: {d:[dates], o:[], h:[], l:[], c:[]}})."""
    x = _local_json().get(symbol)
    if not x:
        return None
    df = pd.DataFrame({"Open": x["o"], "High": x["h"], "Low": x["l"], "Close": x["c"]},
                      index=pd.to_datetime(x["d"]))
    df = df[df.index >= pd.Timestamp(date.today() - timedelta(days=int(365 * years)))]
    if interval == "1wk":
        df = df.resample("W").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    return _clean(df)


@st.cache_data(ttl=900, show_spinner=False)
def load_many(symbols, years, interval, source="live"):
    out = {}
    if source == "local":  # use data.json first, live only for symbols missing from the file
        for s in symbols:
            df = _local_df(s, years, interval)
            if df is not None and len(df) >= 60:
                out[s] = df
        symbols = tuple(s for s in symbols if s not in out)
        if not symbols:
            return out
    start = date.today() - timedelta(days=int(365 * years))
    tick = [s + ".NS" for s in symbols]
    try:
        raw = yf.download(tick, start=start, interval=interval, group_by="ticker",
                          auto_adjust=True, progress=False, threads=True)
    except Exception:
        raw = pd.DataFrame()
    for s, t in zip(symbols, tick):
        df = None
        try:
            if isinstance(raw.columns, pd.MultiIndex) and t in raw.columns.get_level_values(0):
                df = _clean(raw[t])
        except Exception:
            df = None
        if df is None or len(df) < 60:  # fallback: NSE directly
            try:
                df = _clean(_jugaad(s, start, interval))
            except Exception:
                pass
        if df is not None and len(df) >= 60:
            out[s] = df
    return out


# ----------------------------------------------------------------------------
# STRATEGY
# ----------------------------------------------------------------------------
def add_ind(df, length, mult):
    d = df.copy()
    basis = d["Close"].rolling(length).mean()
    dev = mult * d["Close"].rolling(length).std(ddof=0)  # Pine stdev = population
    d["basis"], d["upper"], d["lower"] = basis, basis + dev, basis - dev
    c, o, u, lo = (d[x].values for x in ("Close", "Open", "upper", "lower"))
    sell = np.zeros(len(d), bool)
    buy = np.zeros(len(d), bool)
    g = r = 0
    for i in range(len(d)):
        if np.isnan(u[i]):
            continue
        green, red = c[i] > o[i], c[i] < o[i]
        if green and c[i] > u[i]:
            g += 1
        if red and c[i] < lo[i]:
            r += 1
        sell[i] = red and c[i] < u[i] and g > 0
        buy[i] = green and c[i] > lo[i] and r > 0
        if sell[i]:
            g = 0
        if buy[i]:
            r = 0
    d["sell"], d["buy"] = sell, buy
    return d


def find_exit(d, start, sl, use_sl):
    """First exit at/after bar `start`. Priority: stop-loss > target (upper BB) > sell signal."""
    O, H, L, C, U, S = (d[x].values for x in ("Open", "High", "Low", "Close", "upper", "sell"))
    for k in range(start, len(d)):
        if use_sl and sl > 0 and L[k] <= sl:
            return k, min(O[k], sl), "Stop-loss"
        if H[k] >= U[k]:
            return k, max(O[k], U[k]), "Target (Upper BB)"
        if S[k]:
            return k, C[k], "Sell signal"
    return None


def pending_buy(d, symbol, lookback, valid, use_sl):
    """Latest buy signal (within `lookback` bars) and its confirmation status."""
    n = len(d)
    idx = np.where(d["buy"].values)[0]
    if len(idx) == 0 or n - 1 - idx[-1] > lookback:
        return None
    s = idx[-1]
    H, L, O = d["High"].values, d["Low"].values, d["Open"].values
    trig, sl = H[s], L[s]
    status, conf = "⏳ WAITING – buy above trigger", None
    for j in range(s + 1, n):
        if H[j] > trig:
            conf = j
            status = "✅ BUY CONFIRMED"
            break
        if L[j] < sl:
            status = "❌ Failed (signal low broken)"
            break
        if j - s >= valid:
            status = "⌛ Expired"
            break
    if conf is not None:
        ex = find_exit(d, conf + 1, sl, use_sl)
        if ex:
            status = "🏁 Already exited (" + ex[2] + ")"
    ltp, up = d["Close"].values[-1], d["upper"].values[-1]
    entry = max(O[conf], trig) if conf is not None else trig
    return {
        "Symbol": symbol, "Signal Date": d.index[s].date(),
        "Buy Above (Trigger)": round(trig, 2), "Stop-loss (Signal Low)": round(sl, 2),
        "Target (Upper BB)": round(up, 2), "LTP": round(ltp, 2), "Status": status,
        "Confirmed On": d.index[conf].date() if conf is not None else None,
        "Reward:Risk": round((up - entry) / (entry - sl), 2) if entry > sl else None,
    }


def evaluate_holding(h, d, use_sl):
    n = len(d)
    idx = int(d.index.searchsorted(pd.Timestamp(h["entry_date"])))
    sl = h.get("sl", 0) or 0
    if sl == 0:  # auto: low of latest buy-signal candle on/before entry
        b = np.where(d["buy"].values[: min(idx + 1, n)])[0]
        sl = d["Low"].values[b[-1]] if len(b) else 0
    ex = find_exit(d, idx + 1, sl, use_sl)
    ltp = d["Close"].values[-1]
    pnl = (ltp - h["price"]) * h["shares"]
    if ex is None:
        status = "🟢 HOLD"
    elif ex[0] == n - 1:
        status = f"🚨 EXIT NOW – {ex[2]}"
    else:
        status = f"🔴 EXIT DUE – {ex[2]} on {d.index[ex[0]].date()}"
    return {
        "Symbol": h["symbol"], "Shares": h["shares"], "Entry Price": h["price"],
        "Entry Date": h["entry_date"], "LTP": round(ltp, 2), "P&L ₹": round(pnl, 2),
        "P&L %": round((ltp / h["price"] - 1) * 100, 2),
        "Stop-loss": round(sl, 2) if sl else None,
        "Target (Upper BB)": round(d["upper"].values[-1], 2), "Status": status,
    }


def backtest(d, symbol, valid, use_sl, capital):
    n = len(d)
    O, H, L, C = (d[x].values for x in ("Open", "High", "Low", "Close"))
    trades, free = [], 0
    for s in np.where(d["buy"].values)[0]:
        if s < free:
            continue
        trig, sl = H[s], L[s]
        j = None
        for t in range(s + 1, min(n, s + 1 + valid)):
            if H[t] > trig:
                j = t
                break
            if L[t] < sl:
                break
        if j is None:
            continue
        entry = max(O[j], trig)
        ex = find_exit(d, j + 1, sl, use_sl)
        k, price, reason = ex if ex else (n - 1, C[-1], "Still open")
        qty = max(int(capital // entry), 1)
        trades.append({
            "Symbol": symbol, "Signal Date": d.index[s].date(), "Entry Date": d.index[j].date(),
            "Entry": round(entry, 2), "Exit Date": d.index[k].date(), "Exit": round(price, 2),
            "Exit Reason": reason, "Return %": round((price / entry - 1) * 100, 2),
            "Qty": qty, "P&L ₹": round((price - entry) * qty, 2), "Bars Held": int(k - j),
        })
        free = k + 1 if ex else n
    return trades


# ----------------------------------------------------------------------------
# PORTFOLIO STORAGE
# ----------------------------------------------------------------------------
def load_pf():
    try:
        with open(PORTFOLIO_FILE) as f:
            return json.load(f)
    except Exception:
        return []


def save_pf(pf):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(pf, f, indent=1)


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
st.sidebar.title("📈 BB Exhaustion Scanner")
section = st.sidebar.radio("Section", SECTIONS)

with st.sidebar.expander("⚙️ Strategy settings"):
    BB_LEN = st.number_input("BB length", 5, 100, 22)
    BB_MULT = st.number_input("BB multiplier", 0.5, 4.0, 1.5, 0.1)
    INTERVAL = st.selectbox("Timeframe", ["1d", "1wk"], format_func=lambda x: {"1d": "Daily", "1wk": "Weekly"}[x])
    YEARS = st.slider("History to load (years)", 1, 10, 3)
    LOOKBACK = st.slider("Show buy signals from last N candles", 1, 30, 10)
    VALID = st.slider("Signal valid for N candles (wait for breakout)", 1, 15, 5)
    USE_SL = st.checkbox("Use stop-loss = low of signal candle", True)
    SOURCE = st.selectbox("Data source", ["live", "local"], format_func=lambda x: {
        "live": "Live (Yahoo / NSE) – use for daily scanning",
        "local": "Local file data.json – fast, for backtests"}[x])

with st.sidebar.expander("➕ Add a bought stock", expanded=False):
    f_idx = st.selectbox("Index", INDEX_SECTIONS, key="f_idx")
    f_sym = st.selectbox("Stock", get_symbols(f_idx), key="f_sym")
    f_qty = st.number_input("No. of shares", 1, 10_000_000, 10)
    f_px = st.number_input("Buy price ₹", 0.0, 1_000_000.0, 100.0, 0.05)
    f_dt = st.date_input("Buy date", date.today())
    f_sl = st.number_input("Stop-loss ₹ (0 = auto from signal candle)", 0.0, 1_000_000.0, 0.0, 0.05)
    if st.button("Save holding", type="primary"):
        pf = load_pf()
        pf.append({"index": f_idx, "symbol": f_sym, "shares": int(f_qty), "price": float(f_px),
                   "entry_date": str(f_dt), "sl": float(f_sl)})
        save_pf(pf)
        st.success(f"Saved {f_sym}")

if st.sidebar.button("🔄 Refresh lists from NSE"):
    bad = refresh_lists()
    st.sidebar.warning("Could not refresh: " + ", ".join(bad)) if bad else st.sidebar.success("Lists updated")
st.sidebar.caption("Signals use closed candles. Best to scan after 3:30 PM IST.")


def ind_data(symbols):
    raw = load_many(tuple(symbols), YEARS, INTERVAL, SOURCE)
    return {s: add_ind(df, BB_LEN, BB_MULT) for s, df in raw.items()}


def holdings_table(pf, data):
    rows = [evaluate_holding(h, data[h["symbol"]], USE_SL) for h in pf if h["symbol"] in data]
    missing = [h["symbol"] for h in pf if h["symbol"] not in data]
    if missing:
        st.warning("No price data for: " + ", ".join(missing))
    if rows:
        df = pd.DataFrame(rows)
        urgent = df["Status"].str.contains("EXIT")
        if urgent.any():
            st.error(f"{int(urgent.sum())} holding(s) need an exit: " + ", ".join(df.loc[urgent, "Symbol"]))
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(f"Total P&L: ₹{df['P&L ₹'].sum():,.2f}")
    else:
        st.info("No holdings yet. Use '➕ Add a bought stock' in the sidebar.")


# ----------------------------------------------------------------------------
# PAGES
# ----------------------------------------------------------------------------
if section in INDEX_SECTIONS:
    st.title(section)
    syms = get_symbols(section)
    with st.spinner(f"Loading {len(syms)} stocks..."):
        data = ind_data(syms)
    st.caption(f"{len(data)}/{len(syms)} stocks loaded • last candle: "
               f"{max(d.index[-1] for d in data.values()).date() if data else 'n/a'}")
    if len(data) < len(syms):
        st.warning("Missing data for: " + ", ".join(sorted(set(syms) - set(data))))

    st.subheader("🟢 Buy signals")
    rows = [r for s, d in data.items() if (r := pending_buy(d, s, LOOKBACK, VALID, USE_SL))]
    show_all = st.checkbox("Also show failed / expired / exited signals", False, key=f"all_{section}")
    if not show_all:
        rows = [r for r in rows if r["Status"].startswith(("✅", "⏳"))]
    if rows:
        df = pd.DataFrame(rows).sort_values(["Status", "Signal Date"], ascending=[False, False])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption("BUY only when status is ✅ (price has crossed the signal candle's high). "
                   "⏳ = place a buy-stop above the trigger price.")
    else:
        st.info("No active buy signals right now.")

    st.subheader("🔴 Exit alerts – your holdings in this index")
    pf = [h for h in load_pf() if h["index"] == section]
    holdings_table(pf, data)

elif section == "My Portfolio":
    st.title("My Portfolio")
    pf = load_pf()
    if pf:
        need = sorted({h["symbol"] for h in pf})
        with st.spinner("Loading prices..."):
            data = ind_data(need)
        holdings_table(pf, data)
        st.divider()
        labels = [f"{i + 1}. {h['symbol']} – {h['shares']} @ {h['price']} ({h['entry_date']})" for i, h in enumerate(pf)]
        pick = st.selectbox("Remove a holding (sold)", labels)
        if st.button("Remove"):
            pf.pop(labels.index(pick))
            save_pf(pf)
            st.rerun()
    else:
        st.info("No holdings yet. Use '➕ Add a bought stock' in the sidebar.")

else:  # Backtest
    st.title("Backtest")
    c1, c2, c3 = st.columns(3)
    which = c1.selectbox("Stocks", INDEX_SECTIONS + ["All three indices"])
    bt_years = c2.slider("Years of history", 1, 10, 5)
    capital = c3.number_input("Capital per trade ₹", 1000, 10_000_000, 50_000, 1000)
    if st.button("Run backtest", type="primary"):
        syms = sorted({s for n in INDEX_SECTIONS for s in get_symbols(n)}) if which == "All three indices" else get_symbols(which)
        with st.spinner("Downloading data..."):
            raw = load_many(tuple(syms), bt_years, INTERVAL, SOURCE)
        trades = []
        bar = st.progress(0.0)
        for i, (s, df) in enumerate(raw.items()):
            trades += backtest(add_ind(df, BB_LEN, BB_MULT), s, VALID, USE_SL, capital)
            bar.progress((i + 1) / len(raw))
        if not trades:
            st.warning("No trades generated.")
        else:
            t = pd.DataFrame(trades).sort_values("Exit Date")
            wins, losses = t[t["P&L ₹"] > 0], t[t["P&L ₹"] <= 0]
            m = st.columns(5)
            m[0].metric("Trades", len(t))
            m[1].metric("Win rate", f"{len(wins) / len(t) * 100:.1f}%")
            m[2].metric("Avg return/trade", f"{t['Return %'].mean():.2f}%")
            m[3].metric("Net P&L", f"₹{t['P&L ₹'].sum():,.0f}")
            pf_ratio = wins["P&L ₹"].sum() / abs(losses["P&L ₹"].sum()) if len(losses) and losses["P&L ₹"].sum() else float("inf")
            m[4].metric("Profit factor", f"{pf_ratio:.2f}")
            eq = t.assign(Equity=t["P&L ₹"].cumsum()).set_index(pd.to_datetime(t["Exit Date"]))["Equity"]
            st.subheader("Equity curve (cumulative P&L ₹)")
            st.line_chart(eq)
            st.subheader("By exit reason")
            st.dataframe(t.groupby("Exit Reason").agg(Trades=("Symbol", "count"), AvgReturnPct=("Return %", "mean"),
                                                      NetPnL=("P&L ₹", "sum")).round(2), use_container_width=True)
            st.subheader("All trades")
            st.dataframe(t, use_container_width=True, hide_index=True)
            st.download_button("Download trades CSV", t.to_csv(index=False), "backtest_trades.csv", "text/csv")
            st.caption("Entry = max(open, signal-candle high) on the breakout candle. Exit priority: stop-loss, "
                       "upper-BB target, sell signal. No brokerage/slippage included. Past results ≠ future returns.")
