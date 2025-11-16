# polymarket_touch_pricer.py
# Skeleton framework to price touch-style BTC/ETH options using Deribit data
# and compare model fair values to Polymarket quotes.

import os
import math
import json
import time
import urllib.parse
import urllib.request
from typing import Optional, Tuple, Dict, List
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # currently unused but kept for future plotting
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm

# ============================
# Global config
# ============================

print("Dependencies ready.")

# --- Core asset / horizon ---
UNDERLYING = "BTC"                  # "BTC" or "ETH"
EXPIRY_ISO = "2026-01-01"           # Target option/barrier horizon (YYYY-MM-DD)
VALUATION_DT = None                 # None = now (UTC); or "YYYY-MM-DD HH:MM"

# --- Monte Carlo config ---
N_PATHS_HIT = 200_000               # Paths for hit probability estimation
N_STEPS = None                      # None = ~daily; or set explicit int
SEED = 42                           # RNG seed
USE_MID_SIGMA = True                # left in for future refinements

# --- K sweep (strike / barrier grid) ---
SWEEP_POINTS = 301                  # Number of K points across the grid
S_MIN_FACTOR = 0.25                 # Grid min = factor * spot
S_MAX_FACTOR = 3.00                 # Grid max = factor * spot

# --- Gating (Polymarket comparison) ---
ABS_THRESHOLD = 0.03                # Must exceed both CI margin and this absolute threshold

# --- Output locations ---
BASE_DIR = os.getcwd()
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data_snapshots")
SMILES_DIR = os.path.join(BASE_DIR, "smiles_term_structure")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Output filenames (lowercase for convenience)
EXP_STR = EXPIRY_ISO.replace("-", "")
SWEEP_CSV = os.path.join(OUTPUTS_DIR, f"{UNDERLYING.lower()}_probability_sweep.csv")
META_JSON = os.path.join(OUTPUTS_DIR, f"{UNDERLYING.lower()}_probability_sweep_meta.json")
PLOT_PNG = os.path.join(OUTPUTS_DIR, f"polymarket_vs_model_{UNDERLYING.lower()}_{EXP_STR}.png")

# Ensure directories exist
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(SMILES_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

print("Parameters set.")

DERIBIT_BASE = "https://www.deribit.com/api/v2"


# ============================
# Generic helpers
# ============================

def yearfrac_365(start: datetime, end: datetime) -> float:
    """
    Year fraction using 365.25-day convention.
    """
    return max(0.0, (end - start).total_seconds() / (365.25 * 24 * 3600.0))


def fetch_url_text(url: str, timeout: float = 10.0) -> Optional[str]:
    """
    Simple HTTP GET that returns response body as text or None on failure.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8")
    except Exception:
        return None


def http_get_json(url: str, timeout: float = 10.0) -> Optional[dict]:
    """
    GET URL and parse JSON response.
    """
    txt = fetch_url_text(url, timeout=timeout)
    if not txt:
        return None
    try:
        return json.loads(txt)
    except Exception:
        return None


# ============================
# Risk-free rates (FRED)
# ============================

def fetch_fred_latest(series_id: str) -> Optional[float]:
    """
    Fetch latest non-missing value from FRED (percent -> decimal).
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    txt = fetch_url_text(url, timeout=10)
    if not txt:
        return None
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    # Skip header row; walk backwards until we find a valid value
    for ln in reversed(lines[1:]):
        parts = ln.split(",")
        if len(parts) < 2:
            continue
        val = parts[1].strip()
        if val in (".", "", "NaN"):
            continue
        try:
            return float(val) / 100.0
        except ValueError:
            continue
    return None


def simple_to_cc(y_simple: float, T_years: float,
                 daycount_base: float, quote_base: float) -> float:
    """
    Convert a simple annualized rate (quoted on quote_base) to
    continuous compounding over T_years.

    df_simple = 1 / (1 + y_simple * T_quote)
    with T_quote = T_years * (daycount_base / quote_base)
    r_cc = -ln(df_simple) / T_years
    """
    T_quote = T_years * (daycount_base / quote_base)
    df = 1.0 / max(1e-12, 1.0 + y_simple * T_quote)
    return -math.log(df) / max(T_years, 1e-12)


def get_risk_free_cc(T: float) -> Tuple[float, Dict[str, float]]:
    """
    Compute continuous-comp risk-free rate over T years.

    Preferred: SOFR (ACT/360), fallback: 3M T-bill (DGS3MO).
    """
    meta = {"source": "", "raw_rate": np.nan}

    # Try SOFR first
    sofr = fetch_fred_latest("SOFR")
    if sofr is not None and sofr > 0.0:
        r = simple_to_cc(y_simple=sofr, T_years=T,
                         daycount_base=365.25, quote_base=360.0)
        meta.update({"source": "SOFR (FRED)", "raw_rate": sofr})
        return float(r), meta

    # Fallback: 3-Month T-bill
    tbill3m = fetch_fred_latest("DGS3MO")
    if tbill3m is not None and tbill3m > 0.0:
        r = simple_to_cc(y_simple=tbill3m, T_years=T,
                         daycount_base=365.25, quote_base=365.25)
        meta.update({"source": "DGS3MO (FRED)", "raw_rate": tbill3m})
        return float(r), meta

    # If all else fails, use 0
    meta.update({"source": "fallback_zero", "raw_rate": 0.0})
    return 0.0, meta


# ============================
# Deribit API helpers
# ============================

def deribit_api(path: str, params: Dict[str, str]) -> Optional[dict]:
    """
    Call Deribit public API and return the 'result' field.
    """
    q = urllib.parse.urlencode(params)
    url = f"{DERIBIT_BASE}{path}?{q}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("result") is not None:
                return data["result"]
            return None
    except Exception:
        return None


# ============================
# Spot prices (Binance + fallback)
# ============================

def fetch_spot_binance(underlying: str) -> Optional[float]:
    """
    Get spot price from Binance for BTC/ETH vs USDT.
    """
    symbol = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}.get(underlying.upper())
    if not symbol:
        return None
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return float(data["price"]) if "price" in data else None
    except Exception:
        return None


def fetch_spot_fallback_deribit(underlying: str) -> Optional[float]:
    """
    Fallback spot from Deribit index (BTC/ETH in USD).
    """
    index_name = "btc_usd" if underlying.upper() == "BTC" else "eth_usd"
    res = deribit_api("public/get_index_price", {"index_name": index_name})
    if res and "index_price" in res:
        return float(res["index_price"])
    return None


def fetch_spot(underlying: str) -> Optional[float]:
    """
    Try Binance first, then Deribit.
    """
    px = fetch_spot_binance(underlying)
    if px is not None:
        return px
    return fetch_spot_fallback_deribit(underlying)


# ============================
# Futures & carry (Deribit)
# ============================

def deribit_fetch_futures(currency: str) -> Optional[List[dict]]:
    """
    Fetch all non-perpetual futures for given currency on Deribit.
    """
    res = deribit_api("public/get_instruments",
                      {"currency": currency, "kind": "future", "expired": "false"})
    if not res:
        return None
    return [x for x in res if not x.get("is_perpetual", False)]


def deribit_ticker_mid(instrument_name: str) -> Optional[float]:
    """
    Compute a midprice for a Deribit instrument:
    prefer (bid+ask)/2, fall back to mark, then last.
    """
    res = deribit_api("public/ticker", {"instrument_name": instrument_name})
    if not res:
        return None
    bid = res.get("best_bid_price") or res.get("bid_price")
    ask = res.get("best_ask_price") or res.get("ask_price")
    mark = res.get("mark_price")
    last = res.get("last_price")

    if bid and ask and bid > 0 and ask > 0:
        return 0.5 * (bid + ask)
    if mark and mark > 0:
        return float(mark)
    if last and last > 0:
        return float(last)
    return None


def compute_carry_from_deribit(
    spot: float,
    valuation_dt: datetime,
    target_T: float,
    currency: str
) -> Tuple[Optional[float], Dict]:
    """
    Compute implied continuous carry c(T) from Deribit futures term structure.

    Uses log(F/S)/T per future, then linearly interpolates to target_T
    if there are futures bracketing that maturity; otherwise uses nearest.
    """
    meta = {"source": "Deribit", "used": [], "interpolation": ""}
    insts = deribit_fetch_futures(currency)
    if not insts:
        return None, meta

    now_ts = valuation_dt.replace(tzinfo=timezone.utc).timestamp()
    pts = []

    for x in insts:
        exp_ms = x.get("expiration_timestamp")
        name = x.get("instrument_name")
        if not exp_ms or not name:
            continue
        exp_ts = exp_ms / 1000.0
        T_i = max(0.0, (exp_ts - now_ts) / (365.25 * 24 * 3600.0))
        if T_i <= 1e-6:
            continue
        mid = deribit_ticker_mid(name)
        if not mid or mid <= 0:
            continue
        c_i = math.log(mid / spot) / T_i
        pts.append((T_i, c_i, name))

    if not pts:
        return None, meta

    pts.sort(key=lambda z: z[0])
    lower = None
    upper = None

    for T_i, c_i, name in pts:
        if T_i < target_T:
            lower = (T_i, c_i, name)
        elif T_i >= target_T and upper is None:
            upper = (T_i, c_i, name)

    # Interpolate between nearest maturities if possible
    if lower and upper and upper[0] > lower[0] + 1e-12:
        Tl, cl, nl = lower
        Th, ch, nh = upper
        w = (target_T - Tl) / (Th - Tl)
        c_T = cl + (ch - cl) * w
        meta["used"] = [
            {"instrument": nl, "T": Tl, "carry": cl},
            {"instrument": nh, "T": Th, "carry": ch},
        ]
        meta["interpolation"] = "linear"
        return c_T, meta

    # Otherwise use nearest maturity
    Tn, cn, nn = min(pts, key=lambda z: abs(z[0] - target_T))
    meta["used"] = [{"instrument": nn, "T": Tn, "carry": cn}]
    meta["interpolation"] = "nearest"
    return cn, meta


def get_rates_auto(
    spot: float,
    valuation_dt: datetime,
    expiry_dt: datetime,
    currency: str
) -> Tuple[float, float, Dict]:
    """
    Combine risk-free rate and futures-implied carry to get:

    - r_cc : continuous risk-free rate
    - q    : effective funding / dividend / staking yield = r_cc - carry
    """
    T = yearfrac_365(valuation_dt, expiry_dt)
    r_cc, r_meta = get_risk_free_cc(T)
    c_T, c_meta = compute_carry_from_deribit(spot, valuation_dt, T, currency)

    meta = {
        "r_source": r_meta.get("source", ""),
        "r_raw_rate": r_meta.get("raw_rate", np.nan),
        "c_source": c_meta.get("source", "Deribit"),
        "c_meta": c_meta,
        "T_years": T,
    }

    if c_T is None:
        q = 0.0
        meta["note"] = "Futures basis unavailable; funding set to 0.0"
        return r_cc, q, meta

    q = r_cc - c_T
    return r_cc, float(q), meta


# ============================
# Options & volatility smiles
# ============================

def deribit_fetch_option_instruments(currency: str) -> Optional[List[dict]]:
    """
    Fetch all live option instruments for given currency on Deribit.
    """
    res = deribit_api("public/get_instruments",
                      {"currency": currency, "kind": "option", "expired": "false"})
    if not res:
        return None
    return res


def select_expiries_around_target(
    option_instruments: List[dict],
    target_dt: datetime,
    max_expiries: int = 6
) -> List[datetime]:
    """
    Given all options, select up to max_expiries expiries closest to target_dt.
    """
    # Collect unique expiry datetimes
    exps_ms = sorted(set(int(x["expiration_timestamp"])
                         for x in option_instruments
                         if x.get("expiration_timestamp")))
    exps_dt = [datetime.utcfromtimestamp(ms / 1000.0) for ms in exps_ms]
    if not exps_dt:
        return []

    # Sort by distance to target
    exps_sorted = sorted(exps_dt, key=lambda d: abs((d - target_dt).total_seconds()))
    chosen = []

    for d in exps_sorted:
        if d not in chosen:
            chosen.append(d)
        if len(chosen) >= max_expiries:
            break

    return sorted(chosen)


def build_smiles_for_expiries(
    currency: str,
    chosen_expiries: List[datetime]
) -> Dict[str, pd.DataFrame]:
    """
    For each expiry, fetch all option tickers and extract 'mark_iv' by strike.

    Returns dict: expiry_iso -> DataFrame with columns
    ['Strike', 'Implied Volatility', 'Expiry'].

    'Implied Volatility' is in percent for easier inspection.
    """
    insts = deribit_fetch_option_instruments(currency)
    if not insts:
        return {}

    # Bucket instruments by expiry
    buckets: Dict[int, List[dict]] = {}
    for x in insts:
        exp_ms = x.get("expiration_timestamp")
        strike = x.get("strike")
        if not exp_ms or strike is None:
            continue
        buckets.setdefault(exp_ms, []).append(x)

    smiles: Dict[str, pd.DataFrame] = {}

    for exp_dt in chosen_expiries:
        exp_ms = int(exp_dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
        group = buckets.get(exp_ms, [])
        rows = []

        for inst in group:
            name = inst.get("instrument_name")
            strike = inst.get("strike")
            if not name or strike is None:
                continue
            tick = deribit_api("public/ticker", {"instrument_name": name})
            if not tick:
                continue
            mark_iv = tick.get("mark_iv")
            if mark_iv is None or mark_iv <= 0:
                continue

            iv_val = float(mark_iv)
            # Some APIs return vol in [0,1], others directly in %
            iv_pct = iv_val * 100.0 if iv_val <= 1.5 else iv_val
            rows.append({"Strike": float(strike), "Implied Volatility": iv_pct})

        if not rows:
            continue

        df = (
            pd.DataFrame(rows)
            .dropna()
            .drop_duplicates("Strike")
            .sort_values("Strike")
        )
        df["Expiry"] = exp_dt.strftime("%Y-%m-%d")
        smiles[df["Expiry"].iloc[0]] = df

    return smiles


# ============================
# Vol surface construction
# ============================

def build_vol_surface(
    smiles: Dict[str, pd.DataFrame],
    valuation_dt: datetime
):
    """
    Build a simple implied volatility surface σ(t, S):

    - For each expiry, fit a PCHIP interpolator over strikes.
    - Interpolate in total variance across maturities.
    """

    expiry_times: List[float] = []
    interpolators: List[PchipInterpolator] = []

    for expiry_iso, df in smiles.items():
        exp_dt = datetime.fromisoformat(expiry_iso)
        T = yearfrac_365(valuation_dt, exp_dt)
        if T <= 0:
            continue

        strikes = df["Strike"].values.astype(float)
        iv_pct = df["Implied Volatility"].values.astype(float)
        iv = iv_pct / 100.0  # convert % -> decimal

        # Ensure strictly increasing strikes
        order = np.argsort(strikes)
        strikes = strikes[order]
        iv = iv[order]

        # PCHIP interpolation over strike
        interp = PchipInterpolator(strikes, iv, extrapolate=True)

        expiry_times.append(T)
        interpolators.append(interp)

    if not expiry_times:
        raise RuntimeError("No valid expiries to build vol surface.")

    expiry_times = np.array(expiry_times)
    sort_idx = np.argsort(expiry_times)
    expiry_times = expiry_times[sort_idx]
    interpolators = [interpolators[i] for i in sort_idx]

    def vol_surface(t: float, S):
        """
        σ(t, S) with t in years from valuation date, S underlying level.
        Interpolates in total variance across maturities.
        """
        S_arr = np.asarray(S, dtype=float)
        t_eff = float(max(t, 1e-6))  # avoid division by zero

        Ts = expiry_times

        # Find bracketing expiries
        idx = np.searchsorted(Ts, t_eff)

        if idx == 0:
            iv = interpolators[0](S_arr)
            iv = np.clip(iv, 1e-4, 5.0)
            w = iv ** 2 * Ts[0]
            sigma = np.sqrt(w / t_eff)
            return sigma

        if idx >= len(Ts):
            iv = interpolators[-1](S_arr)
            iv = np.clip(iv, 1e-4, 5.0)
            w = iv ** 2 * Ts[-1]
            sigma = np.sqrt(w / t_eff)
            return sigma

        T1, T2 = Ts[idx - 1], Ts[idx]
        iv1 = interpolators[idx - 1](S_arr)
        iv2 = interpolators[idx](S_arr)
        iv1 = np.clip(iv1, 1e-4, 5.0)
        iv2 = np.clip(iv2, 1e-4, 5.0)

        # Total variances
        w1 = iv1 ** 2 * T1
        w2 = iv2 ** 2 * T2

        alpha = (t_eff - T1) / max(T2 - T1, 1e-8)
        w = w1 + (w2 - w1) * alpha
        w = np.maximum(w, 1e-8)

        sigma = np.sqrt(w / t_eff)
        return sigma

    return vol_surface


# ============================
# Monte Carlo "touch" pricer
# ============================

def simulate_hit_probs(
    spot: float,
    r: float,
    q: float,
    vol_surface,
    T: float,
    K_grid: np.ndarray,
    n_paths: int,
    n_steps: Optional[int] = None,
    seed: int = SEED
) -> pd.DataFrame:
    """
    Monte Carlo estimation of touch probabilities for each barrier in K_grid.

    Simple GBM with local vol σ(t, S):
        dS_t = S_t * [(r - q) dt + σ(t, S_t) dW_t]

    Hit condition: path's max price over time >= barrier K.
    """

    if n_steps is None:
        # Roughly daily
        n_steps = max(10, int(math.ceil(T * 365.0)))

    dt = T / n_steps
    np.random.seed(seed)

    S = np.full(n_paths, spot, dtype=float)
    max_S = S.copy()

    for step in range(n_steps):
        t = (step + 1) * dt  # time after this step
        sigma = vol_surface(t, S)
        sigma = np.clip(sigma, 1e-4, 5.0)

        dW = np.random.normal(size=n_paths) * math.sqrt(dt)
        drift = (r - q - 0.5 * sigma ** 2) * dt
        S = S * np.exp(drift + sigma * dW)

        max_S = np.maximum(max_S, S)

    K_grid = np.asarray(K_grid, dtype=float)
    probs = []
    ci_low = []
    ci_high = []

    n = float(n_paths)

    for K in K_grid:
        hits = (max_S >= K)
        p = hits.mean()
        se = math.sqrt(max(p * (1.0 - p), 1e-12) / n)
        ci_l = max(0.0, p - 1.96 * se)
        ci_h = min(1.0, p + 1.96 * se)

        probs.append(p)
        ci_low.append(ci_l)
        ci_high.append(ci_h)

    df = pd.DataFrame({
        "Strike": K_grid,
        "HitProb": probs,
        "CI_lower": ci_low,
        "CI_upper": ci_high
    })
    return df


# ============================
# Polymarket comparison helper
# ============================

def compare_with_polymarket(
    hit_df: pd.DataFrame,
    polymarket_quotes: Dict[float, float],
    abs_threshold: float = ABS_THRESHOLD
) -> pd.DataFrame:
    """
    Compare model probabilities with Polymarket quotes.

    polymarket_quotes: dict mapping Strike -> YES price (0..1).
    """
    df = hit_df.copy()
    df["PolymarketPrice"] = df["Strike"].map(polymarket_quotes)
    df = df.dropna(subset=["PolymarketPrice"])

    df["ModelPrice"] = df["HitProb"]
    df["Diff"] = df["PolymarketPrice"] - df["ModelPrice"]
    df["CI_halfwidth"] = 0.5 * (df["CI_upper"] - df["CI_lower"])

    # Edge must exceed both absolute threshold and CI halfwidth
    df["EdgeEnough"] = (
        (np.abs(df["Diff"]) > abs_threshold) &
        (np.abs(df["Diff"]) > df["CI_halfwidth"])
    )

    df["Signal"] = "NONE"
    df.loc[df["EdgeEnough"] & (df["Diff"] < 0), "Signal"] = "BUY YES"
    df.loc[df["EdgeEnough"] & (df["Diff"] > 0), "Signal"] = "BUY NO / SHORT YES"

    return df


# ============================
# Main runner
# ============================

def main():
    if VALUATION_DT is None:
        valuation_dt = datetime.utcnow()
    else:
        valuation_dt = datetime.fromisoformat(VALUATION_DT)

    expiry_dt = datetime.fromisoformat(EXPIRY_ISO)
    currency = UNDERLYING.upper()

    spot = fetch_spot(UNDERLYING)
    if spot is None:
        raise RuntimeError("Could not fetch spot price.")

    print(f"Spot {UNDERLYING}: {spot:.2f}")

    r, q, rates_meta = get_rates_auto(spot, valuation_dt, expiry_dt, currency)
    print(f"Risk-free r (cc): {r:.4%}, q (funding/dividend): {q:.4%}")

    option_insts = deribit_fetch_option_instruments(currency)
    if not option_insts:
        raise RuntimeError("Could not fetch Deribit option instruments.")

    chosen_expiries = select_expiries_around_target(option_insts, expiry_dt, max_expiries=6)
    if not chosen_expiries:
        raise RuntimeError("No expiries around target found.")

    print("Chosen expiries for smiles:")
    for d in chosen_expiries:
        print("  ", d.strftime("%Y-%m-%d"))

    smiles = build_smiles_for_expiries(currency, chosen_expiries)
    if not smiles:
        raise RuntimeError("No smiles built – maybe mark_iv not available.")

    vol_surface = build_vol_surface(smiles, valuation_dt)

    T = yearfrac_365(valuation_dt, expiry_dt)
    print(f"Time to expiry T: {T:.4f} years")

    K_grid = np.linspace(S_MIN_FACTOR * spot, S_MAX_FACTOR * spot, SWEEP_POINTS)
    print("Running Monte Carlo hit-probability sweep...")
    hit_df = simulate_hit_probs(
        spot=spot,
        r=r,
        q=q,
        vol_surface=vol_surface,
        T=T,
        K_grid=K_grid,
        n_paths=N_PATHS_HIT,
        n_steps=N_STEPS,
        seed=SEED
    )

    hit_df.to_csv(SWEEP_CSV, index=False)

    meta = {
        "underlying": UNDERLYING,
        "spot": spot,
        "valuation_dt": valuation_dt.isoformat(),
        "expiry_dt": expiry_dt.isoformat(),
        "T_years": T,
        "r": r,
        "q": q,
        "rates_meta": rates_meta,
        "n_paths": N_PATHS_HIT,
        "n_steps": N_STEPS,
        "s_min_factor": S_MIN_FACTOR,
        "s_max_factor": S_MAX_FACTOR,
        "sweep_points": SWEEP_POINTS,
    }

    with open(META_JSON, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved hit probability sweep to {SWEEP_CSV}")
    print(f"Saved meta to {META_JSON}")
    print("To compare with Polymarket, build a dict {strike: yes_price} and call compare_with_polymarket().")


if __name__ == "__main__":
    main()
