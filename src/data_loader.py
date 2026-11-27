import os
import pandas as pd
import numpy as np
from src import config
from src.data_fetcher import fetch_historical_deribit_data


def _apply_microstructure_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply consistent liquidity, spread, and sanity filters to snapshot data."""
    df = df.copy()
    df = df[(df["best_bid"] > 0.0001) & (df["best_ask"] > 0.0001)]
    df = df[df["best_ask"] > df["best_bid"]]
    # Reject placeholder dummy asks (>= 0.95 BTC)
    df = df[df["best_ask"] < 0.95]
    df = df[(df["underlying_price"] > 1000) & (df["strike"] > 1000)]
    df = df[(df["mark_iv"] >= 10.0) & (df["mark_iv"] <= 300.0)]
    if "relative_spread" in df.columns:
        df = df[df["relative_spread"] < 2.0]
    if "T" in df.columns:
        df = df[df["T"] >= (6.0 / (24.0 * 365.25))]
    if "mark_usd" not in df.columns and "mark_price" in df.columns:
        df["mark_usd"] = df["mark_price"] * df["underlying_price"]
    return df.sort_values(["T", "strike"]).reset_index(drop=True)


def extract_clean_btc_snapshot(
    raw_file_name: str = config.RAW_OPTIONS_FILE,
    target_utc_timestamp: int = 1704096000000000,  # 2024-01-01 08:00:00 UTC
    output_file_name: str = config.SNAPSHOT_FILE,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Loads or extracts a clean Bitcoin options settlement snapshot at 08:00 UTC.

    Fallback order when `force_recompute` is False:
    1. Cached processed snapshot in `PROCESSED_DATA_DIR`.
    2. Raw compressed Deribit feed in `RAW_DATA_DIR` (if present).
    3. Tracked lightweight sample snapshot in `SAMPLE_DATA_DIR` (for fresh clones).
    4. Tardis.dev download via `fetch_historical_deribit_data` (requires network).
    """
    raw_path = os.path.join(config.RAW_DATA_DIR, raw_file_name)
    output_path = os.path.join(config.PROCESSED_DATA_DIR, output_file_name)
    sample_path = os.path.join(config.SAMPLE_DATA_DIR, output_file_name)
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)

    # 1. Use processed snapshot if available
    if os.path.exists(output_path) and not force_recompute:
        raw_cached = pd.read_csv(output_path)
        df = _apply_microstructure_filters(raw_cached)
        # Refresh stale cache on disk if extra/dummy rows were stripped
        if len(df) != len(raw_cached) or "mark_usd" not in raw_cached.columns:
            df.to_csv(output_path, index=False)
        return df

    # 2. If raw archive missing and not forcing raw recompute, try tracked sample first
    if not os.path.exists(raw_path):
        if os.path.exists(sample_path) and not force_recompute:
            print(f"  [data_loader] Raw archive absent; loading bundled sample snapshot: {sample_path}")
            df = _apply_microstructure_filters(pd.read_csv(sample_path))
            df.to_csv(output_path, index=False)
            return df
        print("  [data_loader] Raw archive and sample missing locally; fetching via Tardis.dev...")
        downloaded = fetch_historical_deribit_data("2024-01-01", "2024-01-02")
        raw_path = downloaded[0]

    print("  [data_loader] Streaming raw data to extract 08:00 UTC BTC settlement snapshot...")

    use_cols = [
        "symbol", "timestamp", "type", "strike_price", "expiration",
        "open_interest", "bid_price", "bid_amount", "ask_price", "ask_amount",
        "mark_price", "mark_iv", "underlying_price",
    ]
    rename_map = {
        "type": "option_type",
        "strike_price": "strike",
        "bid_price": "best_bid",
        "ask_price": "best_ask",
    }

    btc_quotes = []
    for chunk in pd.read_csv(raw_path, chunksize=250000, usecols=use_cols):
        btc_chunk = chunk[chunk["symbol"].str.startswith("BTC-")]
        if btc_chunk.empty:
            continue

        before_target = btc_chunk[btc_chunk["timestamp"] <= target_utc_timestamp]
        if not before_target.empty:
            btc_quotes.append(before_target)

        if btc_chunk["timestamp"].max() >= target_utc_timestamp:
            break

    if not btc_quotes:
        raise RuntimeError(f"No BTC quotes found before target timestamp {target_utc_timestamp}")

    all_btc = pd.concat(btc_quotes, ignore_index=True).rename(columns=rename_map)

    # Extract latest BBO quote for each unique contract up to 08:00 UTC
    snapshot = all_btc.sort_values("timestamp").groupby("symbol").last().reset_index()

    # Derived pricing, time-to-expiry, and microstructure features
    snapshot["implied_vol"] = snapshot["mark_iv"] / 100.0
    snapshot["market_mid_btc"] = (snapshot["best_bid"] + snapshot["best_ask"]) / 2.0
    snapshot["market_mid"] = snapshot["market_mid_btc"] * snapshot["underlying_price"]
    snapshot["mark_usd"] = snapshot["mark_price"] * snapshot["underlying_price"]

    time_to_exp_sec = (snapshot["expiration"] - snapshot["timestamp"]) / 1e6
    snapshot["T"] = time_to_exp_sec / (365.25 * 86400)

    # SVI Log-moneyness: k = ln(K / F)
    snapshot["log_moneyness"] = np.log(snapshot["strike"] / snapshot["underlying_price"])

    raw_spread_btc = snapshot["best_ask"] - snapshot["best_bid"]
    snapshot["bid_ask_spread"] = raw_spread_btc * snapshot["underlying_price"]
    snapshot["relative_spread"] = raw_spread_btc / np.maximum(snapshot["market_mid_btc"], 1e-9)

    total_depth = snapshot["bid_amount"] + snapshot["ask_amount"] + 1e-6
    snapshot["orderbook_imbalance"] = (snapshot["bid_amount"] - snapshot["ask_amount"]) / total_depth

    snapshot["expiry_str"] = snapshot["symbol"].apply(
        lambda s: s.split("-")[1] if len(s.split("-")) > 1 else "UNKNOWN"
    )

    snapshot = _apply_microstructure_filters(snapshot)
    snapshot.to_csv(output_path, index=False)
    print(f"  [data_loader] Clean snapshot saved to {output_path} ({len(snapshot)} BTC options).")
    return snapshot
