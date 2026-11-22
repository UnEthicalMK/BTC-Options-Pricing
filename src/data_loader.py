import os
import pandas as pd
import numpy as np
from src import config


def extract_clean_btc_snapshot(
    raw_file_name: str = config.RAW_OPTIONS_FILE,
    target_utc_timestamp: int = 1704096000000000,  # 2024-01-01 08:00:00 UTC
    output_file_name: str = config.SNAPSHOT_FILE,
    force_recompute: bool = False
) -> pd.DataFrame:
    """
    Extracts a clean, high-signal Bitcoin options settlement snapshot at 08:00 UTC
    (Deribit's daily settlement time).

    Key Operations:
    1. Strict BTC filter: rejects all ETH/SOL contracts (eliminates 4.17M ETH rows).
    2. Isolates the official 08:00 UTC settlement book state.
    3. Microstructure cleaning: positive bids/asks, non-crossed markets,
       rejects non-executable 1.0 BTC dummy asks and dust spreads.
    4. Feature derivation: log-moneyness (k = ln(K/F)), relative spread, orderbook imbalance.
    5. Saves a lightweight, deterministic snapshot (~500 rows, <100 KB).
    """
    raw_path = os.path.join(config.RAW_DATA_DIR, raw_file_name)
    output_path = os.path.join(config.PROCESSED_DATA_DIR, output_file_name)
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)

    if os.path.exists(output_path) and not force_recompute:
        df = pd.read_csv(output_path)
        # Apply dummy quote guard on loaded cache
        df = df[(df['best_ask'] < 0.95) & (df['relative_spread'] < 2.0)].reset_index(drop=True)
        return df

    print(f"  [data_loader] Streaming raw data to extract 08:00 UTC BTC settlement snapshot...")
    
    use_cols = [
        'symbol', 'timestamp', 'type', 'strike_price', 'expiration',
        'open_interest', 'bid_price', 'bid_amount', 'ask_price', 'ask_amount',
        'mark_price', 'mark_iv', 'underlying_price'
    ]

    rename_map = {
        'type': 'option_type',
        'strike_price': 'strike',
        'bid_price': 'best_bid',
        'ask_price': 'best_ask',
    }

    chunks = pd.read_csv(
        raw_path,
        chunksize=250000,
        usecols=use_cols
    )

    btc_quotes = []

    for chunk in chunks:
        # 1. Filter strictly for BTC contracts (0% ETH)
        btc_chunk = chunk[chunk['symbol'].str.startswith('BTC-')]
        if btc_chunk.empty:
            continue

        before_target = btc_chunk[btc_chunk['timestamp'] <= target_utc_timestamp]
        if not before_target.empty:
            btc_quotes.append(before_target)

        if btc_chunk['timestamp'].max() >= target_utc_timestamp:
            break

    if not btc_quotes:
        raise RuntimeError(f"No BTC quotes found before target timestamp {target_utc_timestamp}")

    all_btc = pd.concat(btc_quotes, ignore_index=True)
    all_btc = all_btc.rename(columns=rename_map)

    # 2. Extract latest BBO quote for each unique contract up to 08:00 UTC
    snapshot = all_btc.sort_values('timestamp').groupby('symbol').last().reset_index()

    # 3. Microstructure cleaning
    # Valid positive quotes
    snapshot = snapshot[(snapshot['best_bid'] > 0.0001) & (snapshot['best_ask'] > 0.0001)]
    # Uncrossed order book
    snapshot = snapshot[snapshot['best_ask'] > snapshot['best_bid']]
    # Reject placeholder dummy asks (>= 0.95 BTC)
    snapshot = snapshot[snapshot['best_ask'] < 0.95]
    # Valid forward and strike
    snapshot = snapshot[(snapshot['underlying_price'] > 1000) & (snapshot['strike'] > 1000)]
    # Reasonable mark IV (10% to 300%)
    snapshot = snapshot[(snapshot['mark_iv'] >= 10.0) & (snapshot['mark_iv'] <= 300.0)]

    # 4. Feature engineering
    snapshot['implied_vol'] = snapshot['mark_iv'] / 100.0
    snapshot['market_mid_btc'] = (snapshot['best_bid'] + snapshot['best_ask']) / 2.0
    snapshot['market_mid'] = snapshot['market_mid_btc'] * snapshot['underlying_price']  # USD

    # Time to maturity T in fractional years
    time_to_exp_sec = (snapshot['expiration'] - snapshot['timestamp']) / 1e6
    snapshot['T'] = time_to_exp_sec / (365.25 * 86400)
    # Filter contracts with at least 6 hours to expiry
    snapshot = snapshot[snapshot['T'] >= (6.0 / (24.0 * 365.25))]

    # SVI Log-moneyness: k = ln(K / F)
    # k < 0: OTM puts / ITM calls; k > 0: OTM calls / ITM puts
    snapshot['log_moneyness'] = np.log(snapshot['strike'] / snapshot['underlying_price'])

    # Microstructure features
    raw_spread_btc = snapshot['best_ask'] - snapshot['best_bid']
    snapshot['bid_ask_spread'] = raw_spread_btc * snapshot['underlying_price']
    snapshot['relative_spread'] = raw_spread_btc / snapshot['market_mid_btc']
    # Filter out extreme non-executable spreads
    snapshot = snapshot[snapshot['relative_spread'] < 2.0]

    # Order book depth imbalance [-1, 1]
    total_depth = snapshot['bid_amount'] + snapshot['ask_amount'] + 1e-6
    snapshot['orderbook_imbalance'] = (snapshot['bid_amount'] - snapshot['ask_amount']) / total_depth

    # Expiry string extraction: BTC-12JAN24-40000-P -> 12JAN24
    snapshot['expiry_str'] = snapshot['symbol'].apply(
        lambda s: s.split('-')[1] if len(s.split('-')) > 1 else 'UNKNOWN'
    )

    snapshot = snapshot.sort_values(['T', 'strike']).reset_index(drop=True)
    snapshot.to_csv(output_path, index=False)
    print(f"  [data_loader] Clean snapshot saved to {output_path} ({len(snapshot)} BTC options).")
    return snapshot


def clean_deribit_options_data(file_name: str = config.RAW_OPTIONS_FILE) -> pd.DataFrame:
    return extract_clean_btc_snapshot(raw_file_name=file_name)


def save_processed_data(df: pd.DataFrame, file_name: str) -> None:
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    output_path = os.path.join(config.PROCESSED_DATA_DIR, file_name)
    df.to_csv(output_path, index=False)
