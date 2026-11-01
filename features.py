"""
features.py - shared by training script, notebook and app.
Feature engineering + target creation + DATA LEAKAGE prevention notes.
"""
import numpy as np
import pandas as pd
import ta  # technical analysis library

SEQ_LEN = 60  # LSTM looks at the last 60 days

# All features are SCALE-FREE (ratios/percentages), so one model can learn
# from many different stocks/cryptos without being confused by price level.
FEATURES = ["ret_1", "ret_2", "ret_5", "ret_10", "ma20_gap", "ma50_gap",
            "rsi", "macd_pct", "macd_hist_pct", "bb_pos", "bb_width",
            "vol_chg", "volatility_10"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Takes OHLCV dataframe, returns it with indicators + Target column."""
    df = df.copy()
    c = df["Close"]

    # --- Technical indicators (ta library) ---
    df["MA20"] = c.rolling(20).mean()
    df["MA50"] = c.rolling(50).mean()
    df["RSI"] = ta.momentum.RSIIndicator(c, window=14).rsi()
    macd = ta.trend.MACD(c)
    df["MACD"], df["MACD_signal"] = macd.macd(), macd.macd_signal()
    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    df["BB_high"], df["BB_low"] = bb.bollinger_hband(), bb.bollinger_lband()

    # --- Lag features (Close-1, Close-2, Close-5 expressed as % change) ---
    df["ret_1"] = c.pct_change(1)
    df["ret_2"] = c.pct_change(2)
    df["ret_5"] = c.pct_change(5)
    df["ret_10"] = c.pct_change(10)

    # --- Scale-free versions of indicators (model inputs) ---
    df["ma20_gap"] = c / df["MA20"] - 1
    df["ma50_gap"] = c / df["MA50"] - 1
    df["rsi"] = df["RSI"] / 100
    df["macd_pct"] = df["MACD"] / c
    df["macd_hist_pct"] = (df["MACD"] - df["MACD_signal"]) / c
    df["bb_pos"] = (c - df["BB_low"]) / (df["BB_high"] - df["BB_low"])
    df["bb_width"] = (df["BB_high"] - df["BB_low"]) / c
    df["vol_chg"] = df["Volume"].pct_change().replace([np.inf, -np.inf], 0).fillna(0).clip(-1, 3)
    df["volatility_10"] = df["ret_1"].rolling(10).std()

    # --- TARGET: 1 if tomorrow's close > today's close else 0 ---
    # shift(-1) looks one day INTO THE FUTURE, so it is used ONLY as the label.
    # DATA LEAKAGE RULE: every feature above uses only today's and past data.
    df["Target"] = (c.shift(-1) > c).astype(float)
    df.loc[df.index[-1], "Target"] = np.nan  # last day: tomorrow unknown

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def time_split(df, ratio=0.8):
    """Time-based split. NO shuffle: past -> train, future -> test.
    Shuffling would let the model 'see the future' = data leakage."""
    cut = int(len(df) * ratio)
    return df.iloc[:cut], df.iloc[cut:]
