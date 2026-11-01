"""
train_models.py - trains RandomForest (baseline) and LSTM on ALL available
history of many assets, saves rf_model.pkl, lstm_model.h5, scaler.pkl, metrics.json
Run:  python train_models.py
"""
import json, joblib
import numpy as np, pandas as pd, yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from features import add_features, time_split, FEATURES, SEQ_LEN

TICKERS = ["TCS.NS", "RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS",
           "ITC.NS", "LT.NS", "BHARTIARTL.NS", "^NSEI", "^BSESN", "BTC-USD", "GC=F", "AAPL"]

def get(t):
    df = yf.download(t, period="max", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(how="all").ffill()

train_parts, test_parts, seq_tr, seq_te = [], [], [], []
raw = {}
for t in TICKERS:
    try:
        d = add_features(get(t)).dropna(subset=FEATURES)
        d = d.iloc[:-1] if d["Target"].isna().iloc[-1] else d   # drop unlabeled last day
        d = d.dropna(subset=["Target"])
        if len(d) < 300: continue
        raw[t] = time_split(d, 0.8)      # 80-20 TIME split per asset, no shuffle
        print("loaded", t, len(d))
    except Exception as e:
        print("skip", t, e)

tr_all = pd.concat([v[0] for v in raw.values()])
te_all = pd.concat([v[1] for v in raw.values()])

# ---------------- Model A: RandomForest (baseline) ----------------
# Baseline = simple, fast, explainable (feature importance). If a complex model
# can't beat it, the complexity is not worth it. Shallow trees + min_samples_leaf
# reduce OVERFITTING (stock data is very noisy).
rf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=30,
                            random_state=42, n_jobs=-1)
rf.fit(tr_all[FEATURES], tr_all["Target"])
rf_pred = rf.predict(te_all[FEATURES])

# ---------------- Model B: LSTM ----------------
# Scaler is fitted on TRAIN data only -> test data statistics never leak in.
scaler = StandardScaler().fit(tr_all[FEATURES])

def make_seq(part):
    X, y = [], []
    for d in part:
        v = scaler.transform(d[FEATURES]); t = d["Target"].values
        for i in range(SEQ_LEN, len(d)):
            X.append(v[i - SEQ_LEN:i]); y.append(t[i - 1])  # window ends today, label = tomorrow
    return np.array(X, dtype="float32"), np.array(y)

Xtr, ytr = make_seq([v[0] for v in raw.values()])
Xte, yte = make_seq([v[1] for v in raw.values()])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(SEQ_LEN, len(FEATURES))),  # layer 1
    Dropout(0.2),                                                           # randomly drops 20% -> less overfit
    LSTM(50),                                                               # layer 2
    Dropout(0.2),
    Dense(1, activation="sigmoid")])                                        # probability of UP
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
# validation_split takes the LAST 10% of train (no shuffle) -> time order respected
model.fit(Xtr, ytr, epochs=30, batch_size=64, validation_split=0.1, shuffle=False,
          callbacks=[EarlyStopping(patience=4, restore_best_weights=True)], verbose=2)
lstm_pred = (model.predict(Xte, verbose=0).ravel() > 0.5).astype(int)

def metrics(y, p):
    return {"accuracy": accuracy_score(y, p), "precision": precision_score(y, p, zero_division=0),
            "recall": recall_score(y, p, zero_division=0), "f1": f1_score(y, p, zero_division=0),
            "confusion_matrix": confusion_matrix(y, p).tolist()}

out = {"RandomForest": metrics(te_all["Target"], rf_pred), "LSTM": metrics(yte, lstm_pred),
       "baseline_always_up": float(te_all["Target"].mean()),
       "assets": list(raw.keys()), "train_rows": int(len(tr_all)), "test_rows": int(len(te_all)),
       "feature_importance": dict(zip(FEATURES, rf.feature_importances_.round(4).tolist()))}
joblib.dump(rf, "rf_model.pkl"); joblib.dump(scaler, "scaler.pkl"); model.save("lstm_model.h5")
json.dump(out, open("metrics.json", "w"), indent=2)
print(json.dumps({k: out[k] for k in ["RandomForest", "LSTM", "baseline_always_up"]}, indent=2))
