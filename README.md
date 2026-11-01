# TrendIQ - Intelligent Stock Market Trend Predictor using ML
BTech CSE/AIML final year project by **Bhanu Kiran Bonam**, DNR College of Engineering and Technology, Bhimavaram.
**Educational only - not financial advice. No model can predict markets with certainty.**

## Features
Stocks, indices, crypto, commodities, forex, ETFs • search + duration selector • candlestick/RSI/MACD/Bollinger charts •
RandomForest + LSTM next-day UP/DOWN prediction • model comparison • virtual paper trading • learning manual • broker guide • safety tools.

## Repository files
| File | Purpose |
|---|---|
| `app.py` | Streamlit web app (entry point) |
| `features.py` | Indicators, target, time split (shared) |
| `train_models.py` | Trains RandomForest + LSTM, creates model files |
| `model_training.ipynb` | Step-by-step notebook for viva |
| `requirements.txt` | Cloud/deploy packages (no TensorFlow) |
| `requirements-train.txt` | Local training packages (with TensorFlow) |
| `.streamlit/config.toml`, `.devcontainer/` | Theme and dev container |
| `rf_model.pkl`, `scaler.pkl`, `metrics.json`, `lstm_model.h5` | Created by `train_models.py` - commit them |

## Run locally
```bash
pip install -r requirements-train.txt
python train_models.py          # one time (10-30 min), creates the model files
streamlit run app.py
```

## Deploy on Streamlit Community Cloud (free, no card)
1. Push all files (including the generated model files) to GitHub.
2. Go to share.streamlit.io → sign in with GitHub → **Create app**.
3. Repo: `BBhanuKiran/stock-prediction`, branch `main`, main file `app.py`.
4. Advanced settings → Python **3.11** → **Deploy**.

The deployed app uses a RandomForest trained live on each asset plus the saved multi-asset RandomForest. The LSTM is trained locally; its results appear in the Model Performance tab through `metrics.json`. To run the LSTM live in the cloud, add `tensorflow-cpu` to `requirements.txt` (may exceed free memory).

## Accuracy note
Realistic next-day direction accuracy is about 50-56%. "100% accurate" is impossible. Models are compared with a naive baseline and data leakage is prevented (time split, scaler fitted on train only, no future features).
