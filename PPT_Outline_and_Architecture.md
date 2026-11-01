# PPT Outline (15 slides)
1. **Title** - Intelligent Stock Market Trend Predictor using ML | Bhanu Kiran Bonam | DNR College, Bhimavaram
2. **Abstract** - NSE trend (UP/DOWN) prediction with RandomForest & LSTM for beginner learning; not advice
3. **Problem Statement** - exact-price prediction overfits; need trend-based, explainable system
4. **Objectives** - yfinance data, indicators, RF vs LSTM, web app, >55% accuracy target (compare with naive baseline)
5. **Literature Survey** - table: ARIMA, SVM, RandomForest, LSTM, sentiment-based papers (cite 4-5 real papers you read)
6. **Methodology** - Agile: Collection → Preprocessing → Features → Training → Evaluation → Deployment
7. **Architecture Diagram** - see below
8. **Modules** - Data, Preprocessing/Features, Models, Visualization, Web App
9. **Tech Stack** - Streamlit, Python, yfinance, pandas, numpy, ta, scikit-learn, TensorFlow/Keras, Plotly
10. **Data Leakage Prevention** - time split, scaler on train, no shuffle
11. **LSTM Explained** - forget/input/output gates, cell state, 60-day window
12. **Results** - metrics table, confusion matrices, comparison bar chart (paste from app)
13. **Web App Demo** - screenshots: charts, prediction, learn, practice
14. **Conclusion** - realistic accuracy, baseline comparison, learning platform
15. **Future Scope** - news sentiment analysis, GRU/Transformer, portfolio optimisation, alerts

# Architecture Diagram (describe/draw in draw.io)
```
[Yahoo Finance API / yfinance]
          │  OHLCV (stocks, crypto, forex, commodities)
          ▼
[1 Data Collection] → handle missing (ffill)
          ▼
[2 Preprocessing & Feature Engineering]  MA20/50, RSI, MACD, Bollinger, lag returns, Target
          ▼
[Time Split 80/20, no shuffle] ──► Train set ─┬─► [3A RandomForest] ─┐
                                              └─► [3B LSTM 60-day] ──┤
                                   Test set ──────────────────────────┴─► [Evaluation: Acc/Prec/Rec/F1/CM]
                                                                                   ▼
                                                              [Saved models: .pkl / .h5 / scaler.pkl]
                                                                                   ▼
[4 Visualization: Plotly/Matplotlib] ◄──────────────► [5 Streamlit Web App]
                                                       Home|Live|Charts|Predict|Performance|Practice|Learn|Brokers|Safety|Team
                                                                                   ▼
                                                               User (beginner investor)
```
