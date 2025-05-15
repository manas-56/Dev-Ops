import yfinance as yf
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pickle
import os

# User Input
symbol = "AAPL"
start_date = "2020-01-01"
end_date = "2023-01-01"

# Fetch historical stock data
data = yf.download(symbol, start=start_date, end=end_date)

# Add technical indicators
data.ta.rsi(length=14, append=True)
data.ta.macd(append=True)
data.ta.ema(length=20, append=True)
data.ta.ema(length=50, append=True)
data.ta.ema(length=200, append=True)

# Drop rows with NaNs from indicator calculation
data.dropna(inplace=True)

# Target variable: will next day's close be higher than today?
data['target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
data.dropna(inplace=True)

# Features and target
X = data[['RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'EMA_20', 'EMA_50', 'EMA_200']]
y = data['target']

# Split into training and testing sets
split_index = int(0.8 * len(data))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Standardize features
scaler_path = f'{symbol}_scaler.pkl'
model_path = f'{symbol}_model.pkl'

if os.path.exists(scaler_path) and os.path.exists(model_path):
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(model_path, 'rb') as f:
        rf = pickle.load(f)
else:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)

    # Save model and scaler
    with open(model_path, 'wb') as f:
        pickle.dump(rf, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

# Apply scaler
X_scaled = scaler.transform(X)

# Predictions
data['predicted_risk'] = rf.predict(X_scaled)

# Add human-readable explanation
def explain_risk(row):
    if row['predicted_risk'] == 1:
        if row['RSI_14'] < 30:
            return "Low risk - Oversold condition (RSI < 30)"
        elif row['MACD_12_26_9'] > 0 and row['MACDh_12_26_9'] > 0:
            return "Low risk - Uptrend with strong momentum"
        elif row['EMA_20'] > row['EMA_50'] > row['EMA_200']:
            return "Low risk - Bullish EMA crossover"
        else:
            return "Moderate risk"
    else:
        if row['RSI_14'] > 70:
            return "High risk - Overbought condition (RSI > 70)"
        elif row['MACD_12_26_9'] < 0 and row['MACDh_12_26_9'] < 0:
            return "High risk - Downtrend with weak momentum"
        elif row['EMA_20'] < row['EMA_50'] < row['EMA_200']:
            return "High risk - Bearish EMA crossover"
        else:
            return "Moderate risk"

data['risk_reason'] = data.apply(explain_risk, axis=1)

# Evaluate the model
X_test_scaled = scaler.transform(X_test)
y_pred = rf.predict(X_test_scaled)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Output sample with explanations
print("\nSample Risk Predictions:")
print(data[['Close', 'predicted_risk', 'risk_reason']].tail(10))
