import yfinance as yf
import os
import json
import pandas as pd

DATA_PATH = "tsla_data.json"

if os.path.exists(DATA_PATH):
    # Read from file if we've already downloaded the data.
    with open(DATA_PATH) as f:
        tsla_hist = pd.read_json(DATA_PATH)
else:
    tsla = yf.Ticker("TSLA")
    tsla_hist = tsla.history(period="max")

    # Save file to json in case we need it later.  This prevents us from having to re-download it every time.
    tsla_hist.to_json(DATA_PATH)


#use the .tail method on DataFrames to check the last 5 rows  of the data

tsla_hist.tail(5)

#built-in plot method on DataFrame will show a figure os stock price change over years
tsla_hist.plot.line(y="Close", use_index=True)


#indicate if the price went up or down, we set the target
data = tsla_hist[["Close"]]
data = data.rename(columns = {'Close':'Actual_Close'})


data["Target"] = tsla_hist.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]

#show figure
data.tail()

#predict future prices


tsla_prev = tsla_hist.copy()
tsla_prev = tsla_prev.shift(1)

#create training data

predictors = ["Close", "Volume", "Open", "High", "Low"]
data = data.join(tsla_prev[predictors]).iloc[1:]


#random forest classifier to generate  predictions

from sklearn.ensemble import RandomForestClassifier
import numpy as np

# random forest classification model
model = RandomForestClassifier(n_estimators=100, min_samples_split=400, random_state=1)


train = data.iloc[:-365]
test = data.iloc[-365:]

model.fit(train[predictors], train["Target"])

from sklearn.metrics import precision_score

#error of predictions
preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)
precision_score(test["Target"], preds)

combined = pd.concat({"Target": test["Target"],"Predictions": preds}, axis=1)
combined.plot()





