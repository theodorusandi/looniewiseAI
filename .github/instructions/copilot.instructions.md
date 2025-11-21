---
applyTo: "**"
---

I want to build a forecasting model that answers the question:

“If I buy today, what will the average return over the next 15 days be?”

To achieve this, I take a price series and convert it into returns (log returns or percentage returns).
Then for each sliding window of size 2 \* window (e.g., 30 days when window = 15), I construct:

A backward-looking 15-day window → this forms the input features

A forward-looking 15-day window → this forms the target

Specifically, for each index t:

The features are the past window returns:
returns[t : t+window]

The target is the average return for the next window days:
mean(returns[t+window : t+2*window])

This ensures:

No data leakage

The model only sees past information

The prediction corresponds to the average future 15-day return starting immediately after the input window

The setup answers the trading question directly:
“Given the last 15 days of returns, what average return should I expect over the next 15 days?”

The final model input has shape (window, 1) and the output is a single scalar representing the predicted average forward return.
