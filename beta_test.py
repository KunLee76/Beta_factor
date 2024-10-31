import yfinance as yf # type: ignore
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression # type: ignore
import matplotlib.pyplot as plt

# 1. 下載數據（以S&P 500和個別股票為例）
# 取得個別股票（如AAPL）和市場指數（S&P 500）
stock = yf.download('AAPL', start='2017-01-01', end='2023-01-01')['Adj Close']
market = yf.download('^GSPC', start='2017-01-01', end='2023-01-01')['Adj Close']

# 2. 計算日回報
stock_returns = stock.pct_change().dropna()
market_returns = market.pct_change().dropna()

# 3. 合併數據集
data = pd.concat([stock_returns, market_returns], axis=1, join='inner')
data.columns = ['stock_returns', 'market_returns']

# 4. 設定無風險利率 (這裡假設為2%)
Rf = 0.02 / 252  # 日無風險利率，假設年化2%，轉換為日利率

# 計算超額市場回報
data['excess_market_returns'] = data['market_returns'] - Rf

# 5. 定義回歸模型，y = 股票的超額回報，X = 市場的超額回報
X = data['excess_market_returns'].values.reshape(-1, 1)
y = (data['stock_returns'] - Rf).values

# 建立線性回歸模型並訓練
model = LinearRegression()
model.fit(X, y)

# 取得 beta 值
beta = model.coef_[0]
print(f"Estimated Beta: {beta}")

# 6. 畫出回歸線
plt.scatter(data['excess_market_returns'], y, color='blue', label='Data points')
plt.plot(data['excess_market_returns'], model.predict(X), color='red', label='Regression Line')
plt.xlabel('Excess Market Returns')
plt.ylabel('Excess Stock Returns')
plt.legend()
plt.show()