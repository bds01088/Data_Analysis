from sklearn import linear_model
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
matplotlib.style.use('ggplot')
data = {'x' : [13, 19, 16, 14, 15, 14], 'y' : [40,83,62,48,58,43]}
data = pd.DataFrame(data)


linear_regression = linear_model.LinearRegression()
linear_regression.fit(X = pd.DataFrame(data["x"]), y = data["y"])
prediction = linear_regression.predict(X = pd.DataFrame(data["x"]))
print(linear_regression.intercept_)
print(linear_regression.coef_)

residuals = data["y"] - prediction
print(residuals.describe())

SSE = (residuals**2).sum()
SST = ((data["y"]-data["y"].mean())**2).sum()
R_squared = 1-(SSE/SST)
print("R_squared = ", R_squared)

#그래프 출력
data.plot(kind="scatter", x='x', y='y', figsize=(5,5), color = "black")
plt.plot(data["x"], prediction, color = "blue")
plt.show()

#성능평가
print('Score = ', linear_regression.score(X = pd.DataFrame(data["x"]), y = data["y"]))
print('MSE = ', mean_squared_error(prediction, data['y']))
print('MSE = ', mean_squared_error(prediction, data['y'])**0.5)