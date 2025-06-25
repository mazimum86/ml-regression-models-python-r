# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 12:44:59 2025

Random Forest Regression on Position Salaries Dataset
Author: Chukwuka Chijioke Jerry
"""

# ===========================
# 📦 Import Libraries
# ===========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# ===========================
# 📂 Load Dataset
# ===========================
dataset = pd.read_csv('../data/Position_Salaries.csv')
dataset = dataset.iloc[:, 1:]  # Keep only Level and Salary columns
X = dataset.iloc[:, [0]].values
y = dataset.iloc[:, -1].values

# ===========================
# 🌳 Fit Random Forest Regression
# ===========================
regressor = RandomForestRegressor(
    n_estimators=100, random_state=0
)
regressor.fit(X, y)

# ===========================
# 🔍 Prediction
# ===========================
y_pred = regressor.predict([[6.5]])
print(f"💰 Predicted Salary for Level 6.5: {y_pred[0]:.2f}")

# ===========================
# 📊 Visualization
# ===========================
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Random Forest Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.grid(True)
plt.tight_layout()
plt.show()

# ===========================
# 🖼️ High-Resolution Plot
# ===========================
X_grid = np.arange(X.min(), X.max(), 0.001).reshape(-1, 1)

plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Random Forest Regression (High Resolution)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.grid(True)
plt.tight_layout()
plt.show()
