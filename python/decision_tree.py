# ===========================
# 🌳 Decision Tree Regression
# ===========================

# 📦 Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# 📥 Load Dataset
dataset = pd.read_csv('../data/Position_Salaries.csv')
dataset = dataset.iloc[:, 1:]  # Extract Level and Salary columns
X = dataset.iloc[:, [0]].values
y = dataset.iloc[:, -1].values

# 🧠 Fit the Decision Tree model
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# 🔍 Predict the salary for level 6.5
y_pred = regressor.predict([[6.5]])
print(f"💰 Predicted Salary for level 6.5: ${y_pred[0]:,.2f}")

# ===========================
# 📊 Visualize Decision Tree Fit
# ===========================

# 🔵 Basic scatter + regression
plt.scatter(X, y, color='red', label='Actual')
plt.plot(X, regressor.predict(X), color='blue', label='Predicted')
plt.title('Decision Tree Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 🔵 High-resolution curve
X_grid = np.arange(X.min(), X.max(), 0.01).reshape(-1, 1)  # Finer resolution
plt.scatter(X, y, color='red', label='Actual')
plt.plot(X_grid, regressor.predict(X_grid), color='blue', label='Predicted')
plt.title('Decision Tree Regression (High Resolution)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ✅ Done
print("✅ Decision Tree Regression complete. Plots displayed successfully!")
