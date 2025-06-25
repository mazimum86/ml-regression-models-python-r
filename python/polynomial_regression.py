# ==================================================
# 📈 Polynomial Regression: Salary vs Level
# Language: Python
# Author: Chukwuka Chijioke Jerry
# ==================================================

# 📁 1. Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# 📁 2. Load Dataset
dataset = pd.read_csv('../Data/Position_Salaries.csv')
X = dataset.iloc[:, [1]].values  # Extract Level column as feature matrix
y = dataset.iloc[:, -1].values   # Extract Salary as target

# 🧠 3. Fit Linear Regression Model
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# 🧠 4. Fit Polynomial Regression Model
poly_feat = PolynomialFeatures(degree=3)
X_poly = poly_feat.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

# 📈 5. Visualize Linear Regression Fit
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, lin_reg.predict(X), color='red', label='Linear Fit')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.title('Salary vs Level (Linear Regression)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 📈 6. Visualize Polynomial Regression Fit
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, poly_reg.predict(X_poly), color='red', label='Polynomial Fit')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.title('Salary vs Level (Polynomial Regression)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 📈 7. High-Resolution Polynomial Curve
X_grid = np.linspace(X.min(), X.max(), num=100).reshape(-1, 1)
X_grid_poly = poly_feat.transform(X_grid)
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X_grid, poly_reg.predict(X_grid_poly), color='red', label='High-Res Polynomial Fit')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.title('Salary vs Level- Python (High-Resolution Polynomial Regression)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 🔍 8. Make Predictions
level = 6.5
linear_pred = lin_reg.predict([[level]])[0]
poly_pred = poly_reg.predict(poly_feat.transform([[level]]))[0]

print(f"Linear Prediction for Level {level}: ${linear_pred:,.2f}")
print(f"Polynomial Prediction for Level {level}: ${poly_pred:,.2f}")

# 📊 9. Evaluate Models Using R² Score
r2_linear = r2_score(y, lin_reg.predict(X))
r2_poly = r2_score(y, poly_reg.predict(X_poly))

print(f"Linear Regression R² Score: {r2_linear:.4f}")
print(f"Polynomial Regression R² Score: {r2_poly:.4f}")
