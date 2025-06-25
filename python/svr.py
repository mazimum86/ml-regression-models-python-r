# Support Vector Regression (SVR)

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Load Dataset
dataset = pd.read_csv('../data/Position_Salaries.csv')
dataset = dataset.iloc[:, 1:]  # Remove the first column (Position)

# Extract Features and Target
X = dataset.iloc[:, [0]].values  # Level
y = dataset.iloc[:, -1].values   # Salary

# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()

X_scaled = sc_X.fit_transform(X)
y_scaled = sc_y.fit_transform(y.reshape(-1, 1)).ravel()  # Flatten to 1D

# Fit the SVR Model
regressor = SVR(kernel='rbf')
regressor.fit(X_scaled, y_scaled)

# Predict a new result (e.g., for Level 6.5)
level_input = [[6.5]]
scaled_input = sc_X.transform(level_input)
scaled_prediction = regressor.predict(scaled_input)
prediction = sc_y.inverse_transform(scaled_prediction.reshape(-1, 1))

print(f"💰 Predicted Salary for Level 6.5: ${prediction[0, 0]:,.2f}")

# Visualize SVR Results
plt.figure(figsize=(8, 5))
plt.scatter(X_scaled, y_scaled, color='red', label='Original Data')
plt.plot(X_scaled, regressor.predict(X_scaled), color='blue', label='SVR Model')
plt.title('Support Vector Regression (SVR)')
plt.xlabel('Level (scaled)')
plt.ylabel('Salary (scaled)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualize SVR in High Resolution
X_grid = np.linspace(X_scaled.min(), X_scaled.max(), 100).reshape(-1, 1)

plt.figure(figsize=(8, 5))
plt.scatter(X_scaled, y_scaled, color='red', label='Original Data')
plt.plot(X_grid, regressor.predict(X_grid), color='blue', label='SVR High Res')
plt.title('Support Vector Regression (High Resolution)')
plt.xlabel('Level (scaled)')
plt.ylabel('Salary (scaled)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
