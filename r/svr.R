# Support Vector Regression

# Import dataset
dataset <- read.csv('../data/Position_Salaries.csv')
dataset <- dataset[, 2:3]

# Fit the SVR model
library(e1071)
regressor <- svm(Salary ~ ., data = dataset, type = 'eps-regression')

# Predict result for level 6.5
y_pred <- predict(regressor, newdata = data.frame(Level = 6.5))
cat("💰 Predicted salary for level 6.5:", y_pred, "\n")

# Base scatter plot with original data
library(ggplot2)

# Visualizing the SVR results (original resolution)
ggplot() +
  geom_point(data = dataset, aes(x = Level, y = Salary), color = 'red', size = 2) +
  geom_line(data = dataset, aes(x = Level, y = predict(regressor, newdata = dataset)), color = 'blue', size = 1) +
  labs(
    title = "Salary vs Level (Support Vector Regression)",
    x = "Level",
    y = "Salary"
  ) +
  theme_minimal()

# === High-resolution version ===
# Create a finer grid for smooth plotting
X_grid <- seq(min(dataset$Level), max(dataset$Level), by = 0.01)
y_pred_grid <- predict(regressor, newdata = data.frame(Level = X_grid))

# Plot the high-resolution curve
ggplot() +
  geom_point(data = dataset, aes(x = Level, y = Salary), color = 'red', size = 2) +
  geom_line(aes(x = X_grid, y = y_pred_grid), color = 'blue', size = 1) +
  labs(
    title = "Salary vs Level (SVR - High Resolution)",
    x = "Level",
    y = "Salary"
  ) +
  theme_minimal()

