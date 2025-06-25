# ============================================
# 📊 Random Forest Regression
# Author: Chukwuka Chijioke Jerry
# ============================================

# ===========================
# 📂 Load Dataset
# ===========================
dataset <- read.csv('../data/position_salaries.csv')
dataset <- dataset[2:3]  # Keep only Level and Salary columns

# ===========================
# 🌳 Fit Random Forest Model
# ===========================
library(randomForest)
set.seed(1234)

# Fit Random Forest with 500 trees
regressor <- randomForest(
  x = dataset[1],           # Level as feature
  y = dataset$Salary,       # Salary as target
  ntree = 500
)

# ===========================
# 🔍 Prediction
# ===========================
y_pred <- predict(
  regressor,
  newdata = data.frame(Level = 6.5)
)
cat("💰 Predicted Salary at level 6.5:", y_pred, "\n")

# ===========================
# 📈 Visualization (Training data)
# ===========================
library(ggplot2)

ggplot() +
  geom_point(
    aes(x = dataset$Level, y = dataset$Salary),
    color = 'red'
  ) +
  geom_line(
    aes(
      x = dataset$Level,
      y = predict(regressor, newdata = dataset)
    ),
    color = 'blue'
  ) +
  labs(
    title = 'Random Forest Regression',
    x = 'Level',
    y = 'Salary'
  ) +
  theme_minimal()

# ===========================
# 📊 High-Resolution Plot
# ===========================
X <- seq(min(dataset$Level), max(dataset$Level), 0.01)

ggplot() +
  geom_point(
    aes(x = dataset$Level, y = dataset$Salary),
    color = 'red'
  ) +
  geom_line(
    aes(
      x = X,
      y = predict(
        regressor,
        newdata = data.frame(Level = X)
      )
    ),
    color = 'blue'
  ) +
  labs(
    title = 'Random Forest Regression (High Resolution)',
    x = 'Level',
    y = 'Salary'
  ) +
  theme_minimal()
