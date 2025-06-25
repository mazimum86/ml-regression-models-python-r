# ===========================
# 🌳 Decision Tree Regression
# ===========================

# 📂 Import Dataset
dataset <- read.csv('../data/Position_Salaries.csv')
dataset <- dataset[, 2:3]  # Keep only Level and Salary columns

# 🧠 Fit Decision Tree Regression Model
library(rpart)
regressor <- rpart(
  formula = Salary ~ ., 
  data = dataset,
  control = rpart.control(minsplit = 2)
)

# 🔍 Predict Salary for Level = 6.5
y_pred <- predict(regressor, newdata = data.frame(Level = 6.5))
cat("💰 Predicted Salary for Level 6.5:", y_pred, "\n")

# ===========================
# 📊 Visualize Decision Tree Fit
# ===========================
library(ggplot2)

ggplot() +
  geom_point(
    aes(x = dataset$Level, y = dataset$Salary),
    color = "red"
  ) +
  geom_line(
    aes(x = dataset$Level, y = predict(regressor, newdata = dataset)),
    color = "blue"
  ) +
  labs(
    title = "Decision Tree Regression",
    x = "Level",
    y = "Salary"
  ) +
  theme_minimal()

# ===========================
# 📊 High-Resolution Plot
# ===========================
X_grid <- seq(min(dataset$Level), max(dataset$Level), 0.001)

ggplot() +
  geom_point(
    aes(x = dataset$Level, y = dataset$Salary),
    color = "red"
  ) +
  geom_line(
    aes(x = X_grid, y = predict(regressor, newdata = data.frame(Level = X_grid))),
    color = "blue"
  ) +
  labs(
    title = "Decision Tree Regression (High Resolution)",
    x = "Level",
    y = "Salary"
  ) +
  theme_minimal()
