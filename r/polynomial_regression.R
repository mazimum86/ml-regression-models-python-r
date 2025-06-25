# ==================================================
# 📈 Polynomial Regression: Salary vs Level in R
# Author: Chukwuka Chijioke Jerry
# ==================================================

# 1. Load Required Libraries
library(ggplot2)

# 2. Import Dataset
dataset <- read.csv('../data/Position_Salaries.csv')
dataset <- dataset[ , -1]  # Remove 'Position' column if not needed

# 3. Fit Linear Regression Model
lin_reg <- lm(formula = Salary ~ Level, data = dataset)
summary(lin_reg)

# 4. Fit Polynomial Regression Model (Degree 4)
dataset$Level2 <- dataset$Level^2
dataset$Level3 <- dataset$Level^3
dataset$Level4 <- dataset$Level^4
poly_reg <- lm(formula = Salary ~ Level + Level2 + Level3 + Level4, data = dataset)
summary(poly_reg)

# 5. Visualize Linear Regression Results
ggplot(dataset, aes(x = Level, y = Salary)) +
  geom_point(color = 'blue', size = 2) +
  geom_line(aes(y = predict(lin_reg, newdata = dataset)), color = 'red', linewidth = 1.2) +
  labs(title = 'Salary vs Level (Linear Regression)',
       x = 'Level', y = 'Salary') +
  theme_minimal()

# 6. Visualize Polynomial Regression Results
ggplot(dataset, aes(x = Level, y = Salary)) +
  geom_point(color = 'blue', size = 2) +
  geom_line(aes(y = predict(poly_reg, newdata = dataset)), color = 'red', linewidth = 1.2) +
  labs(title = 'Salary vs Level (Polynomial Regression)',
       x = 'Level', y = 'Salary') +
  theme_minimal()

# 7. Predict Salary for Level 6.5 using Linear Regression
linear_pred <- predict(lin_reg, newdata = data.frame(Level = 6.5))
cat("Linear Regression Prediction for Level 6.5: $", round(linear_pred, 2), "\n")

# 8. Predict Salary for Level 6.5 using Polynomial Regression
poly_pred <- predict(poly_reg, newdata = data.frame(
  Level = 6.5,
  Level2 = 6.5^2,
  Level3 = 6.5^3,
  Level4 = 6.5^4
))
cat("Polynomial Regression Prediction for Level 6.5: $", round(poly_pred, 2), "\n")

# 9. Visualize Polynomial Regression with Higher Resolution
X_grid <- seq(min(dataset$Level), max(dataset$Level), by = 0.01)
high_res_data <- data.frame(
  Level = X_grid,
  Level2 = X_grid^2,
  Level3 = X_grid^3,
  Level4 = X_grid^4
)

ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), color = 'blue', size = 2) +
  geom_line(aes(x = X_grid, y = predict(poly_reg, newdata = high_res_data)),
            color = 'red', linewidth = 1.2) +
  labs(title = 'Polynomial Regression-R (High Resolution)',
       x = 'Level', y = 'Salary') +
  theme_minimal()
