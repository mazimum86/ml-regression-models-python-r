# Polynomial Regression

# Import dataset 
dataset = read.csv('../data/Position_Salaries.csv')
dataset = dataset[,-1]


# Fit to Linear Regression
lin_reg <- lm(formula = Salary ~ Level, data = dataset )
summary(lin_reg)


# Fit to Polynomial Regression
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4

poly_reg <- lm(formula = Salary ~ ., data= dataset)
summary(poly_reg)

# Visualize Linear Regression
library(ggplot2)

ggplot(dataset, aes(Level, Salary, )) +
  geom_point(aes(x=Level, y= Salary), data = dataset, col='blue') +
  geom_line(aes(x=Level, y= predict(lin_reg,newdata = dataset)), data = dataset, colour='red', )+
  ggtitle('Linear Regression')


# Visualize Polynomial Regression
library(ggplot2)

ggplot(dataset, aes(Level, Salary, )) +
  geom_point(aes(x=Level, y= Salary), data = dataset, col='blue') +
  geom_line(aes(x=Level, y= predict(poly_reg,newdata = dataset)), data = dataset, colour='red', )+
  ggtitle('Polynomial Regression')


# Predicting 6.5
y_pred = predict(poly_reg, newdata= data.frame(Level=6.5,
                                               Level2=6.5^2,
                                               Level3=6.5^3,
                                               Level4=6.5^4))
print(y_pred)
