install.packages('RColorBrewer')
library('RColorBrewer')
display.brewer.all(colorblindFriendly = TRUE)
colours <- brewer.pal(n = 12, name = "Paired")

set.seed(2) #basically assigns set of possiblke variable values

x = seq(0,1, length = 1000)
f = sin(2*pi*x)

x0 = seq(0,1, length = 15)
f0 = sin(2*pi*x0)

y = rnorm(15, f0, 0.3)

png("ML_Plot_1.png", width = 800, height = 600, res = 100)
plot(x,f,xlab = 'x', ylab = 'f', col = colours[4], type = 'l')
points(x0,y,col=colours[2])
dev.off()

### In real life, we do not know the noise-free version of 𝑦 is 𝑓(𝑥)=𝑠𝑖𝑛(2𝜋𝑥) (green line). 
### One of the main goals is to estimate the function with a model using the noisy observations. 
### Having an estimate of the function will allow us to predict the value 𝑦 for any given value of 𝑥.

### The type (task) is regression (supervised learning)
### The data consist of xo and y.

## The Model- Least squares estimator

fit1 = lm(y~x0)
pol1 <- function(x) fit1$coefficients[2]*x + fit1$coefficients[1]

png("ML_Plot_2.png", width = 800, height = 600, res = 100)
plot(x,f,xlab = 'x',ylab = 'f', col = colours[4], type = 'l')
lines(x,pol1(x),col=colours[6])
points(x0,y, col=colours[2])
dev.off()

## Loss function 
### To check the performance of the linear model we can measure how ‘far’ are its 
### predictions from the unknown noise free part of 𝑦 (green line).

### This is done by the mean squared error (MSE) that squares all these deviations and then 
### averages them.

mse1 = mean((f-poll(x))^2)
mse1

## Repeat the previous process with a 3rd degree polynomial.
fit3 = lm(y~x0+I(x0^2)+I(x0^3))
pol3 <- function(x) fit3$coefficients[4]*x^3 + fit3$coefficients[3]*x^2 + fit3$coefficients[2]*x + fit3$coefficients[1]

png("ML_Plot_3.png", width = 800, height = 600, res = 100)
plot(x,f,xlab = 'x', ylab = 'f', col = colours[4], type = 'l')
lines(x,pol3(x),col=colours[6])
points(x0,y, col=colours[2])
dev.off()

mse3 = mean((f-pol3(x))^2)
mse3

## Optional activity 1
### "Polynomials of degree from 1 to 8"
### 8-degree polynomial regression

fit8 = lm(y~x0+I(x0^2)+I(x0^3)+I(x0^4)+I(x0^5)+I(x0^6)+I(x0^7)+I(x0^8))

pol8 <- function(x) fit8$coefficients[9]*x^8 + fit8$coefficients[8]*x^7 +
  fit8$coefficients[7]*x^6 + fit8$coefficients[6]*x^5 + fit8$coefficients[5]*x^4 +
  fit8$coefficients[4]*x^3 + fit8$coefficients[3]*x^2 + fit8$coefficients[2]*x + fit8$coefficients[1]   

png("ML_Plot_4.png", width = 800, height = 600, res = 100)
plot(x,f,xlab = 'x', ylab = 'f', col = colours[4], type = 'l')
lines(x,pol8(x),col=colours[6])
points(x0,y, col=colours[2])
dev.off()

mse8 = mean((f-pol8(x))^2)
mse8

### overfitting & variance 




