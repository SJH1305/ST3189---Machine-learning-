# Practice assignment 1

## 1. Consider polynomial regression models for the target variable ‘mpg’ based on
## the input variable horsepower. Train these models and provide plots to depict
## the fit of them on the training data.

### Noted: Supervised learning, estimating mpg(y) based on horsepower(x).
### Data - auto.csv (obv)
### Model - Polynomial regression model
### Cost function - 

auto <- read.csv("auto.csv")

library('RColorBrewer')

### The actual function unlike the examples is unknown. Use of Least squares estimation:

## 1 Degree 
#### Defining the variables
y <- auto$mpg
x <- auto$horsepower 
x_sorted <- seq(min(x), max(x), length.out = 100) ### sorts the hp values so the graph follows in order

#### Fitting model, defining polynomial function 
fit1 <- lm(y~x)
pol1 <- function(x) fit1$coefficients[2]*x + fit1$coefficients[1]

png("PA_PLOT_1.png", width = 800, height = 600, res = 100)
plot(x, y, xlab = 'horsepower(x)', ylab = 'mpg(y)', col = colours[4])
lines(x_sorted, pol1(x_sorted), col = colours[6])
dev.off()

## 2 Degree
### We sort x so the 'lines' command draws a smooth curve instead of a scribble

fit2 <- lm(y~x + I(x^2))
pol2 <- function(x) fit2$coefficients[3]*x^2 + fit2$coefficients[2]*x + fit2$coefficients[1]

png("PA_PLOT_2.png", width = 800, height = 600, res = 100)
plot(x, y, xlab = 'horsepower(x)', ylab = 'mpg(y)', col = colours[4])
lines(x_sorted, pol2(x_sorted), col = colours[6])
dev.off()

## 5 degree
fit5 <- lm(y~x + I(x^2) + I(x^3) + I(x^4) +I(x^5))
pol5 <- function(x) fit5$coefficients[6]*x^5 + fit5$coefficients[5]*x^4 +
  fit5$coefficients[4]*x^3 + fit5$coefficients[3]*x^2 + fit5$coefficients[2]*x + fit5$coefficients[1]

png("PA_PLOT_3.png", width = 800, height = 600, res = 100)
plot(x, y, xlab = 'horsepower(x)', ylab = 'mpg(y)', col = colours[4])
lines(x_sorted, pol5(x_sorted), col = colours[6])
dev.off()

## 2. Compare polynomial regression models of different order and identify the best
## among them in terms of test error. Use cross-validation for the estimation of the
## test error.

### Thought process: Polynomial regression model with cross validation followed by benchmarking
### to assess performance in terms of test error. 

install.packages('mlr3verse')
library(mlr3verse)
library(mlr3learners)

### Reestablish
y <- auto$mpg
x <- auto$horsepower 

## Linear, 1 degree
data1 = data.frame(x,y)
task1 <-TaskRegr$new('poly1', backend = data1, target = 'y')

#### learner_lm$train(task1, row_ids = train_set) # Holdout method
#### pol1 = learner_lm$predict(task1,row_ids = test_set)$response

rr1 = resample(task1, learner_lm, resampling) #### cv
rr1$aggregate(measure)

## Polynomial order 2
data2 = data.frame(x,x^2,y)
task2 <-TaskRegr$new('poly2', backend = data2, target = 'y')

rr2 = resample(task2, learner_lm, resampling)
rr2$aggregate(measure)

## Polynomial order 3
data3 = data.frame(x,x^2,x^3,y)
task3 <-TaskRegr$new('poly3', backend = data3, target = 'y')

rr3 = resample(task3, learner_lm, resampling)
rr3$aggregate(measure)

## Polynomial order 4
data4 = data.frame(x,x^2,x^3,x^4,y)
task4 = TaskRegr$new('poly4', backend = data4, target = 'y')

rr4 = resample(task4, learner_lm, resampling)
rr4$aggregate(measure)

### Compiled answers without benchmarking 
rr1$aggregate(measure)
rr2$aggregate(measure)
rr3$aggregate(measure)
rr4$aggregate(measure)

### The results indicate the 2-degree poly regression model is the best estimator
### with the linear model underfitting and 3 and 4 overfitting; indicating the 
### expected start of fitting noise. 
### This makes sense based on the previous plots visualised and the single 'u' shape of 2 degree polynomials.

## 3. Apply the k-nearest-neighbours model to with `horsepower’ and `weight’ as
## inputs and ‘origin’ as target. Split the data into a train and test sample, train
## models with different values of k, extract predictions from these models and
## compare them graphically on the test data as done in the lecture notes
## (scatterplots of ‘horsepower’ and ‘weight’ with the points labelled by their true
## label or the predicted labels from the k-nearest-neighbours models).

### Making a model to determine origin based on hp and weight

install.packages("caTools") 
install.packages("class") 
install.packages("ggplot2")

library(caTools) 
library(class)
library(ggplot2)

### geekforgeeks.org

### ensure target is a factor. Its given in the question but this is how to check for future reference
### auto$origin <- as.factor(auto$origin)

set.seed(5)

unique(train_cl$origin) # all unique values of 'origin' variable. (1,2,3)

### create split of test and train data
split <- sample.split(auto$origin, SplitRatio = 0.7) ### split not suggested so template is followed
train_cl <- subset(auto, split == "TRUE") 
test_cl <- subset(auto, split == "FALSE") 

train_scale <- scale(train_cl[, c("horsepower","weight")]) 
test_scale <- scale(test_cl[, c("horsepower", "weight")])

### fit model
### k=1
classifier_knn1 <- knn(train = train_scale, 
                      test = test_scale, 
                      cl = train_cl$origin, 
                      k = 1) 

classifier_knn5 <- knn(train = train_scale, 
                       test = test_scale, 
                       cl = train_cl$origin, 
                       k = 5) 

classifier_knn50 <- knn(train = train_scale, 
                       test = test_scale, 
                       cl = train_cl$origin, 
                       k = 50) 

classifier_knn100 <- knn(train = train_scale, 
                       test = test_scale, 
                       cl = train_cl$origin, 
                       k = 100) 

## Comparison - Graphical, against 'origin' data

### knn_Origin
png("PA_PLOT_Origin.png", width = 800, height = 600, res = 100)
plot(test_cl$horsepower, test_cl$weight, 
     col = test_cl$origin, #### Each point is coloured based on the cateogry estimated.
     xlab = "horsepower", ylab = "weight",
     main = "Actual Origin Data",
     pch = 19)
dev.off()

### knn 1
png("PA_PLOT_knn1.png", width = 800, height = 600, res = 100)
plot(test_cl$horsepower, test_cl$weight, 
     col = as.numeric(classifier_knn1),
     xlab = "horsepower", ylab = "weight",
     main = "k=1", pch = 19)
dev.off()

### knn 5
png("PA_PLOT_knn5.png", width = 800, height = 600, res = 100)
plot(test_cl$horsepower, test_cl$weight, 
     col = as.numeric(classifier_knn5),
     xlab = "horsepower", ylab = "weight",
     main = "k=5", pch = 19)
dev.off()

### knn 50
png("PA_PLOT_knn50.png", width = 800, height = 600, res = 100)
plot(test_cl$horsepower, test_cl$weight, 
     col = as.numeric(classifier_knn50),
     xlab = "horsepower", ylab = "weight",
     main = "k=50", pch = 19)
dev.off()

### knn 100
png("PA_PLOT_knn100.png", width = 800, height = 600, res = 100)
plot(test_cl$horsepower, test_cl$weight, 
     col = as.numeric(classifier_knn100),
     xlab = "horsepower", ylab = "weight",
     main = "k=100", pch = 19)
dev.off()
