install.packages('mlr3')
library(mlr3)
# Simulate Data

## In the graph below the green line represents the noise free part of 𝑦 as a 
## function of the possible values of 𝑥 and the blue points consist of 60 noisy 
## observations 𝑦 scattered evenly across the range of 𝑥.

set.seed(1)

x=seq(0,1,length=60)
f = sin(2*pi*x)
y = rnorm(60,f,0.3)

library("RColorBrewer")

colours<-brewer.pal(n = 12, name = "Paired")
plot(x,f,xlab='x',ylab='f',col=colours[4],type='l',ylim=c(-1.3,1.3))
points(x,y, col=colours[2])

# ML Models in mlr3

install.packages('mlr3verse')
library(mlr3verse)
library(mlr3learners)

### Define a task, where we essentially provide 1.the data, 
### 2.the machine learning type and 3.the loss function.

## Linear
data1 = data.frame(x,y) ### data
task1 <- TaskRegr$new('poly1', backend = data1, target = 'y') ### ML Type
measure <- msr('regr.mse') ### Loss function ### General

## Holdout method 
set.seed(1) ### General
train_set <- seq(from=1,to=60,by=4) ### General
test_set <- setdiff(seq_len(task1$nrow), train_set) ### General

## ML Model
learner_lm <- lrn('regr.lm') ### General
learner_lm$train(task1, row_ids = train_set)
pol1 = learner_lm$predict(task1, row_ids = test_set)$response

## Visualisation 
plot(x,f,xlab='x',ylab='f',col=colours[4],ylim=c(-1.3,1.3),type='l')
lines(x[test_set],pol1,col=colours[6])
points(x,y, col=colours[2])

## MSE
mse1 = learner_lm$predict(task1, row_ids = test_set)$score()
mse1

## Polynomial (3 order)
data3 = data.frame(x,x^2,x^3,y)
task3 <- TaskRegr$new('poly3', backend=data3, target = 'y')

learner_lm$train(task3, row_ids = train_set)
pol3 = learner_lm$predict(task3, row_ids = test_set)$response

plot(x,f,xlab='x',ylab='f',col=colours[4],ylim=c(-1.3,1.3),type='l')
lines(x[test_set],pol3,col=colours[6])
points(x,y, col=colours[2])

mse3 = learner_lm$predict(task3, row_ids = test_set)$score()
mse3

# Cross validation - Specifically K-fold CV
## 3-Fold CV

### Linear 
resampling = rsmp("cv", folds = 3) ### General

rr1 = resample(task1, learner_lm, resampling)

rr1$aggregate(measure)

### Polynomial (3 order)
rr3 = resample(task3, learner_lm, resampling)

rr3$aggregate(measure)

# Underfitting and overfitting

data9 = data.frame(x,x^2,x^3,x^4,x^5,x^6,x^7,x^8,x^9,y)
task9 <- TaskRegr$new('poly9', backend=data9, target = 'y')

learner_lm$train(task9, row_ids = train_set)
pol9 = learner_lm$predict(task9, row_ids = test_set)$response

plot(x,f,xlab='x',ylab='f',col=colours[4],ylim=c(-1.3,1.3),type='l')
lines(x[test_set],pol9,col=colours[6])
points(x,y, col=colours[2])

rr9 = resample(task9, learner_lm, resampling)
rr9$aggregate(measure)

# Benchmarking 

design = benchmark_grid(tasks = c(task1, task3, task9),
                        learners = lrns("regr.lm"),
                        resamplings = rsmps("cv", folds = 3))

bmr = benchmark(design)
bmr$aggregate(measure)

### Visualised
install.packages('ggplot2')
install.packages('mlr3viz')
library('mlr3viz')
library('ggplot2')
p <- autoplot(bmr) + ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))
p

## Optional Activity: Repeat the above for polynomials 
## with order 2,4,5,6,7, and/or 8 and compare.

## Polynomial order 2
data2 = data.frame(x,x^2,y)
task2 <-TaskRegr$new('poly2', backend = data2, target = 'y')
learner_lm$train(task2, row_ids = train_set)
pol2 = learner_lm$predict(task2,row_ids = test_set)$response

rr2 = resample(task2, learner_lm, resampling)
rr2$aggregate(measure)

## Polynomial order 4
data4 = data.frame(x,x^2,x^3,x^4,y)
task4 = TaskRegr$new('poly4', backend = data4, target = 'y')
learner_lm$train(task4, row_ids = train_set)
pol4 = learner_lm$predict(task4, row_ids = test_set)$response

rr4 = resample(task4, learner_lm, resampling)
rr4$aggregate(measure)

## Polynomial order 5
data5 = data.frame(x,x^2,x^3,x^4,x^5,y)
task5 = TaskRegr$new('poly5', backend = data5, target = 'y')
learner_lm$train(task5, row_ids = train_set)
pol5 = learner_lm$predict(task5, row_ids = test_set)$response

rr5 = resample(task5, learner_lm, resampling)
rr5$aggregate(measure)

## Polynomial order 6
data6 = data.frame(x,x^2,x^3,x^4,x^5,x^6,y)
task6 = TaskRegr$new('poly6', backend = data6, target = 'y')
learner_lm$train(task6, row_ids = train_set)
pol6 = learner_lm$predict(task6, row_ids = test_set)$response

rr6 = resample(task6, learner_lm, resampling)
rr6$aggregate(measure)

## Polynomial order 7
data7 = data.frame(x,x^2,x^3,x^4,x^5,x^6,x^7,y)
task7 = TaskRegr$new('poly7', backend = data7, target = 'y')
learner_lm$train(task7, row_ids = train_set)
pol7 = learner_lm$predict(task7, row_ids = test_set)$response

rr7 = resample(task7, learner_lm, resampling)
rr7$aggregate(measure)

## Polynomial order 8
data8 = data.frame(x,x^2,x^3,x^4,x^5,x^6,x^7,x^8,y)
task8 = TaskRegr$new('poly8', backend = data8, target = 'y')
learner_lm$train(task8, row_ids = train_set)
pol8 = learner_lm$predict(task8, row_ids = test_set)$response

rr8 = resample(task8, learner_lm, resampling)
rr8$aggregate(measure)

## Benchmarking (2,4,5,6,7,8)

design2 = benchmark_grid(tasks = c(task2, task4, task5, task6, task7, task8),
                        learners = lrns("regr.lm"),
                        resamplings = rsmps("cv", folds = 3))

bmr = benchmark(design2)
bmr$aggregate(measure)

p2 <- autoplot(bmr) + ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))
p2

### The winner is poly5 with an error of 0.0724.Error starts rising from poly 6 onwards


