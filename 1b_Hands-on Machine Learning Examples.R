
# k - nearest neighbours 
## Classification example 
### The target variable is the wine colour (red or white) while we work 
### with two feature variables indicating the pH and the alcohol of each wine.

data <- read.csv("Wine quality.csv")

y <- data$wine
y <- as.factor(y)
x <- cbind(data$pH,data$alcohol)
table(y)

png("ML_Plot_5.png", width = 800, height = 600, res = 100)
colours2<-colours[c(6,2)]
plot(x[,1],x[,2],col=colours2[y],xlab='pH',ylab='alcohol')
legend('topright', legend=c('red wine','white wine'), col= colours2, pch = 19)
dev.off()

## Seeing the classification as suggested by the k-nearest neighbours model rather than the actual data 𝑦.
install.packages('caret')
install.packages('ggplot2')
install.packages('lattice')
library(ggplot2)
library(lattice)
library(caret)

knn_model <- knn3(x, y, k = 1)
knn_pred <- predict(knn_model, x, type = "class")

png("ML_Plot_6.png", width = 800, height = 600, res = 100)
colours2<-colours[c(6,2)]
plot(x[,1],x[,2],col=colours2[knn_pred],xlab='pH',ylab='alcohol')
legend('topright', legend=c('red wine','white wine'), col= colours2, pch = 19)
dev.off()

### visually very similar but it would be better to check a more objective 
### criterion such as the accuracy, that simply gives the proportion of correctly classified points.

confusionMatrix(knn_pred, y)

### The accuracy is 0.8684, meaning that almost 87% of the points were correctly classified.

## Optional Activity 2: Repeat with a different value of k.

knn_model2 <- knn3(x, y, k = 5)
knn_pred2 <- predict(knn_model, x, type = "class")

png("ML_Plot_7.png", width = 800, height = 600, res = 100)
colours2<-colours[c(6,2)]
plot(x[,1],x[,2],col=colours2[knn_pred2],xlab='pH',ylab='alcohol')
legend('topright', legend=c('red wine','white wine'), col= colours2, pch = 19)
dev.off()

confusionMatrix(knn_pred2, y)
###  Accuracy : 0.8666 

## Optional Activity 3: Pause and try to identify the machine components and 
## the type of machine learning in this example.

### The data: The wine data
### The model: The knn model, k nearest neighbour function 
### Cost function: Classification?,"Misclassification Error Rate". Results shown by confusionmatrix 







