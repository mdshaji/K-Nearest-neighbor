
### Analyzing the business problem ###

# Output Variable (y) = type of glass
# Input Variables = Na,Mg,Ai,Si,K,Ca,Ba,Fe


### Importing Dataset ###
glass <- read.csv(file.choose())
View(glass)

# Removing unnecesary colum
glass <- glass[ , 2:10]
View(glass)

#Reorder of the columns
glass <- glass[,c(9,1,2,3,4,5,6,7,8)]
View(glass)
attach(glass)

head(glass) # Shows first 6 rows of the dataset
tail(glass) # Shwos last 6 rows of the dataset

# Checking for NA values
sum(is.na(glass))

# Standardise the input columns
standard.features <-scale(glass[,2:9])
glass1 <-cbind(standard.features,glass[1])
View(glass1)
?cbind

#Reorder of the columns
glass <- glass1[,c(9,1,2,3,4,5,6,7,8)]
View(glass)

# Exploratory Data Analysis
summary(glass)

# table of diagnosis
table(glass$Type)

str(glass$Type)

# table or proportions with more informative labels
round(prop.table(table(glass$Type)) * 100, digits = 2)

install.packages('Hmisc')
library(Hmisc)
describe(glass)

install.packages("lattice") # Used for Data Visualization
library(lattice) # USed for dotplot visualisation


#Boxplot Representation

boxplot(glass$Na, col = "dodgerblue4" , main = "Sodium")
boxplot(glass$Ca, col = "dodgerblue4" , main = "Calcium")
boxplot(glass$Mg, col = "dodgerblue4",main = "Magnesium")
boxplot(glass$K, col = "red", horizontal = T,main = "Potassium")

#Histogram Representation

hist(glass$Na, col = 'red',main = "sodium")
hist(glass$Ca,col = 'red',main = "calcium")
hist(glass$Mg,col = 'red',main = "magnesium")
hist(glass$K,col = 'red',main = "potassium")

#Scatter plot for all pairs of variables
plot(glass)

# Correlation Matrix
cor(glass)

# Data Partitioning
n <- nrow(glass)
n1 <- n * 0.8
n2 <- n - n1
train_index <- sample(1:n,n1)
train <- glass[train_index,]
test <- glass[-train_index,]


# create labels for training and test data

glass_train_labels <- glass[train_index, 1]

glass_test_labels <- glass[-train_index, 1]

#---- Training a model on the data ----

# load the "class" library
install.packages("class")
library(class)

glass_test_pred <- knn(train = train, test = test,
                       cl = glass_train_labels, k = 9)


## ---- Evaluating model performance ---- ##
confusion_test <- table(x = glass_test_labels, y = glass_test_pred)
confusion_test

Accuracy <- sum(diag(confusion_test))/sum(confusion_test)
Accuracy  # 0.8139535

# Training Accuracy to compare against test accuracy
glass_train_pred <- knn(train = train, test = train, cl = glass_train_labels, k=9)

confusion_train <- table(x = glass_train_labels, y = glass_train_pred)

confusion_train

Accuracy_train <- sum(diag(confusion_train))/sum(confusion_train)
Accuracy_train

## Improving model performance ----

# use the scale() function to z-score standardize a data frame
glass_z <- as.data.frame(scale(glass[-1]))

# confirm that the transformation was applied correctly
summary(glass_z$area_mean)

# create training and test datasets

glass_train <- glass_z[train_index, 1]

glass_test <- glass_z[-train_index, 1]

# re-classify test cases
glass_test_pred <- knn(train = train, test = test,
                      cl = glass_train_labels, k=7)

confusion_train <- table(x = glass_train_labels, y = glass_train_pred)

confusion_train

Accuracy_train <- sum(diag(confusion_train))/sum(confusion_train)
Accuracy_train

# Create the cross tabulation of predicted vs. actual

install.packages("gmodels")
library(gmodels)
CrossTable(x = glass_test_labels, y = glass_test_pred, prop.chisq=FALSE)


# try several different values of k
train <- glass[train_index,]
test <- glass[-train_index,]

glass_test_pred <- knn(train = train, test = test, cl = glass_train_labels, k=1)
CrossTable(x = glass_test_labels, y = glass_test_pred, prop.chisq=FALSE)

glass_test_pred <- knn(train = train, test = test, cl = glass_train_labels, k=5)
CrossTable(x = glass_test_labels, y = glass_test_pred, prop.chisq=FALSE)

glass_test_pred <- knn(train = train, test = test, cl = glass_train_labels, k=11)
CrossTable(x = glass_test_labels, y = glass_test_pred, prop.chisq=FALSE)

glass_test_pred <- knn(train = train, test = test, cl = glass_train_labels, k=8)
CrossTable(x = glass_test_labels, y = glass_test_pred, prop.chisq=FALSE)

glass_test_pred <- knn(train = train, test = test, cl = glass_train_labels, k=15)
CrossTable(x = glass_test_labels, y = glass_test_pred, prop.chisq=FALSE)



########################################################
pred.train <- NULL
pred.val <- NULL
error_rate.train <- NULL
error_rate.val <- NULL
accu_rate.train <- NULL
accu_rate.val <- NULL
accu.diff <- NULL
error.diff <- NULL

for (i in 1:20) {
  pred.train <- knn(train = train, test = train, cl = glass_train_labels, k = i)
  pred.val <- knn(train = train, test = test, cl = glass_train_labels, k = i)
  error_rate.train[i] <- mean(pred.train!=glass_train_labels)
  error_rate.val[i] <- mean(pred.val != glass_test_labels)
  accu_rate.train[i] <- mean(pred.train == glass_train_labels)
  accu_rate.val[i] <- mean(pred.val == glass_test_labels)  
  accu.diff[i] = accu_rate.train[i] - accu_rate.val[i]
  error.diff[i] = error_rate.val[i] - error_rate.train[i]
}

knn.error <- as.data.frame(cbind(k = 1:20, error.train = error_rate.train, error.val = error_rate.val, error.diff = error.diff))
knn.accu <- as.data.frame(cbind(k = 1:20, accu.train = accu_rate.train, accu.val = accu_rate.val, accu.diff = accu.diff))

library(ggplot2)
errorPlot = ggplot() + 
  geom_line(data = knn.error[, -c(3,4)], aes(x = k, y = error.train), color = "blue") +
  geom_line(data = knn.error[, -c(2,4)], aes(x = k, y = error.val), color = "red") +
  geom_line(data = knn.error[, -c(2,3)], aes(x = k, y = error.diff), color = "black") +
  xlab('knn') +
  ylab('ErrorRate')
accuPlot = ggplot() + 
  geom_line(data = knn.accu[,-c(3,4)], aes(x = k, y = accu.train), color = "blue") +
  geom_line(data = knn.accu[,-c(2,4)], aes(x = k, y = accu.val), color = "red") +
  geom_line(data = knn.accu[,-c(2,3)], aes(x = k, y = accu.diff), color = "black") +
  xlab('knn') +
  ylab('AccuracyRate')

# Plot for Accuracy
plot(knn.accu[, c(4)], type = "b", xlab = "K-Value", ylab = "DifferenceInAccu") 

# Plot for Error
plot(knn.error[, c(4)], type = "b", xlab = "K-Value", ylab = "DifferenceInError") 


