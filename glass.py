import pandas as pd
import numpy as np

glass = pd.read_csv("C:/Users/SHAJIUDDIN MOHAMMED/Desktop/glass.csv")

glass = glass.iloc[:, 1:11] # Excluding id column
glass.columns

glass = glass.iloc[:, [8,0,1, 2, 3,4,5,6,7]]
glass.columns

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
glass_n = norm_func(glass.iloc[:, 1:])
glass_n.describe()

X = np.array(glass_n.iloc[:,:]) # Predictors 
Y = np.array(glass['Type']) # Target 

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 11)
knn.fit(X_train, Y_train)

pred = knn.predict(X_test)
pred

# Evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, pred))
pd.crosstab(Y_test, pred, rownames = ['Actual'], colnames= ['Predictions']) 
# 0.46511627906976744 with k = 9
# 0.5813953488372093 with k = 38
# 0.627906976744186 with k = 7
# 0.6976744186046512 with k = 8

# error on train data
pred_train = knn.predict(X_train)
print(accuracy_score(Y_train, pred_train))
pd.crosstab(Y_train, pred_train, rownames=['Actual'], colnames = ['Predictions']) 
# 0.7368421052631579 with k = 9
# 0.5672514619883041 with k = 38
# 0.7076023391812866 with k = 7
# 0.7251461988304093 with k = 8

# creating empty list variable 
acc = []

# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values

for i in range(1,20,2):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, Y_train)
    train_acc = np.mean(neigh.predict(X_train) == Y_train)
    test_acc = np.mean(neigh.predict(X_test) == Y_test)
    acc.append([train_acc, test_acc])


import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(1,20,2),[i[0] for i in acc],"ro-")

# test accuracy plot
plt.plot(np.arange(1,20,2),[i[1] for i in acc],"bo-")

# Conclusion
# As per the above analysis the higher we take the 'k' value the lesser the accuracy.
# So the best model is considered when the 'k' value is 8
