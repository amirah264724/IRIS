import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
iris=sns.load_dataset('iris')
iris.head()

X_iris = iris.drop('species', axis=1)  
y_iris = iris['species']

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris, random_state = 0)
clf = SVC(kernel='rbf', C=1).fit(Xtrain, ytrain)

print('Iris dataset')
print('Accuracy of RBF SVC classifier on training set: {:.2f}'
     .format(clf.score(Xtrain, ytrain)))
print('Accuracy of RBF SVC classifier on test set: {:.2f}'
     .format(clf.score(Xtest, ytest)))

#Confusion matrix SVM:
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC
svm = SVC(random_state=42, kernel='linear')

# Fit the data to the SVM classifier
svm = svm.fit(Xtrain, ytrain)

# Evaluate by means of a confusion matrix
matrix = plot_confusion_matrix(svm, Xtest, ytest, cmap=plt.cm.Blues, normalize='true')
plt.title('Confusion matrix for linear SVM')
fig = plt.figure(figsize=(10, 4))
plt.show(matrix)
plt.show()
