import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib as plt
import pandas as pd
iris=sns.load_dataset('iris')
iris.head()

X_iris = iris.drop('species', axis=1)  
y_iris = iris['species']

from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris,random_state=0)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(Xtrain, ytrain)



clf.fit(Xtrain, ytrain)

tree.plot_tree(clf.fit(Xtrain, ytrain) )

clf.score(Xtest, ytest)

#Confusion matrix SVM:
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC
svm = SVC(random_state=42, kernel='linear')

# Fit the data to the SVM classifier
svm = svm.fit(xtrain, ytrain)

# Evaluate by means of a confusion matrix
matrix = plot_confusion_matrix(svm, xtest, ytest, cmap=plt.cm.Blues, normalize='true')
plt.title('Confusion matrix for linear SVM')
plt.show(matrix)
plt.show()
