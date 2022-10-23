import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib as plt
import pandas as pd
iris=pd.read_csv('iris.data')
iris.head()
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(X_iris, y_iris, random_state = 0)
clf = SVC(kernel='rbf', C=1).fit(xtrain, ytrain)

st.write('Iris dataset')
st.write('Accuracy of RBF SVC classifier on training set: {:.2f}'
     .format(clf.score(xtrain, ytrain)))
st.write('Accuracy of RBF SVC classifier on test set: {:.2f}'
     .format(clf.score(xtest, ytest)))
     
#Confusion matrix SVM:
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC
svm = SVC(random_state=42, kernel='linear')

# Fit the data to the SVM classifier
svm = svm.fit(X_train, y_train)

# Evaluate by means of a confusion matrix
matrix = plot_confusion_matrix(svm, X_test, y_test, cmap=plt.cm.Blues, normalize='true')
plt.title('Confusion matrix for linear SVM')
plt.show(matrix)
plt.show()
