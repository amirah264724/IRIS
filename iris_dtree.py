import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib as plt
import pandas as pd
from sklearn.datasets import load_iris
iris=sns.load_dataset('iris')
X_iris = iris.drop('species', axis=1)  
y_iris = iris['species']

from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris,random_state=1)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(Xtrain, ytrain)



clf.fit(Xtrain, ytrain)

tree.plot_tree(clf.fit(Xtrain, ytrain) )

clf.score(Xtest, ytest)
