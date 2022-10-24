import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib as plt
import pandas as pd
from sklearn.datasets import load_iris
iris=sns.load_dataset('iris')
x_iris = iris.drop('species', axis=1)  
y_iris = iris['species']

from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x_iris, y_iris,random_state=1)
clf.fit(xtrain, ytrain)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(xtrain, ytrain)

fig=ply.figure(figsize=(10,4))



tree.plot_tree(clf.fit(xtrain, ytrain) )

clf.score(xtest, ytest)
