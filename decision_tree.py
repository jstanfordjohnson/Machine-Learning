#!/usr/bin/python3
#CS7641 HW1 by Tian Mi

from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")

import csv
import numpy as np
import pandas as pd
#import graphviz
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import sklearn.tree as tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.externals.six import StringIO  
from IPython.display import display, Image  
import pydotplus
from graphviz import Source

#################################################
#wine quality data set

data = pd.read_csv('data/yeast.csv')
X = data.iloc[:,1:9]
y = data.iloc[:,9]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)
features = list(X_train.columns.values)
print(X_train.head())

#Decision Tree classifier
#decision tree learning curve of tree depth 5
list1=[]
list2=[]
for i in range(1,95):
    clf = DecisionTreeClassifier(random_state=0, criterion='gini', max_depth=5)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=1-i/100)
    clf = clf.fit(X_train, y_train)
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)
    list1.append(accuracy_score(y_train, train_predict))
    list2.append(accuracy_score(y_test, test_predict))
plt.plot(range(len(list2)),list2)
plt.plot(range(len(list1)),list1)
plt.show()

#decision tree learning curve of different function of split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)
list1=[]
list2=[]
for depth in range(3,40):
    clf = DecisionTreeClassifier(random_state=0, criterion='gini', max_depth=depth)
    clf = clf.fit(X_train, y_train)
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)
    list1.append(accuracy_score(y_test, test_predict))
    
    clf = DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=depth)
    clf = clf.fit(X_train, y_train)
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)
    list2.append(accuracy_score(y_test, test_predict))
plt.plot(range(len(list2)),list2)
plt.plot(range(len(list1)),list1)
plt.show()

#choose tree depth of 5 as optimal solution
clf = DecisionTreeClassifier(random_state=0, criterion='gini', max_depth=5)
clf = clf.fit(X_train, y_train)
test_predict = clf.predict(X_test)
print("The prediction accuracy of decision tree is " + str(accuracy_score(y_test, test_predict)))

#visualization of decision tree
dot_data = export_graphviz(clf, out_file=None, 
                         feature_names=features,  
                         class_names=list(map(str, set(y))),  
                         filled=True, rounded=True,  
                         special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png('tree.png')
