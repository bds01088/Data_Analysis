import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error

data = datasets.load_breast_cancer()
x = data.data
y = data.target


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2)

clf = DecisionTreeClassifier()
clf.fit(xtrain, ytrain)

ypred = clf.predict(xtest)

# print ('confusion_matrix')
# print (confusion_matrix(ytest, ypred))

# print ('accuracy')
# print (accuracy_score(ytest, ypred, normalize = True))

# print ('classification report')
# print (classification_report(ytest, ypred))

# print ('auc')
# print (roc_auc_score(ytest, ypred))

# print ('mean squared error')
# print (mean_squared_error(ytest, ypred))


skf = StratifiedKFold(n_splits=10)
skf.get_n_splits(x,y)
print (skf)

# for train_index, test_index in skf.split(x, y) :
#     print ('train set : ', train_index)
#     print ('test set : ', test_index)

scores = cross_val_score(clf, x, y, cv = skf)
print ('K fold cross validation scores')
print (scores)
print ('average accuracy')
print (scores.mean())

skfshf = StratifiedKFold(n_splits=10, shuffle=True)
skfshf.get_n_splits(x, y)
print (skfshf)

shfscores = cross_val_score(clf, x, y, cv = skfshf)
print ('K fold cross validation scores with shuffle')
print (shfscores)
print ('average accuracy')
print (shfscores.mean())

