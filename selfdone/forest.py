from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

iris = load_iris()

#training
# xtr = iris.data[:-30]
# ytr = iris.target[:-30]
#testing
# xte = iris.data[-30:]
# yte = iris.target[-30:]
x = iris.data
y = iris.target
xtr, xte, ytr, yte = train_test_split(x, y, test_size = 0.2)

# rfc = RandomForestClassifier(n_estimators=10)
rfc = RandomForestClassifier(n_estimators= 200, max_features = 4, oob_score = True)


rfc.fit(xtr, ytr)
pred = rfc.predict(xte)
# print (pred == yte)
# print (rfc.score(xte, yte))
print ("Accuracy is : ", accuracy_score(pred, yte))
print (classification_report(pred, yte))

#속성 중요도
for feature, imp in zip(iris.feature_names, rfc.feature_importances_):
    print (feature, imp)
