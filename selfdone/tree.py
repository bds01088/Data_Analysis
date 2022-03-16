from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from IPython.display import Image
import pandas as pd
import numpy as np
import pydotplus
import os

tennis = pd.read_csv('playtennis.csv')
print (tennis)

tennis.Outlook = tennis.Outlook.replace('Sunny', 0)
tennis.Outlook = tennis.Outlook.replace('Overcast', 1)
tennis.Outlook = tennis.Outlook.replace('Rain', 2)

tennis.Temperature = tennis.Temperature.replace('Hot', 3)
tennis.Temperature = tennis.Temperature.replace('Mild', 4)
tennis.Temperature = tennis.Temperature.replace('Cool', 5)

tennis.Humidity = tennis.Humidity.replace('High', 6)
tennis.Humidity = tennis.Humidity.replace('Normal', 7)

tennis.Wind = tennis.Wind.replace('Weak', 8)
tennis.Wind = tennis.Wind.replace('Strong', 9)

tennis.PlayTennis = tennis.PlayTennis.replace('No', 10)
tennis.PlayTennis = tennis.PlayTennis.replace('Yes', 11)

# print (tennis)

X = np.array(pd.DataFrame(tennis, columns=['Outlook', 'Temperature', 'Humidity', 'Wind']))
y = np.array(pd.DataFrame(tennis, columns=['PlayTennis']))

Xtr, Xte, ytr, yte = train_test_split(X, y)

namu = DecisionTreeClassifier()
namu = namu.fit(Xtr, ytr)
namu.pre = namu.predict(Xte)

print (confusion_matrix(yte, namu.pre))
print (classification_report(yte, namu.pre))

os.environ['PATH'] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

fname = tennis.columns.tolist()
fname = fname[0:4]

tname = np.array(['Play No', 'Play Yes'])

dotdata = tree.export_graphviz(namu, out_file = None, feature_names = fname, class_names = tname, filled = True, rounded = True, special_characters = True)
g = pydotplus.graph_from_dot_data(dotdata)

Image(g.create_png())