from sklearn.datasets import load_iris

iris = load_iris()

# print (iris.keys())
# print (iris.feature_names)

# print (iris.data)
X = iris['data']
y = iris['target']

from sklearn.model_selection import train_test_split
xtr, xte, ytr, yte = train_test_split(X,y)

from sklearn.preprocessing import StandardScaler
#정규함수로 바꾸는거
scaler = StandardScaler()

scaler.fit(xtr)
xtr = scaler.transform(xtr)
xte = scaler.transform(xte)

#다층인공신경망
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10,10,10))
mlp.fit(xtr, ytr)

pred = mlp.predict(xte)

from sklearn.metrics import classification_report, confusion_matrix

print (confusion_matrix(yte, pred))
print (classification_report(yte, pred))