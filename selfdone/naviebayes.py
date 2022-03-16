from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np


tennis = pd.read_csv('playtennis.csv')

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

X = np.array(pd.DataFrame(tennis, columns = ['Outlook', 'Temperature', 'Humidity', 'Wind']))
y = np.array(pd.DataFrame(tennis, columns = ['PlayTennis']))

xtr, xte, ytr, yte = train_test_split(X,y)

gnb = GaussianNB()
gnb = gnb.fit(xtr,ytr)

gnb_pred = gnb.predict(xte)

print(gnb_pred)

#성능 테스트
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

print ('Confusion Matrix')
print (confusion_matrix(yte, gnb_pred))

print ('Classification Report')
print (classification_report(yte, gnb_pred))

fmeasure = round(f1_score(yte, gnb_pred, average = 'weighted'),2)
accuracy = round(accuracy_score(yte, gnb_pred, normalize = True), 2)
df_nbclf = pd.DataFrame(columns=['Classifier', 'F-Measure', 'Accuracy'])
df_nbclf.loc[len(df_nbclf)] = ['Naive Bayes', fmeasure, accuracy]

print (df_nbclf)