import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('C:/Users/82102/Desktop/python/selfdone/total_weather.csv', encoding="cp949")
print (len(data))
#print (data.tail())

# 서울부분만 가져오기
seoul = data[data.Location.isin(['서울'])]

#계절별 분류
data_dict = {"01":"겨울","02":"겨울","03":"봄","04":"봄","05":"봄","06":"여름","07":"여름","08":"여름","09":"가을","10":"가을","11":"가을","12":"겨울"}
seoul['Season'] = seoul.Date.str[5:7].map(data_dict)
# print (seoul.head())

seoul = seoul[seoul.Season.isin(['겨울'])]
# print (seoul.tail())

sns.set(font_scale=1.5)
f, ax = plt.subplots(figsize=(20,10))
sns_heatmap = sns.heatmap(seoul.corr(), annot=True, fmt=".2f", linewidths=.5, cmap="RdBu_r")
plt.show()