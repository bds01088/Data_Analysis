import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, datasets

iris = datasets.load_iris()
# print (iris.keys())
# print (iris.data.shape)
# print (iris.feature_names)

# print (iris.DESCR)

x = iris.data[:,:2]
y = iris.target
#선형
#svm = svm.SVC(kernel='linear', C=1).fit(x,y)
#비선형
svm = svm.SVC(kernel='rbf', C=100 ,gamma='auto').fit(x,y)
#감마는 멀리떨어진 요소들의 영향력을 뜻함 낮을수록 영향력이 낮아짐

xmin, xmax = x[:,0].min()-1,x[:,0].max()+1
ymin, ymax = x[:,1].min()-1,x[:,1].max()+1
plot_unit = 0.025
xx,yy = np.meshgrid(np.arange(xmin,xmax,plot_unit),np.arange(ymin,ymax,plot_unit))

z = svm.predict(np.c_[xx.ravel(),yy.ravel()])
z = z.reshape(xx.shape)
plt.pcolormesh(xx,yy,z,alpha=0.1)
plt.scatter(x[:,0],x[:,1],c=y)
plt.xlabel('Sepal lenth')
plt.ylabel('Sepla width')
plt.xlim(xx.min(), xx.max())
plt.title('Support Vector Machine')
plt.show()
print ('정확도 : ', svm.score(X=x, y=y))