import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
# from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
import pylab as plt
import h5py
import sklearn.metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC

df = pd.read_csv("bot_clicks_santized.csv",error_bad_lines=False,sep="\t")
a = df["ip"].unique().shape
df1=df.dropna()
uniqueip=df["ip"].unique()
uniquedev=df["deviceType"].unique()
df2=df1.drop(['userAgent','eventId','publisher','operatingSystem','clicks'],axis=1)
df_new = df2.sample(frac=0.5)
X=df_new.drop('botClicks',axis=1)
Y=df_new['botClicks']
df3=X.groupby(['ip','location']).sum()
cols = []
cols =  X.columns
cat_columns = X.select_dtypes(['object']).columns
X[cat_columns] = X[cat_columns].apply(lambda x: x.astype('category'))
X[cat_columns] = X[cat_columns].apply(lambda x: x.cat.codes)
sc=StandardScaler()
X=sc.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=1,test_size=0.2)

model=SVC(kernel="rbf",probability=True)
model.fit(X_train,y_train)
#print(X_test[1:2].shape)
predictions = model.predict(X_test)
# prediction = prediction[0]
print('Prediction\n',predictions)
thr_pred =(predictions>0.5)*1
print('Thresholded output\n',thr_pred)

# results = confusion_matrix(y_test, l)
results = confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
print ('Confusion Matrix of Support Vector Machine:')
print(results)
print ('Accuracy Score :',accuracy_score(y_test.argmax(axis=1), predictions.argmax(axis=1)))
print ('Report : ')
print (classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1)))


