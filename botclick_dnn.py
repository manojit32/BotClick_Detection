import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
# from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential,model_from_json
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.layers import BatchNormalization
import pylab as plt
import h5py
import sklearn.metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

df = pd.read_csv("bot_clicks_santized.csv",error_bad_lines=False,sep="\t")
a = df["ip"].unique().shape
df1=df.dropna()
uniqueip=df["ip"].unique()
uniquedev=df["deviceType"].unique()
df2=df1.drop(['userAgent','eventId','publisher','operatingSystem','clicks'],axis=1)
X=df2.drop('botClicks',axis=1)
Y=df2['botClicks']
df3=X.groupby(['ip','location']).sum()
cols = []
cols =  X.columns
cat_columns = X.select_dtypes(['object']).columns
X[cat_columns] = X[cat_columns].apply(lambda x: x.astype('category'))
X[cat_columns] = X[cat_columns].apply(lambda x: x.cat.codes)
sc=StandardScaler()
X=sc.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=1,test_size=0.2)

num_classes = 2
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
batch_size = 512

epochs = 100
first_layer_size = 11
model = Sequential()
model.add(Dense(first_layer_size, activation='relu', input_shape=(11,)))
model.add(Dense(10, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(9, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(9, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

classifier = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,verbose=0,validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#print(X_test[1:2].shape)
predictions = model.predict(X_test)
# prediction = prediction[0]
print('Prediction\n',predictions)
thr_pred =(predictions>0.5)*1
print('Thresholded output\n',thr_pred)

# results = confusion_matrix(y_test, l)
results = confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
print ('Confusion Matrix of Neural network:')
print(results)
print ('Accuracy Score :',accuracy_score(y_test.argmax(axis=1), predictions.argmax(axis=1)))
print ('Report : ')
print (classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1)))


