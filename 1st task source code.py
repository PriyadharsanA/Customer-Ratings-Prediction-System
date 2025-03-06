import pandas as pd
import numpy as np
#Data importing and initial preprocessing
df=pd.read_csv("C:/Users/User/Desktop/Dataset.csv").dropna()
df=df.drop(columns=['Restaurant Name','Address','Country Code','Locality','Locality Verbose','Longitude','Latitude'])
df=df.drop_duplicates()

from sklearn.preprocessing import LabelEncoder
#Label Encoding
le=LabelEncoder()
df['Has Table booking']=le.fit_transform(df['Has Table booking'])    
df['Has Online delivery']=le.fit_transform(df['Has Online delivery'])
df['Is delivering now']=le.fit_transform(df['Is delivering now'])
df['Switch to order menu']=le.fit_transform(df['Switch to order menu'])
df['City']=le.fit_transform(df['City'])
df['Currency']=le.fit_transform(df['Currency'])
df['Rating color']=le.fit_transform(df['Rating color'])
df['Rating text']=le.fit_transform(df['Rating text'])

from sklearn.preprocessing import MultiLabelBinarizer
#Multi Label Binarizer for Cuisines
mlb=MultiLabelBinarizer()
df['Cuisines']=mlb.fit_transform(df['Cuisines'])

#Initializing X & Y
x=df.drop(columns='Aggregate rating')
y=df['Aggregate rating']

from sklearn.model_selection import train_test_split
#Split the data into train and test data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#Importing all machine learning models
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,mean_absolute_error
#Model Training and Performance Metrics

lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred1=lr.predict(x_test)
print("1.LINEAR REGRESSION")
print("R^2 Score: ",lr.score(x_test,y_test)*100)
print("Mean Squared Error: ",mean_squared_error(y_test,y_pred1))
print("Mean Absolute Error: ",mean_absolute_error(y_test,y_pred1))
print('\n')

rf=RandomForestRegressor()
rf.fit(x_train,y_train)
y_pred2=rf.predict(x_test)
print("2.RANDOM FOREST")
print("R^2 Score: ",rf.score(x_test,y_test)*100)
print("Mean Squared Error: ",mean_squared_error(y_test,y_pred2))
print("Mean Absolute Error: ",mean_absolute_error(y_test,y_pred2))
print('\n')

svr=SVR()
ss=StandardScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.transform(x_test)
svr.fit(x_train,y_train)
y_pred3=svr.predict(x_test)
print("3.SVR")
print("R^2 Score: ",svr.score(x_test,y_test)*100)
print("Mean Squared Error: ",mean_squared_error(y_test,y_pred3))
print("Mean Absolute Error: ",mean_absolute_error(y_test,y_pred3))
