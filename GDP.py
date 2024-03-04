import numpy as np 
import pandas as pd 
import pandas as pd
df=pd.read_csv('ukdata.csv')
df1 = df[['Year','GDP']]
df1.dropna()
df1.drop(df1.index[[61,66]], inplace=True)
df1.drop(df1.index[[63,64]], inplace=True) df1.drop(df1.index[[61,62]], inplace=True)
print(df1[df1['GDP'].isnull()])
df1
df1.shape
#Finding null values in the dataset df1.isnull().sum()
df.dtypes
import matplotlib.pyplot as plt import seaborn as sns sns.pairplot(df1)
#Finding the distribution of data
def distplots(col): sns.distplot(df[col]) plt.show()
for i in list(df1.columns)[1:]: distplots(i)
#Finding the outliers of data
def boxplots(col): sns.boxplot(df[col]) plt.show()
for i in list(df1.select_dtypes(exclude=['object']).columns)[1:]: boxplots(i)
plt.figure(figsize=(20,20)) corr=df1.corr() sns.heatmap(corr,annot=True)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split from sklearn.neighbors import KNeighborsRegressor from sklearn.metrics import mean_squared_error from sklearn.metrics import accuracy_score
import pandas as pd
Predict UK's GDP
df=pd.read_csv('ukdata.csv')
df1 = df[['Year','GDP']]
df1.dropna()
df1.drop(df1.index[[61,66]], inplace=True)
df1.drop(df1.index[[63,64]], inplace=True) df1.drop(df1.index[[61,62]], inplace=True)
print(df1[df1['GDP'].isnull()]) df1.corr()
X=df1[['Year']] Y=df1[['GDP']]
X_train ,X_test ,Y_train ,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0) clf = KNeighborsRegressor(2)
clf.fit(X_train,Y_train)
y_pred = clf.predict(X_test)
print(y_pred) print((mean_squared_error(Y_test,y_pred))) #Plotting the observed and predicted data import matplotlib.pyplot as plt
x_ax =range(len(X_test)) plt.plot(x_ax,Y_test,label='Observed',color='k',linestyle = '-') plt.plot(x_ax,y_pred,label='Prediction',color='k',linestyle = '--') # Support Vector Regression
from sklearn.svm import SVR
#Creating and fitting SVR model
ll_svm = SVR().fit(X_train,Y_train) print(ll_svm.score(X_train,Y_train))
ytrain_pred = ll_svm.predict(X_train)
print(ytrain_pred) print(mean_squared_error(Y_train,ytrain_pred))
#Prediction on the testing dataset ytest_pred=ll_svm.predict(X_test)
print(ytest_pred) print(mean_squared_error(Y_test,ytest_pred))
#Plotting the observed and predicted data
import matplotlib.pyplot as plt
x_ax =range(len(X_test)) plt.plot(x_ax,Y_test,label='Observed',color='k',linestyle = '-') plt.plot(x_ax,ytest_pred,label='Prediction',color='k',linestyle = '--')
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42) # Train the model on training data
rf.fit(X_train, Y_train);
# Use the forest's predict method on the test data
predictions = rf.predict(X_test)
print(predictions) print(mean_squared_error(Y_test,predictions)) #Plotting the observed and predicted data import matplotlib.pyplot as plt
x_ax =range(len(X_test)) plt.plot(x_ax,Y_test,label='Observed',color='k',linestyle = '-') plt.plot(x_ax,predictions,label='Prediction',color='k',linestyle = '--')
Predict UK's GDP