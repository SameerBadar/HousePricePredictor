import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns',None)
df = pd.read_csv('C:/Users/ALI/Documents/Machine Learning/Data PreProcessing/House_Price.csv',header = 0)
# here we will put 80 % of our data for training and 20 % 
#for testing
df['n_hos_beds'] = df['n_hos_beds'].mean()
df['waterbody'] = df['waterbody'].mode()[0]
#now we will find outliers with depndent and indepedent var
df['crime_rate'] = np.log(1+df['crime_rate'])
#now graph looks more linear
df['avgDistance'] = df['dist1'] + df['dist2'] + df['dist3'] + df['dist4']/4
del df['dist1']
del df['dist2']
del df['dist3']
del df['dist4']
uv = np.percentile(df.n_hot_rooms,[99])[0]#[] without this it returns array
df.loc[df['n_hot_rooms'] > 3 * uv, 'n_hot_rooms'] = 3 * uv
lv = np.percentile(df.rainfall,[1])[0]
df.loc[df['rainfall']<0.3 * lv,'rainfall'] = 0.3 * lv
del df['bus_ter']
df = pd.get_dummies(df)
del df['airport_NO']
# del df['waterbody_Lake and River']
x_data = df.drop('price',axis=1)
y_data = df['price']
# 20 percent data goes for testing
x_train,x_test,y_train,y_test= train_test_split(x_data,y_data,test_size = 0.2,random_state = 1)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
# now fitting our training x,y training data in linearRegresion
lr = LinearRegression()
lr.fit(x_train,y_train)
#now the model has been trained
y_test_a = lr.predict(x_test) # that 20 percent data as x_test
y_train_a = lr.predict(x_train) # 80 percent data prediciton 
#now to check the accuracy we need to find rSquare
from sklearn.metrics import r2_score
#now find training rSquare and Testing rSquare
#and both will be compared
print(r2_score(y_test,y_test_a)) # orignal value and predicted test value
print(r2_score(y_train,y_train_a)) # orignal value and predicted training value
# we should always look at testing r_Score cuz its more important
#both values are perfect and are high 

print("Testing Data : ",y_test_a)
