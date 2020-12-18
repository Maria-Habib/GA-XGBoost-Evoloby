# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 11:45:39 2019

@author: Maria
"""

import numpy as np 
import pandas as pd 
import math
import statistics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb


train = pd.read_csv('train_scour_weir.csv')
test = pd.read_csv('test_scour_weir.csv')

y_test = test.iloc[:,-1]
y_train = train.iloc[:,-1]
x_train = train.drop(labels='class', axis=1)
x_test = test.drop(labels='class', axis=1)

#search based on random numbers
"""
parameters_ = {'reg_lambda':[np.random.uniform(0,1),np.random.uniform(0,1),np.random.uniform(0,1)],
                            'learning_rate':[np.random.uniform(0,1),np.random.uniform(0,1),np.random.uniform(0,1)],
              'n_estimators': [int(np.random.uniform(0,1)*450+50),int(np.random.uniform(0,1)*450+50),int(np.random.uniform(0,1)*450+50)],
              'max_depth':[int(np.random.uniform(0,1)*20),int(np.random.uniform(0,1)*20),int(np.random.uniform(0,1)*20)],
              'min_child_weight':[int(np.random.uniform(0,1)*10),int(np.random.uniform(0,1)*10),
               int(np.random.uniform(0,1)*10)],
               'gamma':[int(np.random.uniform(0,1)*5),int(np.random.uniform(0,1)*5),int(np.random.uniform(0,1)*5)],
                        'subsample':[np.random.uniform(0,1),np.random.uniform(0,1),np.random.uniform(0,1)],
                        'colsample_bytree':[np.random.uniform(0,1),np.random.uniform(0,1),np.random.uniform(0,1)],
                        'reg_alpha':[np.random.uniform(0,1),np.random.uniform(0,1),np.random.uniform(0,1)],
                        'colsample_bylevel':[np.random.uniform(0,1),np.random.uniform(0,1),np.random.uniform(0,1)]}
"""
#search with fixed numbers
parameters = {'reg_lambda':[0.1,0.5,0.9],'learning_rate':[0.02,0.8,0.055],
            'n_estimators': [50,110,302], 'max_depth':[5,10,18],
            'min_child_weight':[2,5,8],'gamma':[0.1,2,5],
            'subsample':[0.1,0.5,0.9],'colsample_bytree':[0.1,0.5,0.9],
            'reg_alpha':[0.1,0.5,0.9],'colsample_bylevel':[0.1,0.5,0.9]}


xgbb = xgb.XGBRegressor()
clf = GridSearchCV(xgbb, parameters,cv=5,scoring='neg_mean_squared_error') 
       
iters=30
vaf=[]
rmse=[]
mae=[]
corr=[]
r2=[]
for i in range(1,iters):
    clf.fit(x_train, y_train)
    y_pred =clf.predict(x_test)
    y_ypred=y_test- y_pred
    var_y=statistics.variance(y_test)
    var_yy=statistics.variance(y_ypred,statistics.mean(y_ypred))
    
    vaf.append(1-var_yy/var_y)  
    rmse.append(math.sqrt(mean_squared_error(y_test, y_pred)))
    mae.append(mean_absolute_error(y_test, y_pred))
    corr.append(np.corrcoef(y_test, y_pred)[0,1])
    r2.append(r2_score(y_test, y_pred))


print(y_pred)
print('RMSE=',statistics.mean(rmse),statistics.stdev(rmse))
print('MAE=',statistics.mean(mae),statistics.stdev(mae))
print('R2:', statistics.mean(r2),statistics.stdev(r2))  
print('correlation:', statistics.mean(corr),statistics.stdev(corr))
print('VAF=',statistics.mean(vaf),statistics.stdev(vaf))


