# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:50:25 2016

@author: Hossam
@modified: Maria, Jan 2020
"""
import PSO as pso
import MVO as mvo
import GWO as gwo
import MFO as mfo
import CS as cs
import BAT as bat
import WOA as woa
import FFA as ffa
import SSA as ssa
import GA as ga
import HHO as hho
import SCA as sca
import JAYA as jaya
import benchmarks
import csv
import numpy
import time
import statistics
import pandas as pd 
import numpy
import math
from math import sqrt
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

#import the required dataset
train = pd.read_csv('train_scour_weir.csv')
test = pd.read_csv('test_scour_weir.csv')
y_test = test.iloc[:,-1]
y_train = train.iloc[:,-1]

x_train = train.drop(labels = 'class', axis = 1)
x_test = test.drop(labels = 'class', axis = 1)

def selector(algo,func_details,popSize,Iter):
    function_name = func_details[0]
    
    lb = func_details[1]
    ub = func_details[2]
    dim = len(train.columns)+9 
    #alpha = numpy.random.uniform(0.001,1)                 #if you don't want to use alpha put it 0
    alpha = 0
    fid = 2
   
    if(algo == 0):
        x = pso.PSO(getattr(benchmarks, function_name),alpha,lb,ub,dim,popSize,Iter,fid)
    if(algo == 1):
        x = mvo.MVO(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter)
    if(algo == 2):
        x = gwo.GWO(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter)
    if(algo == 3):
        x = mfo.MFO(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter)
    if(algo == 4):
        x = cs.CS(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter)
    if(algo == 5):
        x = bat.BAT(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter)
    if(algo == 6):
        x = woa.WOA(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter)
    if(algo == 7):
        x = ffa.FFA(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter)
    if(algo == 8):
        x = ssa.SSA(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter)
    if(algo == 9):
        x = ga.GA(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter,fid)
    if(algo == 10):
        x = hho.HHO(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter)
    if(algo == 11):
        x = sca.SCA(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter)
    if(algo == 12):
        x = jaya.JAYA(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter)
    return x
    
    
# Select optimizers
PSO = False
MVO = False
GWO = False
MFO = False
CS = False
BAT = False
WOA = False
FFA = False
SSA = False
GA = True
HHO = False
SCA = False
JAYA = False


# Select benchmark function
F1 = False
F2 = True
F3 = False
F4 = False
F5 = False
F6 = False
F7 = False
F8 = False
F9 = False
F10 = False
F11 = False
F12 = False
F13 = False
F14 = False
F15 = False
F16 = False
F17 = False
F18 = False
F19 = False



optimizer = [PSO, MVO, GWO, MFO, CS, BAT, WOA, FFA, SSA, GA, HHO, SCA, JAYA]
benchmarkfunc = [F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,F14,F15,F16,F17,F18,F19] 
        
# Select number of repetitions for each experiment. 
# To obtain meaningful statistical results, usually 30 independent runs 
# are executed for each algorithm.
NumOfRuns = 30

# Select general parameters for all optimizers (population size, number of iterations)
PopulationSize = 50
Iterations = 50

#Export results ?
Export = True


#ExportToFile = "YourResultsAreHere.csv"
#Automaticly generated name by date and time
ExportToFile = "experiment-scoup-pred"+time.strftime("%Y-%m-%d-%H-%M-%S")+".csv" 

# Check if it works at least once
Flag = False

# CSV Header for for the cinvergence 
CnvgHeader = []

RMSE = numpy.zeros(Iterations)
MAE = numpy.zeros(Iterations)
R2 = numpy.zeros(Iterations)
CORR = numpy.zeros(Iterations)
VAF = numpy.zeros(Iterations)

for l in range(0,Iterations):
	CnvgHeader.append("Iter"+str(l+1))


for i in range (0, len(optimizer)):
    for j in range (0, len(benchmarkfunc)):
        if((optimizer[i] = = True) and (benchmarkfunc[j] = = True)): # start experiment if an optimizer and an objective function is selected
            for k in range (0,NumOfRuns):
                print(['At run '+ str(k+1)]);
                
                func_details = benchmarks.getFunctionDetails(j)
                x = selector(i,func_details,PopulationSize,Iterations)
               
                #************************************************************                
                w = x.gbestP.nonzero()
                
                q = numpy.array(w)[0]
                q = q[:-10]
                
                x_train = train.iloc[:, lambda train: q]
                x_test = test.iloc[:, lambda test: q]
                
                x2 = x.gbestP
                
                lambda1 = float(x2[len(x2)-1]*0.5)+0.5 
                alpha = float(x2[len(x2)-2]*0.00009)+0.0001
                bylevel = float(x2[len(x2)-3]*.0001)+0.999
                bytree = float(x2[len(x2)-4]*.0001)+0.999
                subsample1 = float(x2[len(x2)-5]*.01)+0.99
                minChWeight = int(x2[len(x2)-6]*5)+1
                gamma1 = float(x2[len(x2)-7]*0.00009)+0.0001
                NoEstimator = int(x2[len(x2)-8]*50)+100
                learnRate = float(x2[len(x2)-9]*0.05)+0.03
                maxDepth = int(x2[len(x2)-10]*12)+3
                
                x3 = [lambda1,alpha,bylevel,bytree,subsample1,minChWeight,gamma1,NoEstimator,learnRate,maxDepth]
                              
                xgbb = xgb.XGBRegressor(max_depth = maxDepth, n_estimators = NoEstimator, learning_rate = learnRate,
                    gamma = gamma1, min_child_weight = minChWeight,subsample = subsample1, 
                    colsample_bytree = bytree,colsample_bylevel = bylevel,
                    reg_alpha = alpha,reg_lambda = lambda1,silent = True,verbosity = 0,verbose = True)
                
                xgbb.fit(x_train, y_train)
                y_pred = xgbb.predict(x_test)
                
                y_ypred = y_test- y_pred   
                var_y = statistics.variance(y_test)
                var_yy = statistics.variance(y_ypred,statistics.mean(y_ypred))
                vaf = 1-var_yy/var_y
                
                rmse = math.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                corr = numpy.corrcoef(y_test, y_pred)[0,1]
                r2 = r2_score(y_test, y_pred)
                
                print('RMSE = ',rmse)
                print('MAE = ',mae)
                print('R2:', r2)    
                print('correlation:', corr)
                print('VAF:', vaf)
                
                RMSE[0] = rmse
                MAE[0] = mae
                R2 [0] = r2
                CORR[0] = corr
                VAF[0] = vaf
                
                if(Export = = True):
                    with open(ExportToFile, 'a',newline = '\n') as out:
                        writer = csv.writer(out,delimiter = ',')
                        if (Flag = = False): # just one time to write the header of the CSV file
                            header = numpy.concatenate([["Optimizer","objfname","startTime","EndTime","ExecutionTime","Measure "],CnvgHeader])
                            writer.writerow(header)
                        a = numpy.concatenate([[x.optimizer,x.objfname,x.startTime,x.endTime,x.executionTime,'Convergence'],x.convergence])
                        writer.writerow(a)
                        b = numpy.concatenate([[x.optimizer,x.objfname,x.startTime,x.endTime,x.executionTime,'Best RMSE'],RMSE])
                        c = numpy.concatenate([[x.optimizer,x.objfname,x.startTime,x.endTime,x.executionTime,'Best MAE'],MAE])
                        e = numpy.concatenate([[x.optimizer,x.objfname,x.startTime,x.endTime,x.executionTime,'Best R2'],R2])
                        d = numpy.concatenate([[x.optimizer,x.objfname,x.startTime,x.endTime,x.executionTime,'Best Corr'],CORR])
                        dd = numpy.concatenate([[x.optimizer,x.objfname,x.startTime,x.endTime,x.executionTime,'Best VAF'],VAF])
                        writer.writerow(b)
                        writer.writerow(c)
                        writer.writerow(e)
                        writer.writerow(d)
                        writer.writerow(dd)
                        writer.writerow(["selected features"])
                        writer.writerow(q)
                        writer.writerow(["lambda1","alpha","bylevel","bytree","subsample1","minChWeight","gamma1","NoEstimator","learnRate","maxDepth"])
                        writer.writerow(x3)
                        writer.writerow(["test predictions"])
                        writer.writerow(y_pred)
                        writer.writerow("")
                            
                        out.close()
                Flag = True # at least one experiment
                            
                           
                
if (Flag = = False): # Faild to run at least one experiment
    print("No Optomizer or Cost function is selected. Check lists of available optimizers and cost functions") 
        
        
