#%%Kaggle code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.gridspec as gridspec
import matplotlib.backends.backend_pdf as backends
from pylab import figure, show, legend, ylabel
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn import preprocessing
import pickle
import itertools
#%% Load data
Data=pd.read_csv('C:/Users/roxanne/Desktop/cs-training.csv')
#%%
Data.NumberOfDependents=Data.NumberOfDependents.fillna(0)
Data['NaNMonthlyIncome'] = Data['MonthlyIncome'] == np.nan
Data['NoMonthlyIncome'] = Data['MonthlyIncome'] == 0
Data['LowMonthlyIncome'] = Data['MonthlyIncome'] < 100
Data['LogMonthlyIncome'] = np.log(Data['MonthlyIncome'])
Data.loc[~np.isfinite(Data['LogMonthlyIncome']), 'LogMonthlyIncome'] = 0
Data.loc[pd.isnull(Data['LogMonthlyIncome']), 'LogMonthlyIncome'] = 0
# log of income per person
Data['LogIncomePerPerson'] = Data['LogMonthlyIncome'] / Data['NumberOfDependents']
Data.loc[~np.isfinite(Data['LogIncomePerPerson']), 'LogIncomePerPerson'] = 0
# log of RevolvingUtilizationOfUnsecuredLines
Data['LogRevolvingUtilizationOfUnsecuredLines'] = np.log(Data['RevolvingUtilizationOfUnsecuredLines'])
Data.loc[~np.isfinite(Data['LogRevolvingUtilizationOfUnsecuredLines']), 'LogRevolvingUtilizationOfUnsecuredLines'] = 0
# age related
Data['YoungAge'] = Data['age'] < 21
Data['OldAge'] = Data['age'] > 65
Data['ZeroAge'] = Data['age'] == 0
Data['LogAge'] = np.log(Data['age'])
Data.loc[Data['ZeroAge'] == True, 'LogAge'] = 0
# restore debt and take log
Data['LogDebt'] = np.log(Data['DebtRatio'] * Data['LogMonthlyIncome'])
Data.loc[~np.isfinite(Data['LogDebt']), 'LogDebt'] = 0
Data['LogDebtPerPerson'] = Data['LogDebt'] / Data['NumberOfDependents']
Data.loc[~np.isfinite(Data['LogDebtPerPerson']), 'LogDebtPerPerson'] = 0
# binary columns late or not
Data['NoPastDue30-59'] = Data['NumberOfTime30-59DaysPastDueNotWorse'] == 0
Data['NoPastDue60-89'] = Data['NumberOfTime60-89DaysPastDueNotWorse'] == 0
Data['NoLateOver90'] = Data['NumberOfTimes90DaysLate'] == 0
Data['Util_bin'] = pd.qcut(Data['RevolvingUtilizationOfUnsecuredLines'], 4)
Data['DRat_bin'] = pd.qcut(Data['DebtRatio'], 4)
Data['Inc_bin'] = pd.qcut(Data['MonthlyIncome'], 4)
Data['OCL_bin'] = pd.qcut(Data['NumberOfOpenCreditLinesAndLoans'], 4)
b=[0,2,100]
Data['Dep_bin'] = pd.cut(Data['NumberOfDependents'], b)
#Predictors table
Predictors = Data.iloc[:,11:26].copy()
x= pd.get_dummies(Data['Util_bin']).rename(columns=lambda x: 'Util_' + str(x))
Predictors = pd.concat([Predictors, x], axis=1)
x = pd.get_dummies(Data['DRat_bin']).rename(columns=lambda x: 'DebtRatio_' + str(x))
Predictors = pd.concat([Predictors, x], axis=1)
x = pd.get_dummies(Data['Inc_bin']).rename(columns=lambda x: 'IncomeBin_' + str(x))
Predictors = pd.concat([Predictors, x], axis=1)
x = pd.get_dummies(Data['OCL_bin']).rename(columns=lambda x: 'OpenCrBin_' + str(x))
Predictors = pd.concat([Predictors, x], axis=1)
x = pd.get_dummies(Data['Dep_bin']).rename(columns=lambda x: 'DepBin_' + str(x))
Predictors = pd.concat([Predictors, x], axis=1)
#%%
f1 = open('C:/Users/roxanne/Desktop/classifier.pkl', 'rb')
rf1 = pickle.load(f1)
f1.close()
y_pred2 = rf1.predict(Predictors)
