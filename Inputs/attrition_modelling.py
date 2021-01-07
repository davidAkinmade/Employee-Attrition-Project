# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 14:10:49 2020

@author: akinmade
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import re
from sklearn.dummy import DummyClassifier

df3 = pd.read_csv('problem case combined.csv')

#assigning target column
y = df3['attrition']
df3 = df3.drop(['attrition'], axis=1)

#creating dummy variables
df_dum = pd.get_dummies(df3)
df_dum.columns


#using rfe to select important features
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

#checking 15 most important features
rfe = RFE(model, 15)
rfe = rfe.fit(df_dum, y)
print(rfe.support_)
print(rfe.ranking_)

rfe_pick = np.array([False,  True , True , True ,False , True , True , True , True , True, False,  True,
  True, False , True, False, False,  True , True,  True,  True])
columns = np.array(df_dum.columns)
columns_rfe = columns[rfe_pick] 

#but we still add the remaining 4 features related to department so as not to compromise effieciency in our test
dept_cols =['dept_accounting','dept_marketing','dept_support' ,'dept_sales']
columns_rfe = np.append(columns_rfe, dept_cols)
df_dum = df_dum[columns_rfe]

#assigning predictors
X = df_dum

#performing a stratified train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3)
print('Original dataset %s' % Counter(y_train))

#creating baseline model that always predicts mean value of target
dum = DummyClassifier(strategy='uniform', random_state=1)
dum.fit(X_train, y_train)
print(classification_report(y_test, dum.predict(X_test)))

# testing with xgboost model
model = XGBClassifier()
# define grid weights to help with imbalaced data
weights = [1, 10, 25, 50, 75, 99, 100, 1000]
param_grid = dict(scale_pos_weight=weights)
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='roc_auc')
# execute the grid search
grid_result = grid.fit(X_train, y_train)
# report the best configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#print classification report
print(classification_report(y_test, grid_result.best_estimator_.predict(X_test)))


#random forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
# define grid
#weights = [1, 10, 25, 50, 75, 99, 100, 1000]
#param_grid = dict(scale_pos_weight=weights)
# define evaluation procedure
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
results = model_selection.cross_val_score(rf, X_train, y_train, cv=cv, scoring='accuracy')
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))

rf.fit(X_train, y_train)
print(classification_report(y_test, rf.predict(X_test)))


from sklearn.svm import SVC
svc = SVC()
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
results = model_selection.cross_val_score(svc, X_train, y_train, cv=cv, scoring='accuracy')
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))
# define grid
svc.fit(X_train, y_train)

print(classification_report(y_test, svc.predict(X_test)))

#heatmap for the best performing model (rf)
y_pred = rf.predict(X_test)
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns
forest_cm = metrics.confusion_matrix(y_pred, y_test, ['yes', 'no'])
sns.heatmap(forest_cm, annot=True, fmt='.2f',xticklabels = ["Left", "Stayed"] ,cmap = 'coolwarm', yticklabels = ["Left", "Stayed"] )
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('Random Forest')
plt.savefig('random_forest')

#feature importance
importance = rf.feature_importances_
from matplotlib import pyplot
# summarize feature importance
feature_indexes_by_importance = importance.argsort()
print('Factors/features affecting employee turnover by percentage')

#printing out each feature by their percentage of importance and plotting it on a barchart
for index in feature_indexes_by_importance:
   print('{}-{:.2f}%'.format(columns_rfe[index], (importance[index] *100.0)))
   plt.xticks(rotation=90)
   pyplot.bar(columns_rfe[index], (importance[index] *100.0))


#predicting which employees are to leave next
df1 = pd.read_excel('problem case.xlsx', sheet_name='Existing employees')
employee_id = df1['Emp ID']
df1 = df1.drop(['Emp ID'], axis=1)
df1_dum = pd.get_dummies(df1)
df1_dum.columns
df1_dum = df1_dum.drop(['attrition_no','average_montly_hours'], axis=1)

employee_pred = pd.Series(rf.predict(df1_dum))
(unique, counts) = np.unique(employee_pred, return_counts=True)
np.asarray((unique, counts)).T

employee_attrition = pd.concat([employee_id, employee_pred], axis=1)
employee_attrition = employee_attrition.rename(columns={0:'Leave'})


