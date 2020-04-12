# -*- coding: utf-8 -*-
"""
Created on Wed May  1 18:25:40 2019

@author: vignesh
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 17:05:31 2019

@author: CAIA79
"""
1+1
#dataimport
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV 
from scipy.stats import randint 
df=pd.read_csv('E:\\ML_PROJECT\\Dataset.csv')
#dataprep
df.columns
df.info()
df.shape
df.describe()
df.head()
df.count()


#missing detection
df.isnull().sum()

#dropping_features_with_70%_and _more_null_values
d=df[df.columns[df.isnull().mean() <= 0.70]]    
d.info()
d.dtypes
#dropping_unique_values
d=d[d.columns[d.nunique()>1]]

#dropping_columns
d.columns
d.drop(['id'],axis=1,inplace=True)

#categoricalcolumns

#labelencoding
d['LN2_RIndLngBEOth']=d['LN2_RIndLngBEOth'].astype(str)
d['LN2_WIndLngBEOth']=d['LN2_WIndLngBEOth'].astype(str)
labelencoder = LabelEncoder()
d['LN2_RIndLngBEOth']=labelencoder.fit_transform(d['LN2_RIndLngBEOth'])
d['LN2_WIndLngBEOth']=labelencoder.fit_transform(d['LN2_WIndLngBEOth'])
#fillingnull
for column in d.columns:
    d[column].fillna(d[column].mode()[0], inplace=True)
#correlationcheck
corr_matrix = d.corr()
x=d.drop(['is_female'],axis=1)  
x.info()
x.head()
#removing_features_based_on_correlation_value
x=x.drop(['FL3'],axis=1)
x=x.drop(['FF7_5'],axis=1)
x=x.drop(['MT12_6'],axis=1)
x=x.drop(['DG9c'],axis=1)
x=x.drop(['FB19B_1'],axis=1)
x=x.drop(['FF7_96'],axis=1)
x=x.drop(['IFI24'],axis=1)
x=x.drop(['MT12_10'],axis=1)
x=x.drop(['MT12_8'],axis=1)
x=x.drop(['FL10'],axis=1)
x=x.drop(['MT12_14'],axis=1)
x=x.drop(['DG8c'],axis=1)
x=x.drop(['DG10b'],axis=1)
x=x.drop(['FF5'],axis=1)
x=x.drop(['MT12_1'],axis=1)
x=x.drop(['MT12_2'],axis=1)
x=x.drop(['MT12_12'],axis=1)
x=x.drop(['MT12_13'],axis=1)
#collinearity
# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]

# Drop features 
x=x.drop(to_drop,axis=1)

#Variableinflationfactor
from statsmodels.stats.outliers_influence import variance_inflation_factor    

   

 
def calculate_vif_(x, thresh=5.0):
    variables = list(range(x.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(x.iloc[:, variables].values, ix)
               for ix in range(x.iloc[:, variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + x.iloc[:, variables].columns[maxloc] +
                  '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True
    print('Remaining variables:')
    print(x.columns[variables])
    return x.iloc[:, variables]

calculate_vif_(x, thresh=5.0)
#remainingvariables
variables=['AA14', 'DG3', 'DG3A', 'DG4', 'DG6', 'DG8a', 'DG8b', 'DG8c', 'DG9a',
       'DG9b', 'DG9c', 'DG10b', 'DG10c', 'DL1', 'DL2', 'DL8', 'DL11', 'DL24',
       'MT1', 'MT1A', 'MT3_2', 'MT3_3', 'MT5', 'MT6', 'MT6A', 'MT6B', 'MT6C',
       'MT12_1', 'MT12_2', 'MT12_3', 'MT12_4', 'MT12_5', 'MT12_6', 'MT12_7',
       'MT12_8', 'MT12_9', 'MT12_10', 'MT12_11', 'MT12_12', 'MT12_13',
       'MT12_14', 'MT12_96', 'MT17_2', 'FF2A', 'FF5', 'FF6_1', 'FF6_2',
       'FF6_3', 'FF6_4', 'FF6_5', 'FF6_8', 'FF6_9', 'FF7_1', 'FF7_2', 'FF7_3',
       'FF7_5', 'FF7_6', 'FF7_7', 'FF7_96', 'IFI14_1', 'IFI14_2', 'IFI15_1',
       'IFI16_1', 'IFI16_2', 'IFI17_1', 'IFI17_2', 'IFI18', 'IFI24', 'FL3',
       'FL4', 'FL7_1', 'FL7_2', 'FL7_3', 'FL9A', 'FL9B', 'FL9C', 'FL10',
       'FL11', 'FL12', 'FL13', 'FL14', 'FL15', 'FL16', 'FL17', 'FL18', 'FB13',
       'FB19', 'FB19B_1', 'FB20', 'FB24', 'LN2_1', 'LN2_RIndLngBEOth', 'GN1',
       'GN2', 'GN3', 'GN4']

x=x[variables]    
y=d['is_female']
#converting the categorical features to a category datatype
for i in x[variables]:
        x[i]=x[i].astype('category')

d['is_female']=d['is_female'].astype('category')
    
#modelbuilding

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#logistic

mymodel = LogisticRegression()
mymodel.fit(X_train, Y_train)
y_pred = mymodel.predict(X_test)
accuracy= accuracy_score(Y_test,y_pred)
print("Accuracy of Our Model: ", round(accuracy* 100, 2), ' %')
print( "Classification Report :\n ", classification_report(Y_test, y_pred))
#cross_validation_for_hyperparameters_dt
dt=DecisionTreeClassifier()
param_dist = {"max_depth": [3, None], 
              "max_features": randint(1, 9), 
              "min_samples_leaf": randint(1, 9), 
              "criterion": ["gini", "entropy"]} 
tree_cv = RandomizedSearchCV(dt, param_dist, cv = 5) 
tree_cv.fit(x, y) 
# Print the tuned parameters and score 
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_)) 
print("Best score is {}".format(tree_cv.best_score_)) 

#decisiontree
dt = DecisionTreeClassifier(criterion='gini', max_depth= None, max_features= 6, min_samples_leaf= 4)
dt.fit(X_train, Y_train)
pred_dt = dt.predict(X_test)
accuracy= accuracy_score(Y_test,pred_dt)
print("Accuracy of Our Model: ", round(accuracy* 100, 2), ' %')
print( "Classification Report :\n ", classification_report(Y_test, pred_dt))

#cross_validation_for_hyperparameters
rf = RandomForestClassifier()
param_dist1 = {"n_estimators":[20,800,3],"max_depth": [3, None], 
              "max_features": randint(1, 9), 
              "min_samples_leaf": randint(1, 9), 
              "criterion": ["gini", "entropy"]} 
tree_cv1 = RandomizedSearchCV(rf, param_dist, cv = 5) 
tree_cv1.fit(x, y) 
# Print the tuned parameters and score 
print("Tuned Decision Tree Parameters: {}".format(tree_cv1.best_params_)) 
print("Best score is {}".format(tree_cv1.best_score_))
#random
rf = RandomForestClassifier(n_estimators = 100,  random_state = 0,criterion='gini',max_depth=None,
                            max_features=7,min_samples_leaf=1)
rf.fit(X_train, Y_train)
pred_rf = rf.predict(X_test)
accuracy= accuracy_score(Y_test,pred_rf)
print("Accuracy of Our Model: ", round(accuracy* 100, 2), ' %')
print( "Classification Report :\n ", classification_report(Y_test, pred_rf))


#kappa score
from sklearn.metrics import cohen_kappa_score
kappa= cohen_kappa_score(pred_rf,Y_test)
print('kappa value',round(kappa*100,2),'%')

#roc curve
from sklearn import metrics
y_pred_proba = rf.predict_proba(X_test)[::,1]
y_pred_proba
fpr, tpr, _ = metrics.roc_curve(Y_test,  y_pred_proba)
auc = metrics.roc_auc_score(Y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

#important features by random foreest
imp=list(zip(x.columns,rf.feature_importances_))
impacting_features=pd.Series(rf.feature_importances_,x.columns).sort_values( ascending=False)[:20].plot(kind='bar')



#gradientboost
from sklearn.ensemble import GradientBoostingClassifier
glf = GradientBoostingClassifier(n_estimators=100, learning_rate=1, max_depth=1)
glf.fit(X_train, Y_train)
glf_pred=glf.predict(X_test)
accuracy= accuracy_score(Y_test,glf_pred)
print("Accuracy of Our Model: ", round(accuracy* 100, 2), ' %')
print( "Classification Report :\n ", classification_report(Y_test, glf_pred))
