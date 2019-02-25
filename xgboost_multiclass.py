
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  11 09:07:17 2019

@author: bx_chen
"""
from dataprocess import get_all_data, get_all_labels
from sklearn.preprocessing import minmax_scale
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from mlxtend.classifier import StackingClassifier

from sklearn.metrics import roc_curve, auc, accuracy_score

import matplotlib.pyplot as plt

Signals = get_all_data()
Labels = get_all_labels()

SignalsNormaliz = minmax_scale(Signals, axis=1)
'''
特征维度：声发射9*19  振动3*9*14   = 549 dim
通道=['ae_rms','vibration_x', 'vibration_y', 'vibration_z']
'''

X_train, y_train = SignalsNormaliz[0:315*2,:], Labels[0:315*2]

X_test, y_test = SignalsNormaliz[315*2:, :], Labels[315*2:]
'''
XGBoost
'''
#model = xgb.XGBClassifier (max_depth=6,
#                         min_child_weight=5,
#                         learning_rate=0.1, 
#                         n_estimators=250, 
#                         silent=False, 
#                         objective='multi:softmax',
#                         eval_metric='auc',
#                         gamma=0, 
#                         max_delta_step=0, 
#                         subsample=1, 
#                         colsample_bytree=0.75, 
#                         colsample_bylevel=0.8,
#                         reg_alpha=0.5,
#                         reg_lambda=0.85, 
#                         scale_pos_weight=0.5,
#                         base_score=0.5,
#                         random_state=27
#                         )

'''
rf
'''
#model = RandomForestClassifier(n_estimators=250,
#                               criterion='gini',
#                                max_depth=6,
#                              min_samples_split=2, 
#                               min_samples_leaf=1, 
#                               min_weight_fraction_leaf=0.0,                                          
#                               max_features='auto', 
#                               max_leaf_nodes=None, 
#                               bootstrap=True,                                          
#                               n_jobs=1, 
#                               random_state=None, 
#                               verbose=0,                                          
#                               warm_start=False, 
#                               class_weight=None) 

'''
stacking
'''
KNN = KNeighborsClassifier()
GNB = GaussianNB()
RF = RandomForestClassifier()
LR = LogisticRegression()
XGBOOST = xgb.XGBClassifier()
model = StackingClassifier(classifiers=[KNN, LR, GNB], meta_classifier=LR)

model.fit(X_train, y_train)
pre = model.predict(X_test)

acc = accuracy_score(y_test, pre)


