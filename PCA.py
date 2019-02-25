# -*- coding: utf-8 -*-
"""
Created on Fri Jan  11 09:07:17 2019

@author: bx_chen
"""
from dataprocess import get_all_data, get_all_labels
from sklearn.preprocessing import minmax_scale
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA, KernelPCA
from sklearn.grid_search import GridSearchCV  #参数优化
from sklearn import manifold



Signals = get_all_data()

SignalsNormaliz = minmax_scale(Signals, axis=1)

#estimator = PCA(n_components=256)
#estimator = KernelPCA(n_components=256, kernel="rbf", fit_inverse_transform=True, gamma=10)

#estimator = manifold.LocallyLinearEmbedding(n_neighbors=256, n_components=128, method='modified')
estimator = manifold.Isomap(n_neighbors=256, n_components=128)
pca_SignalsNormaliz = estimator.fit_transform(SignalsNormaliz)

Labels = get_all_labels()


#X_train, y_train = pca_SignalsNormaliz[0:315*2,:], Labels[0:315*2]
#X_test, y_test = pca_SignalsNormaliz[315*2:, :], Labels[315*2:]
X_train, X_test, y_train, y_test = train_test_split(SignalsNormaliz, Labels, test_size=0.3, random_state=42)

model = xgb.XGBClassifier(max_depth=5,
                         min_child_weight=5,
                         learning_rate=0.1, 
                         n_estimators=250, 
                         silent=False, 
                         objective='multi:softmax',
                         gamma=0, 
                         max_delta_step=0, 
                         subsample=1, 
                         colsample_bytree=0.75, 
                         colsample_bylevel=0.8,
                         reg_alpha=0.5,
                         reg_lambda=0.5, 
                         scale_pos_weight=0.5,
                         base_score=0.5,
                         random_state=27
                         )

param_list = {'colsample_bytree':np.array([1]),
              'colsample_bylevel':np.array([1])
              }

Gsearch = GridSearchCV(estimator=model,
                       param_grid=param_list,
                       scoring='r2',
                       cv=5)

Gsearch.fit(X_train, y_train)

pre = Gsearch.predict(X_test)

acc = accuracy_score(y_test, pre)