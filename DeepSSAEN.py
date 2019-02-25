# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 10:47:00 2019
堆叠自编码
@author: bx_chen
"""
from sklearn.preprocessing import scale, minmax_scale
from keras.layers import Input, LSTM, RepeatVector, Dense
from keras.models import Model
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from keras import regularizers
import numpy as np
from keras.models import load_model
from dataprocess import get_all_data, get_all_labels
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV  #参数优化


EPOCHS = 50

Signals = get_all_data()

SignalsNormaliz = minmax_scale(Signals, axis=1)
timesteps= Signals.shape[1]

'''
第一层自编码器
'''
inputs_1 = Input(shape=(timesteps, ))
#编码shape[1]->128
encoded_1 = Dense(256, activation='relu')(inputs_1)

#解码
decoded_1 = Dense(timesteps, activation='sigmoid')(encoded_1)
# model 输入到重构映射
autoencoder_1 = Model(input=inputs_1, output=decoded_1)
# 编码器 输入到编码
encoder_1 = Model(input=inputs_1, output=encoded_1)

autoencoder_1.compile(optimizer='adam', loss='binary_crossentropy')

callbacks_list = [ModelCheckpoint('checkpoints/model.h5', period=EPOCHS/5)]

autoencoder_1.fit(SignalsNormaliz, SignalsNormaliz,
                nb_epoch=EPOCHS,
                batch_size=32,
                shuffle=True,
                callbacks=callbacks_list,
                validation_split=0)
print('堆叠自动编码器第一层训练完成')

'''
第二层自动编码器
'''
encoded_inputs_1 = encoder_1.predict(SignalsNormaliz)

inputs_2 = Input(shape=(256, ))
#编码shape[1]->128
encoded_2 = Dense(128, activation='relu')(inputs_2)

#解码
decoded_2 = Dense(256, activation='sigmoid')(encoded_2)
# model 输入到重构映射
autoencoder_2 = Model(input=inputs_2, output=decoded_2)
# 编码器 输入到编码
encoder_2 = Model(input=inputs_2, output=encoded_2)

autoencoder_2.compile(optimizer='adam', loss='binary_crossentropy')

callbacks_list = [ModelCheckpoint('checkpoints/model.h5', period=EPOCHS/5)]

autoencoder_2.fit(encoded_inputs_1, encoded_inputs_1,
                nb_epoch=EPOCHS,
                batch_size=32,
                shuffle=True,
                callbacks=callbacks_list,
                validation_split=0)
print('堆叠自动编码器第二层训练完成')

'''
第三层自动编码器
'''
encoded_inputs_2 = encoder_2.predict(encoded_inputs_1)

inputs_3 = Input(shape=(128, ))
#编码shape[1]->128
encoded_3 = Dense(64, activation='relu')(inputs_3)

#解码
decoded_3 = Dense(128, activation='sigmoid')(encoded_3)
# model 输入到重构映射
autoencoder_3 = Model(input=inputs_3, output=decoded_3)
# 编码器 输入到编码
encoder_3 = Model(input=inputs_3, output=encoded_3)

## 创建解码输入
#encoded_input = Input(shape=(256,))
## 编码器最后一层
#decoder_layer = autoencoder.layers[-1]
## 创建编码器
#decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

autoencoder_3.compile(optimizer='adam', loss='binary_crossentropy')

callbacks_list = [TensorBoard(log_dir='logs/'),ModelCheckpoint('checkpoints/model.h5', period=EPOCHS/5)]

autoencoder_3.fit(encoded_inputs_2, encoded_inputs_2,
                nb_epoch=EPOCHS,
                batch_size=32,
                shuffle=True,
                callbacks=callbacks_list,
                validation_split=0)

print('堆叠自动编码器第三层训练完成')


## 在测试集上进行编码和解码
encoded_signals = encoder_3.predict(encoded_inputs_2)

Labels = get_all_labels()


X_train, y_train = encoded_signals[0:315*2,:], Labels[0:315*2]
X_test, y_test = encoded_signals[315*2:, :], Labels[315*2:]

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

