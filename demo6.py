import json
import pickle
import numpy as np
import nltk
from nltk.stem import PorterStemmer

import pandas as pd
from tensorflow import keras

from dataBaseDump import DataBaseDump

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
porter = PorterStemmer()


class Demo6:
    def get_prediction_data(self):

        chat = pd.read_csv('item.csv')
        train_x = chat[{'ram', 'screen', 'age'}]
        train_y = chat[{'item_code'}]
        train_y = pd.get_dummies(train_y)
        train_x = train_x.fillna(0)
        # print(train_y.head(2))
        coloumn_names = train_y[0:1].columns
        coloumn_names = np.array(coloumn_names).astype('object')
        print(coloumn_names)
        print(coloumn_names.dtype)
        a = train_y[0:1]
        a = np.array(a).astype('int32')
        print(train_x)
        model = keras.Sequential()
        model.add(keras.layers.Dense(3, activation='relu', input_shape=(3,)))
        model.add(keras.layers.Dense(5, activation='relu'))
        model.add(keras.layers.Dense(7))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(train_x, train_y, epochs=10, callbacks=[keras.callbacks.EarlyStopping(patience=1)], batch_size=1)
        test_x1 = np.array([15, 2, 30])
        predi = model.predict(test_x1.reshape(1, 3), batch_size=1)
        predi = np.array(predi)

        prediction = json.dumps(predi.tolist())
        return prediction

    def get_columns(self):
        chat = pd.read_csv('item.csv')
        train_x = chat[{'ram', 'screen', 'age'}]
        train_y = chat[{'item_code'}]
        train_y = pd.get_dummies(train_y)
        # print(train_y.head(2))
        coloumn_names = train_y[0:1].columns
        coloumn_names = np.array(coloumn_names).astype('object')
        col_names = json.dumps(coloumn_names.tolist())
        print(col_names)
        return col_names