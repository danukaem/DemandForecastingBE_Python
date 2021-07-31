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

ignore_letters = ['?', '!', '.', ',']

chat = pd.read_csv('item.csv')
chat_column = chat['brand']
# print(chat_csv)

# 1st method
# chat_csv['brand_factorized'] = pd.factorize(chat_csv.brand)[0]

# 2 nd method === label encording
# chat_csv['brand_encorded'] = le.fit_transform(chat_csv.brand)
# print(chat_csv)


# 3rd method  === one hot encording
# br = pd.get_dummies(chat_csv.brand)
# print(br)
# br = pd.concat([chat_csv, br], axis=1)
# print(br)
# br = br.drop('brand',axis=1)
# print(br)

# print(chat_csv.brand.value_counts())
# print(chat_csv['brand'].value_counts())
# print(chat_csv[0])

train_x = chat[{'ram', 'screen', 'age'}]
train_y = chat[{'item_code'}]
# print(train_x)
# print(train_x[0:1])
# print(len(train_x[0:1]))
# print(train_y)
train_y = pd.get_dummies(train_y)
# print(train_y.head(2))
coloumn_names = train_y[0:1].columns
coloumn_names = np.array(coloumn_names).astype('object')
print(coloumn_names)
print(coloumn_names.dtype)
a = train_y[0:1]
a = np.array(a).astype('int32')
# print("length is :::::::::::::::: = ", len(a[0]))
# print("shape is :::::::::::::::: = ", a.shape)
# print('******************************')
# for b in a[0]:
#     print(b)
# print('******************************')
# print(train_x[0:1])
# print(len(train_x.shape))
# print("777777777777777777777777777777")
# arr = np.arange(5)
# len_arr = len(arr)
# print("Array elements: ", arr)
# print("Length of NumPy array:", len_arr)
# print("777777777777777777777777777777")

model = keras.Sequential()
model.add(keras.layers.Dense(3, activation='relu', input_shape=(3,)))
model.add(keras.layers.Dense(5, activation='relu'))
model.add(keras.layers.Dense(7))
model.compile(optimizer='adam', loss='mean_squared_error')
#
model.fit(train_x, train_y, epochs=10, callbacks=[keras.callbacks.EarlyStopping(patience=1)], batch_size=1)
#
test_x1 = np.array([15, 2, 30])
predi = model.predict(test_x1.reshape(1, 3), batch_size=1)
# print(predi)
predi = np.array(predi)

aa = {
    'col_names': coloumn_names,
    'predicted_values': predi
}

print("------------predicted aa")
print(aa)
print("------------predicted aa")
