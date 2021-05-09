import pickle
import numpy as np
import nltk
from nltk.stem import PorterStemmer

import pandas as pd
from tensorflow import keras

from dataBaseDump import DataBaseDump

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model, model_to_dot
from IPython.display import SVG

porter = PorterStemmer()

ignore_letters = ['?', '!', '.', ',']

chat_csv = pd.read_csv('chat1.csv')
chat_column = chat_csv['chat_message']
words_loaded = pickle.load(open('words.pkl', 'rb'))
train_y = chat_csv[
    {'item_category', 'item_discount', 'order_quantity', 'item_price', 'order_total_amount', 'order_status'}]
train_x = chat_csv[{'gender', 'chat_member'}]

bag_x = []
classes = pickle.load(open('classes.pkl', 'rb'))
for chat_row in chat_column:
    bag = []
    word_list = nltk.word_tokenize(chat_row)
    word_list = [porter.stem(word) for word in word_list if word not in ignore_letters]
    word_list = sorted(set(word_list))
    for word in words_loaded:
        bag.append(1) if porter.stem(word.lower()) in word_list else bag.append(0)
    bag_x.append(bag)
bag_x = np.array(bag_x)
train_x = np.concatenate([train_x, bag_x], axis=1).astype('int32')

train_y = np.array(train_y).astype('int32')

test_x = train_x[0]

model = keras.models.Sequential()
model.add(keras.layers.Dense(1024, activation='relu', input_shape=(len(train_x[0]),)))
model.add(keras.layers.Dense(512, activation='relu'))
# model.add(keras.layers.Dense(512, activation='relu'))
# model.add(keras.layers.Dense(512, activation='relu'))
# model.add(keras.layers.Dense(64, activation='relu'))
# model.add(keras.layers.Dense(64, activation='relu'))
# model.add(keras.layers.Dense(1000, activation='relu'))
# model.add(keras.layers.Dense(1000, activation='relu'))
model.add(keras.layers.Dense(len(train_y[0])))

lmt = 50

tr_x = np.array(train_x)[0:len(train_x) - lmt]
tr_y = np.array(train_y)[0:len(train_y) - lmt]

t_x = np.array(train_x)[len(train_x) - lmt:len(train_x)]
t_y = np.array(train_y)[len(train_y) - lmt:len(train_y)]

model.compile(optimizer='adam', loss='mean_squared_error', metrics='accuracy')

hist = model.fit(tr_x, tr_y, epochs=10, batch_size=50, validation_data=(t_x, t_y))

model.save('forecast_demand_model.h5', hist)
predicted_data = model.predict(test_x.reshape(1, len(train_x[0])), batch_size=1)

# print(hist.history)
# print(hist.history['accuracy'])
# loss_train = hist.history['loss']
accuracy = hist.history['accuracy']
# # epochs = range(0, 9)
# plt.plot(loss_train, 'g', label='loss')
plt.plot(accuracy, 'b', label='accuracy')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# plt.show()
# model.summary()

# plot_model(model)
# model_to_dot(model)
# SVG(model_to_dot(model).create(prog='dot', format='svg'))
# print(model.layers[0].weights)
# print(model.layers[1].weights)
# print(model.layers[2].weights)
print(model.weights)