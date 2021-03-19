import pickle
import numpy as np
import nltk
from nltk.stem import PorterStemmer

import pandas as pd
from tensorflow import keras

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
train_x = np.concatenate([train_x, bag_x], axis=1)
train_y = np.array(train_y)
test_x = train_x[0]
model = keras.models.Sequential()
model.add(keras.layers.Dense(1000, activation='relu', input_shape=(len(train_x[0]),)))
model.add(keras.layers.Dense(1000, activation='relu'))
model.add(keras.layers.Dense(1000, activation='relu'))
model.add(keras.layers.Dense(1000, activation='relu'))
model.add(keras.layers.Dense(len(train_y[0])))

model.compile(optimizer='adam', loss='mean_squared_error', metrics='accuracy')

model.fit(train_x, train_y, epochs=30, callbacks=[keras.callbacks.EarlyStopping(patience=3)])

predicted_data = model.predict(test_x.reshape(1, len(train_x[0])), batch_size=1)
print(train_y[0])
print(predicted_data)
