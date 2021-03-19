import pickle
import numpy as np
import nltk
from nltk.stem import PorterStemmer

import pandas as pd
from tensorflow import keras

from dataBaseDump import DataBaseDump

porter = PorterStemmer()


class DemandForecast:
    def forecast_demand(self):
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

        hist = model.fit(train_x, train_y, epochs=30, callbacks=[keras.callbacks.EarlyStopping(patience=3)])
        model.save('forecast_model.h5', hist)
        print('length : ', len(train_x[0]))
        predicted_data = model.predict(test_x.reshape(1, len(train_x[0])), batch_size=1)
        print(train_x[0])
        print(train_y[0])
        print(predicted_data)

    def convert_data(self, ip_address):
        ignore_letters = ['?', '!', '.', ',']

        dt = DataBaseDump()
        fetched_data_message = dt.get_query_data(
            "select  cm.chat_message chat_message  from cart_item ci inner join chat_message cm on ci.ip_address =  cm.ip_address inner join item im on ci.item_id = im.item_id "
            "inner join user usr on usr.user_id=ci.user_id inner join order_details od  on od.order_id= ci.order_detail_id where ci.ip_address='" + ip_address + "' ")

        fetched_data = dt.get_query_data(
            "select   usr.gender gender,cm.chat_member chat_member  from cart_item ci inner join chat_message cm on ci.ip_address =  cm.ip_address inner join item im on ci.item_id = im.item_id "
            "inner join user usr on usr.user_id=ci.user_id inner join order_details od  on od.order_id= ci.order_detail_id where ci.ip_address='" + ip_address + "' ")

        fetched_data = np.array(fetched_data)
        words_loaded = pickle.load(open('words.pkl', 'rb'))
        bag_x = []
        for chat_row in fetched_data_message:
            print(chat_row[0])
            bag = []
            word_list = nltk.word_tokenize(chat_row[0])
            word_list = [porter.stem(word) for word in word_list if word not in ignore_letters]
            word_list = sorted(set(word_list))
            for word in words_loaded:
                bag.append(1) if porter.stem(word.lower()) in word_list else bag.append(0)
            bag_x.append(bag)
        bag_x = np.array(bag_x)
        train_x = np.concatenate([fetched_data, bag_x], axis=1)

        return train_x
