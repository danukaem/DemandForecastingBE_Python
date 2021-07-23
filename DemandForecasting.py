import pickle
import numpy as np
import nltk
from nltk.stem import PorterStemmer

import pandas as pd
from tensorflow import keras

from dataBaseDump import DataBaseDump

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

porter = PorterStemmer()


class DemandForecast:

    def forecast_item_category_demand_model(self):
        ignore_letters = ['?', '!', '.', ',']

        chat_csv = pd.read_csv('chat1.csv')
        chat_column = chat_csv['chat_message']
        words_loaded = pickle.load(open('words.pkl', 'rb'))
        train_y = chat_csv[
            {'item_category'}]
        train_x = chat_csv[{'gender', 'chat_member', 'age'}]

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

        batch_size = 5
        number_of_epochs = 50
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(len(train_x[0]), activation='relu', input_shape=(len(train_x[0]),)))
        model.add(keras.layers.Dense(len(train_x[0]), activation='relu'))
        model.add(keras.layers.Dense(len(train_y[0]), activation='relu'))

        # model.compile(optimizer='adam', loss='mean_squared_error', metrics='accuracy')
        model.compile(optimizer='adam', loss='mean_squared_error')

        hist = model.fit(np.array(train_x), np.array(train_y), epochs=number_of_epochs, batch_size=batch_size,
                         callbacks=[keras.callbacks.EarlyStopping(patience=3)])
        model.save('forecast_item_category_demand_model.h5', hist)
        # predicted_data = model.predict(test_x.reshape(1, len(train_x[0])), batch_size=1)
        # print(train_x[1])
        # print(train_y[1])
        # print(predicted_data)
        model.summary()
        # loss_train = hist.history['loss']
        # accuracy = hist.history['accuracy']
        # plt.plot(loss_train, 'g', label='loss')
        # plt.plot(accuracy, 'b', label='accuracy')
        # plt.title(
        #     'Training and Validation accuracy of forecast_item_category model (batch size = ' + str(batch_size) + ' and epochs = ' + str(
        #         number_of_epochs) + ')')

        # plt.title(
        #     'Training and Validation loss of forecast_item_category model (batch size = ' + str(
        #         batch_size) + ' and epochs = ' + str(
        #         number_of_epochs) + ')')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.show()


    def convert_data_by_ip_address(self, ip_address):
        ignore_letters = ['?', '!', '.', ',']

        dt = DataBaseDump()
        fetched_data_message = dt.get_query_data(
            "select   cm.chat_message  from cart_item ci inner join chat_message cm on ci.ip_address =  cm.ip_address inner join item im on ci.item_id = im.item_id inner join order_details od  on od.order_id= ci.order_detail_id where cm.chat_member=1 and  ci.ip_address='" + ip_address + "'")

        fetched_data = dt.get_query_data(
            "select cm.chat_member chat_member  from cart_item ci inner join chat_message cm on ci.ip_address =  cm.ip_address inner join item im on ci.item_id = im.item_id inner join order_details od  on od.order_id= ci.order_detail_id where  cm.chat_member=1 and ci.ip_address='" + ip_address + "' ")

        if len(fetched_data_message) > 0 and len(fetched_data) > 0:
            fetched_data = np.array(fetched_data)
            words_loaded = pickle.load(open('words.pkl', 'rb'))
            bag_x = []
            for chat_row in fetched_data_message:
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
        else:
            return []

    def convert_data_by_ip_address_chat(self, ip_address):
        ignore_letters = ['?', '!', '.', ',']

        dt = DataBaseDump()

        fetched_data_message = dt.get_query_data(
            "select cm.chat_message from chat_message cm where cm.chat_member=1 and cm.ip_address='" + ip_address + "' order by cm.chat_id desc limit 1 ")

        fetched_data = dt.get_query_data(
            "select cm.chat_member chat_member  from cart_item ci inner join chat_message cm on ci.ip_address =  cm.ip_address inner join item im on ci.item_id = im.item_id inner join order_details od  on od.order_id= ci.order_detail_id where cm.chat_member=1 and  ci.ip_address='" + ip_address + "'  order by cm.chat_id desc limit 1  ")

        if len(fetched_data_message) > 0 and len(fetched_data) > 0:
            fetched_data = np.array(fetched_data)
            words_loaded = pickle.load(open('words.pkl', 'rb'))
            bag_x = []
            for chat_row in fetched_data_message:
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
        else:
            return []

    def convert_data_by_user_id(self, user_id):
        ignore_letters = ['?', '!', '.', ',']

        dt = DataBaseDump()
        fetched_data_message = dt.get_query_data(
            "select  cm.chat_message chat_message  from cart_item ci inner join chat_message cm on ci.ip_address =  cm.ip_address inner join item im on ci.item_id = im.item_id "
            "inner join user usr on usr.user_id=ci.user_id inner join order_details od  on od.order_id= ci.order_detail_id where  cm.chat_member=1 and usr.user_id='" + user_id + "' ")

        fetched_data = dt.get_query_data(
            "select   usr.gender gender,cm.chat_member chat_member ,usr.age age from cart_item ci inner join chat_message cm on ci.ip_address =  cm.ip_address inner join item im on ci.item_id = im.item_id "
            "inner join user usr on usr.user_id=ci.user_id inner join order_details od  on od.order_id= ci.order_detail_id where cm.chat_member=1 and  usr.user_id='" + user_id + "' ")

        if len(fetched_data_message) > 0 and len(fetched_data) > 0:
            fetched_data = np.array(fetched_data)
            words_loaded = pickle.load(open('words.pkl', 'rb'))
            bag_x = []
            for chat_row in fetched_data_message:
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
        else:
            return []

    def convert_data_by_user_id_chat(self, user_id):
        ignore_letters = ['?', '!', '.', ',']

        dt = DataBaseDump()
        fetched_data_message = dt.get_query_data(
            "select cm.chat_message from chat_message cm where cm.chat_member=1 and cm.user_id='" + user_id + "' order by cm.chat_id desc limit 1")

        fetched_data = dt.get_query_data(
            "select   usr.gender gender,cm.chat_member chat_member ,usr.age age from cart_item ci inner join chat_message cm on ci.ip_address =  cm.ip_address inner join item im on ci.item_id = im.item_id "
            "inner join user usr on usr.user_id=ci.user_id inner join order_details od  on od.order_id= ci.order_detail_id where cm.chat_member=1 and usr.user_id='" + user_id + "'   order by cm.chat_id desc limit 1 ")

        if len(fetched_data_message) > 0 and len(fetched_data) > 0:
            fetched_data = np.array(fetched_data)
            words_loaded = pickle.load(open('words.pkl', 'rb'))
            bag_x = []
            for chat_row in fetched_data_message:
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
        else:
            return []
