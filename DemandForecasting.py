import pickle
import numpy as np
import nltk
from nltk.stem import PorterStemmer
import json
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow import keras

from dataBaseDump import DataBaseDump

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

porter = PorterStemmer()


class DemandForecast:

    def forecast_item_model1(self):
        chat = pd.read_csv('model_1.csv')
        train_x = chat[{'age', 'district', 'gender', 'occupation'}]
        # print(train_x.columns)
        train_x = train_x.fillna(0)
        train_x = pd.get_dummies(train_x)
        # print(train_x.columns)
        train_x = np.array(train_x)
        # print(train_x)
        train_y = chat[{'item_code'}]
        train_y = train_y.fillna(0)
        train_y = pd.get_dummies(train_y)
        train_y = np.array(train_y)
        # print(train_y)

        input_length = len(train_x[0:1][0])
        output_length = len(train_y[0:1][0])
        # print("-----------------------")
        # print(train_x[0:1])
        # print(train_y[0:1])
        # print(input_length)
        # print(output_length)
        # print("-----------------------")

        X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.33)
        model = keras.Sequential()
        model.add(keras.layers.Dense(input_length, activation='relu', input_shape=(input_length,)))
        model.add(keras.layers.Dense(input_length, activation='relu'))
        model.add(keras.layers.Dense(input_length, activation='softmax'))
        model.add(keras.layers.Dense(output_length))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        hist = model.fit(train_x, train_y, epochs=500, callbacks=[keras.callbacks.EarlyStopping(patience=1)],
        # hist = model.fit(X_train, y_train, epochs=100, callbacks=[keras.callbacks.EarlyStopping(patience=1)],
                         batch_size=1)
        print("accuracy----------------------------------")
        print(model.evaluate(X_test, y_test))
        print("accuracy----------------------------------")
        model.save('model1.h5', hist)
        model.summary()
        # loss_train = hist.history['loss']
        # accuracy = hist.history['accuracy']
        # plt.plot(loss_train, 'g', label='loss')
        # plt.plot(accuracy, 'b', label='accuracy')
        # plt.title(
        #     'Training and Validation accuracy of forecast_item_category model (batch size = ' + str(1) + ' and epochs = ' + str(
        #         500) + ')')
        #
        # plt.title(
        #     'Training and Validation loss of forecast_item_category model (batch size = ' + str(
        #         1) + ' and epochs = ' + str(
        #         500) + ')')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.show()

    def forecast_item_model2(self):
        chat = pd.read_csv('model_2.csv')
        train_x = chat[
            {'age', 'district', 'gender', 'occupation', 'brand', 'color', 'item_category', 'ram', 'price', 'screen'}]
        train_x = train_x.fillna(0)
        train_x = pd.get_dummies(train_x)
        print(train_x.columns)
        train_x = np.array(train_x)
        for x in train_x:
            print(x)
        train_y = chat[{'item_code'}]
        train_y = train_y.fillna(0)
        train_y = pd.get_dummies(train_y)
        train_y = np.array(train_y)
        # print(train_y)

        input_length = len(train_x[0:1][0])
        output_length = len(train_y[0:1][0])
        # print("-----------------------")
        # print(train_x[0:1])
        # print(train_y[0:1])
        # print(input_length)
        # print(output_length)
        # print("-----------------------")
        X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.33)
        model = keras.Sequential()
        model.add(keras.layers.Dense(input_length, activation='relu', input_shape=(input_length,)))
        model.add(keras.layers.Dense(input_length, activation='relu'))
        model.add(keras.layers.Dense(input_length, activation='softmax'))
        model.add(keras.layers.Dense(output_length))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        hist = model.fit(train_x, train_y, epochs=500, callbacks=[keras.callbacks.EarlyStopping(patience=1)],
                         batch_size=1)
        print("accuracy----------------------------------")
        print(model.evaluate(X_test, y_test))
        print("accuracy----------------------------------")
        model.save('model2.h5', hist)
        model.summary()
        # loss_train = hist.history['loss']
        # accuracy = hist.history['accuracy']
        # plt.plot(loss_train, 'g', label='loss')
        # plt.plot(accuracy, 'b', label='accuracy')
        # plt.title(
        #     'Training and Validation accuracy of forecast_item_category model (batch size = ' + str(1) + ' and epochs = ' + str(
        #         500) + ')')
        #
        # plt.title(
        #     'Training and Validation loss of forecast_item_category model (batch size = ' + str(
        #         1) + ' and epochs = ' + str(
        #         500) + ')')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.show()

    def get_data_by_user_id(self, user_id):
        dt = DataBaseDump()
        user_data = dt.get_query_data(
            "select age,district,gender,occupation ,user_id from user")
        print(user_data)

        # user_data = np.array(user_data)
        user_data = pd.get_dummies(user_data)
        user_data = user_data.fillna(0)
        print(user_data)

        return user_data

    def get_model_1_input_column_names(self):
        chat_model_1 = pd.read_csv('model_1.csv')
        train_x = chat_model_1[{'age', 'district', 'gender', 'occupation'}]
        train_x = train_x.fillna(0)
        train_x = pd.get_dummies(train_x)
        coloumn_names = train_x[0:1].columns
        coloumn_names = np.array(coloumn_names).astype('object')
        col_names = json.dumps(coloumn_names.tolist())
        return col_names

    def get_model_2_input_column_names(self):
        chat_model_2 = pd.read_csv('model_2.csv')
        train_x = chat_model_2[
            {'age', 'district', 'gender', 'occupation', 'brand', 'color', 'item_category', 'ram', 'price', 'screen'}]
        train_x = train_x.fillna(0)
        train_x = pd.get_dummies(train_x)
        coloumn_names = train_x[0:1].columns
        coloumn_names = np.array(coloumn_names).astype('object')
        col_names = json.dumps(coloumn_names.tolist())
        return col_names

    def get_model_1_output_column_names(self):
        chat_model_1 = pd.read_csv('model_1.csv')
        train_y = chat_model_1[{'item_code'}]
        train_y = pd.get_dummies(train_y)
        coloumn_names = train_y[0:1].columns
        coloumn_names = np.array(coloumn_names).astype('object')
        col_names = json.dumps(coloumn_names.tolist())
        return col_names

    def get_model_2_output_column_names(self):
        chat_model_1 = pd.read_csv('model_2.csv')
        train_y = chat_model_1[{'item_code'}]
        train_y = pd.get_dummies(train_y)
        coloumn_names = train_y[0:1].columns
        coloumn_names = np.array(coloumn_names).astype('object')
        col_names = json.dumps(coloumn_names.tolist())
        return col_names


    # def forecast_item_category_demand_model(self):
    #     ignore_letters = ['?', '!', '.', ',']
    #
    #     chat_csv = pd.read_csv('chat1.csv')
    #     chat_column = chat_csv['chat_message']
    #     words_loaded = pickle.load(open('words.pkl', 'rb'))
    #     train_y = chat_csv[
    #         {'item_category'}]
    #     train_x = chat_csv[{'gender', 'chat_member', 'age'}]
    #
    #     bag_x = []
    #     classes = pickle.load(open('classes.pkl', 'rb'))
    #     for chat_row in chat_column:
    #         bag = []
    #         word_list = nltk.word_tokenize(chat_row)
    #         word_list = [porter.stem(word) for word in word_list if word not in ignore_letters]
    #         word_list = sorted(set(word_list))
    #         for word in words_loaded:
    #             bag.append(1) if porter.stem(word.lower()) in word_list else bag.append(0)
    #         bag_x.append(bag)
    #     bag_x = np.array(bag_x)
    #     train_x = np.concatenate([train_x, bag_x], axis=1).astype('int32')
    #
    #     train_y = np.array(train_y).astype('int32')
    #
    #     test_x = train_x[0]
    #
    #     batch_size = 5
    #     number_of_epochs = 50
    #     model = keras.models.Sequential()
    #     model.add(keras.layers.Dense(len(train_x[0]), activation='relu', input_shape=(len(train_x[0]),)))
    #     model.add(keras.layers.Dense(len(train_x[0]), activation='relu'))
    #     model.add(keras.layers.Dense(len(train_y[0]), activation='relu'))
    #
    #     # model.compile(optimizer='adam', loss='mean_squared_error', metrics='accuracy')
    #     model.compile(optimizer='adam', loss='mean_squared_error')
    #
    #     hist = model.fit(np.array(train_x), np.array(train_y), epochs=number_of_epochs, batch_size=batch_size,
    #                      callbacks=[keras.callbacks.EarlyStopping(patience=3)])
    #     model.save('forecast_item_category_demand_model.h5', hist)
    #     # predicted_data = model.predict(test_x.reshape(1, len(train_x[0])), batch_size=1)
    #     # print(train_x[1])
    #     # print(train_y[1])
    #     # print(predicted_data)
    #     model.summary()
    #     # loss_train = hist.history['loss']
    #     # accuracy = hist.history['accuracy']
    #     # plt.plot(loss_train, 'g', label='loss')
    #     # plt.plot(accuracy, 'b', label='accuracy')
    #     # plt.title(
    #     #     'Training and Validation accuracy of forecast_item_category model (batch size = ' + str(batch_size) + ' and epochs = ' + str(
    #     #         number_of_epochs) + ')')
    #
    #     # plt.title(
    #     #     'Training and Validation loss of forecast_item_category model (batch size = ' + str(
    #     #         batch_size) + ' and epochs = ' + str(
    #     #         number_of_epochs) + ')')
    #     # plt.xlabel('Epochs')
    #     # plt.ylabel('Loss')
    #     # plt.legend()
    #     # plt.show()
    #
    # def convert_data_by_ip_address_chat(self, ip_address):
    #     ignore_letters = ['?', '!', '.', ',']
    #
    #     dt = DataBaseDump()
    #
    #     fetched_data_message = dt.get_query_data(
    #         "select cm.chat_message from chat_message cm where cm.chat_member=1 and cm.ip_address='" + ip_address + "' order by cm.chat_id desc limit 1 ")
    #
    #     fetched_data = dt.get_query_data(
    #         "select cm.chat_member chat_member  from cart_item ci inner join chat_message cm on ci.ip_address =  cm.ip_address inner join item im on ci.item_id = im.item_id inner join order_details od  on od.order_id= ci.order_detail_id where cm.chat_member=1 and  ci.ip_address='" + ip_address + "'  order by cm.chat_id desc limit 1  ")
    #
    #     if len(fetched_data_message) > 0 and len(fetched_data) > 0:
    #         fetched_data = np.array(fetched_data)
    #         words_loaded = pickle.load(open('words.pkl', 'rb'))
    #         bag_x = []
    #         for chat_row in fetched_data_message:
    #             bag = []
    #             word_list = nltk.word_tokenize(chat_row[0])
    #             word_list = [porter.stem(word) for word in word_list if word not in ignore_letters]
    #             word_list = sorted(set(word_list))
    #             for word in words_loaded:
    #                 bag.append(1) if porter.stem(word.lower()) in word_list else bag.append(0)
    #             bag_x.append(bag)
    #         bag_x = np.array(bag_x)
    #         train_x = np.concatenate([fetched_data, bag_x], axis=1)
    #
    #         return train_x
    #     else:
    #         return []
