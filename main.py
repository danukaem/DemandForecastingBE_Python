import random
import numpy as np
import nltk
from flask import Flask, jsonify, json, request
from flask_cors import CORS

from DemandForecasting import DemandForecast
from chatBotModel import ChatBotModel
from dataBaseDump import DataBaseDump
from demo6 import Demo6
from initialDataLoading import InitialFileLoader
from tensorflow import keras

app = Flask(__name__)
CORS(app)


def clean_up_sentence(sentence):
    ifl = InitialFileLoader()
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [ifl.getLemmatizer().lemmatize(ifl.getLemmatizer().lemmatize(
        ifl.getLemmatizer().lemmatize(ifl.getLemmatizer().lemmatize(word.strip().lower(), pos='n'), pos='v'), pos='a'),
        pos='r') for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    ifl = InitialFileLoader()

    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(ifl.getWords())
    for w in sentence_words:
        for i, word in enumerate(ifl.getWords()):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    ifl = InitialFileLoader()

    bow = bag_of_words(sentence)
    res = ifl.getModel().predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.8
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': ifl.getClasses()[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intent_list, intents_json):
    result = 'I do not understand...'
    initial_probability = 0.0
    tag = ''

    for prob in intent_list:
        if float(prob['probability']) > initial_probability:
            initial_probability = float(prob['probability'])
            tag = prob['intent']

    list_of_intents = intents_json['intents']

    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


@app.route('/chat', methods=['GET'])
def get_chat_response():
    ifl = InitialFileLoader()
    query_parameters = request.args
    message = query_parameters.get('message')
    ints = predict_class(message)
    res = get_response(ints, ifl.getIntents())
    return jsonify({"robotMessage": res})


@app.route('/getForecastedItems', methods=['GET'])
def getForecastedItems():
    d6 = Demo6()
    # query_parameters = request.args
    # message = query_parameters.get('message')
    # ints = predict_class(message)
    # res = get_response(ints, ifl.getIntents())
    d6.get_prediction_data()
    return jsonify({"items": '{}'.format(d6.get_prediction_data())})


@app.route('/getForecastedItemsCodes', methods=['GET'])
def getForecastedItemsCodes():
    d6 = Demo6()
    # query_parameters = request.args
    # message = query_parameters.get('message')
    # ints = predict_class(message)
    # res = get_response(ints, ifl.getIntents())
    # d6.get_model_1_column_names()
    # return jsonify({"itemCodes": '{}'.format(d6.get_model_1_column_names())})
    return jsonify({"itemCodes": '{}'.format(d6.get_columns())})


@app.route('/generateChatModel', methods=['GET'])
def generate_chat_model():
    data_dump_model = DataBaseDump()
    data_dump_model.create_model_1_csv_file()
    data_dump_model.create_model_2_csv_file()
    demand_forecast = DemandForecast()
    demand_forecast.forecast_item_model1()
    demand_forecast.forecast_item_model2()
    return 'success'


@app.route('/getModel1InputNames', methods=['GET'])
def getModel1InputNames():
    df = DemandForecast()
    return jsonify({"itemCodes": '{}'.format(df.get_model_1_input_column_names())})


@app.route('/getModel2InputNames', methods=['GET'])
def getModel2InputNames():
    df = DemandForecast()
    return jsonify({"itemCodes": '{}'.format(df.get_model_2_input_column_names())})


@app.route('/getModel1OutPutNames', methods=['GET'])
def getModel1OutPutNames():
    df = DemandForecast()
    return jsonify({"itemCodes": '{}'.format(df.get_model_1_output_column_names())})


@app.route('/getModel2OutPutNames', methods=['GET'])
def getModel2OutPutNames():
    df = DemandForecast()
    return jsonify({"itemCodes": '{}'.format(df.get_model_2_output_column_names())})


@app.route('/getForecastedItemsByUserId', methods=['POST'])
def getForecastedItemsByUserId():
    request_json = request.get_json()
    request_data = request_json['array']
    request_data = np.array(request_data)
    model_saved = keras.models.load_model('model2.h5')
    pred = model_saved.predict(request_data.reshape(1, len(request_data)), batch_size=1)
    return jsonify({"abc": '{}'.format(pred)})


@app.route('/getForecastedItemsNewUser', methods=['POST'])
def getForecastedItemsNewUser():
    request_json = request.get_json()
    request_data = request_json['array']
    request_data = np.array(request_data)
    model_saved = keras.models.load_model('model1.h5')
    pred = model_saved.predict(request_data.reshape(1, len(request_data)), batch_size=1)
    return jsonify({"abc": '{}'.format(pred)})


print("GO! bot is running!")

if __name__ == '__main__':
    app.run(debug=True)
