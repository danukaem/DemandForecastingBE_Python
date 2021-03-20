import random
import numpy as np
import nltk
from flask import Flask, jsonify, json, request
from flask_cors import CORS

from DemandForecasting import DemandForecast
from chatBotModel import ChatBotModel
from dataBaseDump import DataBaseDump
from initialDataLoading import InitialFileLoader
from tensorflow import keras

app = Flask(__name__)
CORS(app)


def clean_up_sentence(sentence):
    ifl = InitialFileLoader()
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [ifl.getLemmatizer().stem(word) for word in sentence_words]
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
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': ifl.getClasses()[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intent_list, intents_json):
    tag = intent_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
        else:
            result = 'error ...'
    return result


@app.route('/chat', methods=['GET'])
def get_chat_response():
    ifl = InitialFileLoader()
    query_parameters = request.args
    message = query_parameters.get('message')
    print(message)
    ints = predict_class(message)
    res = get_response(ints, ifl.getIntents())
    print(res)

    return jsonify({"userMessage": res})


@app.route('/generateChatModel', methods=['GET'])
def generate_chat_model():
    chat_model = ChatBotModel()
    data_dump_model = DataBaseDump()
    data_dump_model.create_data_dump()
    res = chat_model.generatechatmodel()
    demand_forecast = DemandForecast()
    demand_forecast.forecast_demand()
    return res


@app.route('/getRecommendCartItems', methods=['GET'])
def get_recommend_cart_items():
    query_parameters = request.args
    user_id = query_parameters.get('userId')
    df = DemandForecast()
    test_x = df.convert_data_by_user_id(user_id)
    test_x = np.unique(test_x, axis=0)
    model_saved = keras.models.load_model('forecast_model.h5')
    prediction = model_saved.predict(test_x.reshape(len(test_x), len(test_x[0])), batch_size=1)
    prediction = json.dumps(prediction.tolist())
    return jsonify({"forecastResults": '{}'.format(prediction)})


# manually generate chat bot related models
# demo = ChatBotModel()
# demo.generatechatmodel()

@app.route('/demandForecasting', methods=['POST'])
def demand_forecasting():
    request_data = request.get_json()
    ip_address = request_data['ipAddress']
    df = DemandForecast()
    test_x = df.convert_data_by_ip_address(ip_address)
    test_x = np.unique(test_x, axis=0)
    model_saved = keras.models.load_model('forecast_model.h5')
    pred = model_saved.predict(test_x.reshape(len(test_x), len(test_x[0])), batch_size=1)
    return '{}'.format(pred)


print("GO! bot is running!")

if __name__ == '__main__':
    app.run(debug=True)
