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

    return jsonify({"robotMessage": res})


@app.route('/generateChatModel', methods=['GET'])
def generate_chat_model():
    chat_model = ChatBotModel()
    data_dump_model = DataBaseDump()
    data_dump_model.create_data_dump()
    data_dump_model.create_data_dump_ip_address()
    res = chat_model.generatechatmodel()
    demand_forecast = DemandForecast()
    demand_forecast.forecast_demand_model()
    demand_forecast.forecast_item_category_demand_model()
    demand_forecast.forecast_item_category_demand_model_without_user_id()
    demand_forecast.forecast_item_price_demand_model()
    demand_forecast.forecast_item_discount_demand_model()
    demand_forecast.forecast_order_quantity_demand_model()
    demand_forecast.forecast_order_total_amount_demand_model()
    demand_forecast.forecast_order_status_demand_model()
    return res


@app.route('/getRecommendCartItems', methods=['GET'])
def get_recommend_cart_items():
    query_parameters = request.args
    user_id = query_parameters.get('userId')
    df = DemandForecast()
    test_x = df.convert_data_by_user_id(user_id)
    if len(test_x) > 0:
        test_x = np.unique(test_x, axis=0)
        model_saved = keras.models.load_model('forecast_model.h5')
        prediction = model_saved.predict(test_x.reshape(len(test_x), len(test_x[0])), batch_size=1)
        prediction = json.dumps(prediction.tolist())
        print(prediction)
        return jsonify({"forecastResults": '{}'.format(prediction)})
    else:
        return 'no data'


# manually generate chat bot related models
# demo = ChatBotModel()
# demo.generatechatmodel()

@app.route('/demandForecasting', methods=['GET'])
def demand_forecasting():
    request_data = request.args
    ip_address = request_data.get('ipAddress')
    df = DemandForecast()
    test_x = df.convert_data_by_ip_address(ip_address)

    if len(test_x) > 0:
        test_x = np.unique(test_x, axis=0)
        model_saved = keras.models.load_model('forecast_demand_model.h5')
        pred = model_saved.predict(test_x.reshape(len(test_x), len(test_x[0])), batch_size=1)
        return jsonify({"forecastResults": '{}'.format(pred)})
    else:
        return 'no data'


@app.route('/itemCategoryDemandForecastingByIpAddress', methods=['GET'])
def item_category_demand_forecasting_ipaddress():
    request_data = request.args
    ip_address = request_data.get('ipAddress')
    df = DemandForecast()
    test_x = df.convert_data_by_ip_address(ip_address)

    if len(test_x) > 0:
        test_x = np.unique(test_x, axis=0)
        model_saved = keras.models.load_model('forecast_item_category_demand_model_without_user_id.h5')
        pred = model_saved.predict(test_x.reshape(len(test_x), len(test_x[0])), batch_size=1)
        return jsonify({"forecastResults": '{}'.format(pred)})
    else:
        return jsonify({"forecastResults": '{}'.format('')})


@app.route('/itemCategoryDemandForecastingByUserId', methods=['GET'])
def item_category_demand_forecasting_by_userid():
    request_data = request.args
    user_id = request_data.get('userId')
    df = DemandForecast()
    test_x = df.convert_data_by_user_id(user_id)

    if len(test_x) > 0:
        test_x = np.unique(test_x, axis=0)
        model_saved = keras.models.load_model('forecast_item_category_demand_model.h5')
        pred = model_saved.predict(test_x.reshape(len(test_x), len(test_x[0])), batch_size=1)
        print(pred)
        return jsonify({"forecastResults": '{}'.format(pred)})
    else:
        return 'no data'


@app.route('/itemCategoryDemandForecastingByIpAddressChat', methods=['GET'])
def item_category_demand_forecasting_ipaddress_chat():
    request_data = request.args
    ip_address = request_data.get('ipAddress')
    df = DemandForecast()
    test_x = df.convert_data_by_ip_address_chat(ip_address)

    if len(test_x) > 0:
        test_x = np.unique(test_x, axis=0)
        model_saved = keras.models.load_model('forecast_item_category_demand_model_without_user_id.h5')
        pred = model_saved.predict(test_x.reshape(len(test_x), len(test_x[0])), batch_size=1)
        return jsonify({"forecastResults": '{}'.format(pred)})
    else:
        return jsonify({"forecastResults": '{}'.format('')})


@app.route('/itemCategoryDemandForecastingByUserIdChat', methods=['GET'])
def item_category_demand_forecasting_by_userid_chat():
    request_data = request.args
    user_id = request_data.get('userId')
    df = DemandForecast()
    test_x = df.convert_data_by_user_id_chat(user_id)

    if len(test_x) > 0:
        test_x = np.unique(test_x, axis=0)
        model_saved = keras.models.load_model('forecast_item_category_demand_model.h5')
        pred = model_saved.predict(test_x.reshape(len(test_x), len(test_x[0])), batch_size=1)
        print(pred)
        return jsonify({"forecastResults": '{}'.format(pred)})
    else:
        return 'no data'


@app.route('/itemDiscountDemandForecasting', methods=['GET'])
def item_discount_demand_forecasting():
    request_data = request.args
    ip_address = request_data.get('ipAddress')
    df = DemandForecast()
    test_x = df.convert_data_by_ip_address(ip_address)

    if len(test_x) > 0:
        test_x = np.unique(test_x, axis=0)
        model_saved = keras.models.load_model('forecast_item_discount_demand_model.h5')
        pred = model_saved.predict(test_x.reshape(len(test_x), len(test_x[0])), batch_size=1)
        return jsonify({"forecastResults": '{}'.format(pred)})
    else:
        return 'no data'


@app.route('/orderQuantityDemandForecasting', methods=['GET'])
def order_quantity_demand_forecasting():
    request_data = request.args
    ip_address = request_data.get('ipAddress')
    df = DemandForecast()
    test_x = df.convert_data_by_ip_address(ip_address)

    if len(test_x) > 0:
        test_x = np.unique(test_x, axis=0)
        model_saved = keras.models.load_model('forecast_order_quantity_demand_model.h5')
        pred = model_saved.predict(test_x.reshape(len(test_x), len(test_x[0])), batch_size=1)
        return jsonify({"forecastResults": '{}'.format(pred)})
    else:
        return 'no data'


@app.route('/itemPriceDemandForecasting', methods=['GET'])
def item_price_demand_forecasting():
    request_data = request.args
    ip_address = request_data.get('ipAddress')
    df = DemandForecast()
    test_x = df.convert_data_by_ip_address(ip_address)

    if len(test_x) > 0:
        test_x = np.unique(test_x, axis=0)
        model_saved = keras.models.load_model('forecast_item_price_demand_model.h5')
        pred = model_saved.predict(test_x.reshape(len(test_x), len(test_x[0])), batch_size=1)
        return jsonify({"forecastResults": '{}'.format(pred)})
    else:
        return 'no data'


@app.route('/orderTotalDemandForecasting', methods=['GET'])
def order_total_demand_forecasting():
    request_data = request.args
    ip_address = request_data.get('ipAddress')
    df = DemandForecast()
    test_x = df.convert_data_by_ip_address(ip_address)

    if len(test_x) > 0:
        test_x = np.unique(test_x, axis=0)
        model_saved = keras.models.load_model('forecast_order_total_amount_demand_model.h5')
        pred = model_saved.predict(test_x.reshape(len(test_x), len(test_x[0])), batch_size=1)
        return jsonify({"forecastResults": '{}'.format(pred)})
    else:
        return 'no data'


@app.route('/orderStatusDemandForecasting', methods=['GET'])
def order_status_demand_forecasting():
    request_data = request.args
    ip_address = request_data.get('ipAddress')
    df = DemandForecast()
    test_x = df.convert_data_by_ip_address(ip_address)

    if len(test_x) > 0:
        test_x = np.unique(test_x, axis=0)
        model_saved = keras.models.load_model('forecast_order_status_demand_model.h5')
        pred = model_saved.predict(test_x.reshape(len(test_x), len(test_x[0])), batch_size=1)
        return jsonify({"forecastResults": '{}'.format(pred)})
    else:
        return 'no data'


print("GO! bot is running!")

if __name__ == '__main__':
    app.run(debug=True)
