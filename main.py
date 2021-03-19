import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

from flask import Flask, jsonify, json, request
from flask_cors import CORS

from chatBotModel import ChatBotModel
from initialDataLoading import InitialFileLoader

app = Flask(__name__)

CORS(app)


# lemmatizer = WordNetLemmatizer()
# intents = json.loads(open('intentCombined.json').read())
#
# words = pickle.load(open('words.pkl', 'rb'))
# classes = pickle.load(open('classes.pkl', 'rb'))
# model = load_model('chatbotmodel.h5')


# model = load_model('chatbot_model.model')


def clean_up_sentence(sentence):
    ifl = InitialFileLoader()
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [ifl.getLemmatizer().stem(word) for word in sentence_words]
    # sentence_words = [ifl.getLemmatizer().lemmatize(word) for word in sentence_words]
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
def generateChatModel():
    demo = ChatBotModel()
    return demo.generatechatmodel()


if __name__ == '__main__':
    app.run(debug=True)

print("GO! bot is running!")
