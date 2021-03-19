import pickle
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

from flask import json


class InitialFileLoader:

    def getLemmatizer(self):
        porter = PorterStemmer()
        # lemmatizer = WordNetLemmatizer()
        return porter

    def getIntents(self):
        intents = json.loads(open('intentCombined.json').read())
        return intents

    def getWords(self):
        words = pickle.load(open('words.pkl', 'rb'))
        return words

    def getClasses(self):
        classes = pickle.load(open('classes.pkl', 'rb'))
        return classes

    def getModel(self):
        model = load_model('chatbotmodel.h5')
        return model
