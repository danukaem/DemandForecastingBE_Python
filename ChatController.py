# from flask import Flask, jsonify, json, request
# from flask_cors import CORS
#
# from chatBotModel import ChatBotModel
#
# app = Flask(__name__)
#
# CORS(app)
#
# @app.route('/chat', methods=['GET'])
# def get_chat_response():
#     query_parameters = request.args
#     message = query_parameters.get('message')
#     print(message)
#
#     ints = predict_class(message)
#     res = get_response(ints, intents)
#     print(res)
#
#     return jsonify({"userMessage": res})
#
#
# @app.route('/generateChatModel', methods=['GET'])
# def generateChatModel():
#     demo = ChatBotModel()
#     return demo.generatechatmodel()
#
#
# if __name__ == '__main__':
#     app.run(debug=True)
