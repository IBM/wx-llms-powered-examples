#
# Copyright IBM Corp. 2024-2025
# SPDX-License-Identifier: Apache-2.0
#
# Author: Nguyen, Hung (Howie) Sy
#

from flask import Flask, request, jsonify
from watsonx_client import WatsonxClient
from threading import Thread
import os
from dotenv import load_dotenv
import json

load_dotenv()

embedding_model = None

app = Flask(__name__)

# it's to help keep order of sorted dictionary passed to jsonify() function
app.json.sort_keys = False

@app.route('/meta', methods=['GET'])
def hello_world():
    return jsonify(meta="This API is to query an embedding model from watsonx")

@app.route('/.well-known/live', methods=['GET'])
def live():
    # return a tuple with HTTP status code 204
    return '', 204

@app.route('/.well-known/ready', methods=['GET'])
def ready():
    if embedding_model:
        return '', 204
    else:
        return jsonify(status="not ready"), 503

@app.route('/vectors', methods=['POST'])
def vectors():
    try:
        if not embedding_model:
            raise Exception("Sorry, the embedding model is not ready")

        data = request.get_data().decode('utf-8')
        try:
            json_data = json.loads(data)
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid JSON"}), 400

        text = json_data['text']

        embedding = embedding_model.embed_query(text)

        dim = -1
        if embedding:
           dim = len(embedding)
            
        response = {
            "text": text,
            "vector": embedding,
            "dim": dim
        }

        return jsonify(response)
    
    except Exception as error:
        print(f"Error: {error}")
        return f"Error: {error}", 500

if __name__ == '__main__':
    def embedding_model_connection_thread():
        print("Connecting the embedding model from WatsonX...")
        global embedding_model
        embedding_model = WatsonxClient.request_embedding_model()

        if embedding_model:
            print("\nConnected the embedding model...")
        else:
            print("\nFailed to connect the embedding model...")

    wx_thread = Thread(target=embedding_model_connection_thread)
    wx_thread.start()

    print("\nStart the server...")

    # 0.0.0.0 is recommended when running on WSL and/or container enviroments for development purposes
    app.run(host="0.0.0.0", port=os.getenv("API_PORT", 5000), debug=False) 