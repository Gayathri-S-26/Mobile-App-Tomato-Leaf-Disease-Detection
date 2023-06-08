import os
import sys
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from util import base64_to_pil

app = Flask(__name__)

print('Model loaded. Check http://127.0.0.1:5000/')
MODEL_PATH = 'model_tomato_inception.h5'
model = load_model(MODEL_PATH)


def model_predict(img, model):
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x=x/255
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds=("The Disease is Bacterial_Spot.The Pathogen is Bacteria.")
    elif preds==1:
        preds=("The Disease is Early_Blight.The Pathogen is Fungi.")
    elif preds==2:
        preds=("The Disease is Late_Blight.The Pathogen is Mold.")
    elif preds==3:
        preds=("The Disease is Leaf_Mold.The Pathogen is Fungi.")
    elif preds==4:
        preds=("The Disease is Septoria_Leaf_Spot.The Pathogen is Fungi.")
    elif preds==5:
        preds=("The Disease is Two-Spotted_Spider_Mite.The Pathogen is Mite.")
    elif preds==6:
        preds=("The Disease is Target_Spot.The Pathogen is Fungi.")
    elif preds==7:
        preds=("The Disease is Yellow_Leaf_Curl_Virus.The Pathogen is Viral.")
    elif preds==8:
        preds=("The Disease is Mosaic_Virus.The Pathogen is Viral.")
    elif preds==9:
        preds=("The Leaf is Healthy.")

    return preds.replace(".",".\n")


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        
        img = base64_to_pil(request.json)
        preds = model_predict(img, model)
        result=preds
        return jsonify(result=result)

    return None


if __name__ == '__main__':
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
