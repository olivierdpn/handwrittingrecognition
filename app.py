from flask import Flask, render_template, make_response
from flask import redirect, request, jsonify
import matplotlib.image as mpimg
from skimage.transform import rescale, resize

import re
import base64
import json

import tensorflow as tf
from tensorflow import keras

import numpy as np



app = Flask(__name__)

def convertImage(imgData):
    img_str = imgData.split('base64,',1)[-1]
    with open("output.png", "wb") as fh:
        fh.write(base64.b64decode(img_str))

@app.route("/", methods=['GET'])
def accueil():
    return render_template("index.html")

@app.route('/predict/',methods=['GET','POST'])
def predict():
    imgData = request.get_data().decode("utf-8")
    print(imgData)
    convertImage(imgData)

    x = mpimg.imread('output.png')
    print(x)
    y = np.zeros((280,280))
    for i in range(len(x)):
        for j in range(len(x)):
            y[i][j] = x[i][j][3]

    y = rescale(y, 0.1)
    y = y.reshape(1, 28, 28)

    #print(y)

    result = str(np.argmax(model.predict(y))) 
    print(result)
    return result

if __name__ == "__main__":
    #load model
    model = keras.models.load_model('model.h5')
    app.run(debug=True) 