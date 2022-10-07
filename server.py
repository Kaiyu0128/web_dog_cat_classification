from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import numpy as np
import cv2
from datetime import datetime
import string
from PIL import Image
import keras
import sys, os
from keras.models import load_model

imsize = (64, 64)
keras_param = "./cnn.h5"

app = Flask(__name__)
'''
def load_image(path):
    print(" * Converting your photo ...")
    img = Image.open(path)
    img = img.convert('RGB')
    img = img.resize(imsize)
    img = np.asarray(img)
    img = img / 255.0
    return img
'''
def start_model():
    print('initiating model')

@app.route('/')
def index():
    return render_template('./flask_api_index.html')

@app.route('/result', methods=['POST'])
def result():
    if request.files['image']:
        img = Image.open(request.files['image'])
        img = img.convert('RGB')
        img = img.resize(imsize)
        img = np.asarray(img)
        img = img / 255.0
        print("loading pre trained model")
        model = load_model(keras_param)
        prd = model.predict(np.array([img]))

        prelabel = np.argmax(prd, axis=1)
        if prelabel == 0:
            result = 'dog'
        elif prelabel == 1:
            result = 'cat'
        test_model_percentage = float(prd)
        return render_template('._result.html', title = 'test result', result = result, test_model_percentage = test_model_percentage)

if __name__ == '__main__':
    start_model()
    app.debug = True
    app.run(host='localhost', port=3000)
