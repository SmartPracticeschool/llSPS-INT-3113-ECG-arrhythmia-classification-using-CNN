# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 18:36:52 2020

@author: saira
"""



from __future__ import division, print_function
from flask import Flask,request, render_template
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from gevent.pywsgi import WSGIServer
import os
global model,graph
app = Flask(__name__)





def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict_classes(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    return render_template('ecg.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        model = load_model('ECG1.h5')

        preds = model_predict(file_path, model)
        ls=['Fusion Beat','Normal Beat','Unknown Beat','Supraventricular ectopic Beat','Ventricular ectopic beat']
        result = ls[preds[0]]             
        return result
    return None


if __name__ == '__main__':
    
      #port = int(os.getenv('PORT', 8000))
     #app.run(host='0.0.0.0', port=port, debug=True)
      #http_server = WSGIServer(('0.0.0.0', port), app)
      #http_server.serve_forever()
    app.run(debug=True)