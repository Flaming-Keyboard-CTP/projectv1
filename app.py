import flask
import os
import pickle
import pandas as pd
import skimage
from skimage import io, transform
import cv2
import numpy as np
import tensorflow
import cv2
from cv2 import cv2
from skimage import feature, color, data
import skimage
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt

import werkzeug.utils

app = flask.Flask(__name__, template_folder='templates')

path_to_image_classifier = 'models/imgage-classifier.pickle'

CATEGORIES=['Buildings','Forest', 'Glacier','Mountain','Sea','Street']


with open(path_to_image_classifier, 'rb') as f:
    image_classifier = pickle.load(f)



# For Uploading Images
UPLOAD_FOLDER = 'static/uploads/'
app.secret_key = "secret key" 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# Create Upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/classify_image/', methods=['GET', 'POST'])
def classify_image():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('classify_image.html'))

    
    if flask.request.method == 'POST':

        if 'file' not in flask.request.files:
            return flask.redirect(flask.request.url)

        file = flask.request.files['file']

        if file.filename == '':
            return flask.redirect(flask.request.url)

        if file and allowed_file(file.filename):
            # Save the image to the backend static folder
            filename = werkzeug.utils.secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            
            # Read image file
            img = cv2.imread(path)

            cv2.imwrite(path, cv2.resize(img, (300, 300)))
            img = cv2.resize(img, (150,150))
            

            def hog_data_extractor(image):
                hog_data = feature.hog(image)/255.0
                return hog_data

            
            prd_image_data = hog_data_extractor(img)


            # Get prediction of image from classifier
            prediction = image_classifier.predict(prd_image_data.reshape(1, -1))[0]

            # Get the value at index of CATEGORIES
            prediction = CATEGORIES[prediction]
            
            return flask.render_template('classify_image.html', prediction=str(prediction), filename=filename)
        else:
            return flask.redirect(flask.request.url)


# @app.route('/', methods=['GET', 'POST'])
# def home_page():
#     if flask.request.method == 'POST':
#         return(flask.render_template('index.html'))

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main.html'))
    


@app.route('/about/')
def about():
    return flask.render_template('about.html')


@app.route('/contributors/')
def contributors():
    return flask.render_template('contributors.html')





if __name__ == '__main__':
    app.run(debug=True)