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

import skimage
import skimage.io
import skimage.transform

import werkzeug.utils

app = flask.Flask(__name__, template_folder='templates')

path_to_image_classifier = 'models/cnn_new'

CATEGORIES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']


try:
    image_classifier = tensorflow.keras.models.load_model(path_to_image_classifier)
except EOFError as e:
    print(e)



# with open(path_to_image_classifier, 'rb') as f:
#     image_classifier = pickle.load(f)



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

        # print(dir(flask.request))
        # Get file object from user input.
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
            # Read image file string data
            # filestr = file.read()
            
            # Convert string data to np arr
            # npimg = np.frombuffer(filestr, np.uint8)
            # Convert np arr to image
            # img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

            # Resize the image to match the input the model will accept
            img = cv2.resize(img, (50,50))
            # Reshape the image into shape (1, 64, 64, 3)
            img = np.asarray([img])

            # Get prediction of image from classifier
            prediction = np.argmax(image_classifier.predict(img), axis=-1)

            # Get the value at index of CATEGORIES
            prediction = CATEGORIES[prediction[0]]
            
            return flask.render_template('main.html', 
                prediction=prediction,
                filename=filename)
        else:
            return flask.redirect(flask.request.url)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # if flask.request.method == 'POST':
    #     # Get file object from user input.
    #     file = flask.request.files['file']

    #     if file:
    #         # Read the image using skimage
    #         img = skimage.io.imread(file)
    #         new_image = cv2.resize(img, (50,50))
    #         t = np.array(new_image).reshape(-1, 50, 50, 1)
    #         predictions = np.argmax(image_classifier.predict(t), axis = 1)
    #         #curr = np.argmax(model.predict(t), axis = 1)

    #         prediction = predictions[0]
    #         return flask.render_template('classify_image.html', prediction=str(prediction))

    # return(flask.render_template('classify_image.html'))

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main.html'))


if __name__ == '__main__':
    app.run(debug=True)