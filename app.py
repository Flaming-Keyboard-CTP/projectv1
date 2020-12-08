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


# try:
#     image_classifier = tensorflow.keras.models.load_model(path_to_image_classifier)
# except EOFError as e:
#     print(e)



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

        # def scene_predict(img_path):
        #     image = cv2.imread(img_path)
        #     ip_image = Image.open(img_path)
        #     image = cv2.resize(image,(150,150))
        #     prd_image_data = hog_data_extractor(img_path)
        #     scene_predicted = image_classifier.predict(prd_image_data.reshape(1, -1))[0]
        #     fig, ax = plt.subplots(1, 2, figsize=(12, 6),
        #                     subplot_kw=dict(xticks=[], yticks=[]))
        #     ax[0].imshow(ip_image)
        #     ax[0].set_title('input image')

        #     ax[1].imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        #     ax[1].set_title('Scene predicted :'+ CATEGORIES[scene_predicted]);

        # ip_img_folder = '../input/seg_pred/seg_pred/'
        # ip_img_files = ['222.jpg','121.jpg','88.jpg','398.jpg','839.jpg', '520.jpg']
        # scene_predicted = [scene_predict(os.path.join(ip_img_folder,img_file))for img_file in ip_img_files]




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
            img = cv2.resize(img, (150,150))
            

            def hog_data_extractor(image):
                hog_data = feature.hog(image)/255.0
                return hog_data

            
            prd_image_data = hog_data_extractor(img)
            
            # Reshape the image into shape (1, 64, 64, 3)
            #img = np.asarray([img])

            # Get prediction of image from classifier
            prediction = image_classifier.predict(prd_image_data.reshape(1, -1))[0]

            # Get the value at index of CATEGORIES
            prediction = CATEGORIES[prediction]
            
            return flask.render_template('classify_image.html', prediction=str(prediction))
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