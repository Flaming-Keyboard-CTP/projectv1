import Flask
import os
import pickle
import pandas as pd 
import skimage


app = Flask(__name__)


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/predict")
def predict_route():
    sentence = request.args.get("sentence")
    output = predict(sentence)


    if output == 1:
        output = "Positive"
        prob = g.model.predict_proba(g.X).round(3)[0][1]

    else:
        output = "Negative"
        prob = g.model.predict_proba(g.X).round(3)[0][0]


    return render_template('index.html', result=output, sentence=sentence, lemmed=g.lemmed, prob=prob, showPrediction=True)

if __name__ == '__main__':
	app.run( port="3000", debug=True)