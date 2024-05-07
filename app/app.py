import numpy as np
from flask import Flask, render_template, request
import pickle

#Create flask app
app = Flask(__name__)

model = pickle.load(open('D:\Iris Flower Prediction On Web Application\model\model.pkl', 'rb'))

@app.route("/")
def Home():
    return  render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)

    return render_template("index.html",prediction_text = "THE FLOWER SPECIE IS {}".format(prediction))

if __name__ == "__main__":
    app.run(debug=True)