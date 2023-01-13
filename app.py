from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd 
import pickle
import json

app = Flask(__name__)
## Load the regression model
model = pickle.load(open('model.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output = model.predict(new_data)
    print(f"Predicted Output: {output[0]}")
    return jsonify(output[0])

if __name__ == "__main__":
    app.run(debug=True)