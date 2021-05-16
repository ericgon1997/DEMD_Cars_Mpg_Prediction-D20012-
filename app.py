from flask import Flask, request,jsonify
import numpy as np 
import pandas as pd
import pickle
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)

#Loading model using pickle file 
pickle_file_open = open("car_mpg.pkl","rb")
regressor = pickle.load(pickle_file_open)

@app.route('/')
def welcome_message():
    try:
        return "Welcome to praxis",200
    except Exception as e:
        return "something went wrong",400

@app.route('/predict',methods=['Get'])
def predict():

    """Testing of prediction .
    ---
    parameters:  
      - name: cyl
        in: query
        type: number
        required: true
      - name: displ
        in: query
        type: number
        required: true
      - name: weight
        in: query
        type: number
        required: true
      - name: hp
        in: query
        type: number
        required: true        
      - name: accel
        in: query
        type: number
        required: true
      - name: ModelYear
        in: query
        type: number
        required: true
      - name: origin
        in: query
        type: number
        required: true        
    responses:
        200:
            description: The response is 
        
    """



    cyl = request.args.get("cyl")
    disp = request.args.get("displ")
    weight = request.args.get("weight")
    hp = request.args.get("hp")
    accel = request.args.get("accel")
    ModelYear = request.args.get("yr")
    Orig = request.args.get("origin")
    result = regressor.predict([[cyl,disp,weight,hp,accel,ModelYear,Orig]])
    return "The mileage is " + str(result)

if __name__ == "__main__":
    app.run(debug=True)


