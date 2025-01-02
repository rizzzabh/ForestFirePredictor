import pickle
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from flask import Flask,request,jsonify,render_template


application = Flask(__name__)
app = application


lasso_model = pickle.load(open('models/forestLassoModel.pkl','rb'))
standard_scalar = pickle.load(open('models/scaler.pkl','rb'))

@app.route("/")
def index (): 
      return render_template('index.html')

@app.route("/predictdata", methods=['GET', 'POST'])
def predict_data():
    if request.method == "POST":
      Temperature = float(request.form['temperature'])
      RH = float(request.form['rh'])
      Ws = float(request.form['ws'])
      Rain = float(request.form['rain'])
      FFMC = float(request.form['ffmc'])
      DMC = float(request.form['dmc'])
      ISI = float(request.form['isi'])
      Classes = request.form['classes']  
      Region = request.form['region']    

      new_data=standard_scalar.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
      result = lasso_model.predict(new_data)
      return render_template('home.html',results=result[0])
    else:
        return render_template('home.html')

if __name__ =="__main__":
      app.run(debug=True,port= 8000)
