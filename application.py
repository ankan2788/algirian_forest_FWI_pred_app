
from flask import Flask,request,jsonify,render_template 
import pickle 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

application = Flask(__name__)
app=application
ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standardising_model=pickle.load(open('models/scaler.pkl','rb'))
@app.route('/',methods=['GET','POST'])
def predict():
    if request.method=="POST":
        Temperature=float(request.form.get('Temperature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        DC=float(request.form.get('DC'))
        ISI=float(request.form.get('ISI'))
        BUI=float(request.form.get('BUI'))
        new_data=standardising_model.transform([[ Temperature,RH,Ws,Rain,FFMC,DMC,DC,ISI,BUI]])
        result=ridge_model.predict(new_data)
        return render_template('home.html',results=result[0])
        
        
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=1)
