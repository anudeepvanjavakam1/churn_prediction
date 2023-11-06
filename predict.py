
import joblib
from flask import Flask
from flask import request

# load model with joblib
model = joblib.load("churn_xgb_model.sav")
dv = joblib.load("dv.sav")

app = Flask('get-churn-prob')

@app.route('/predict', methods=['POST'])
def predict():

    customer = request.get_json()
    
    X = dv.transform([customer])

    y_pred = model.predict_proba(X)[0,1]
    
    result = {'Churn probability': float(y_pred)}

    return result

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)