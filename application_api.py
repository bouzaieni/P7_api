import flask
import pandas as pd
import json
import numpy as np
import pickle

model = pickle.load(open('lgb.pkl', 'rb'))

path = 'https://raw.githubusercontent.com/bouzaieni/p7/main/data_api.csv'
data_api = pd.read_csv(path, index_col=0)
data_api = data_api.loc[:, ~data_api.columns.str.match('TARGET')]


threshold = 0.57

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/', methods=['GET'])
def home():
    return "<h1><CENTER>Home Credit Default Risk</CENTER></h1>"
# results of predictions
@app.route('/credit_client', methods=['GET'])
def predict():
    
    if type(flask.request.args.get('index')) is None:
        client_index = '1'
    else:
        client_index = flask.request.args.get('index')
    
    client = data_api[data_api.index == int(client_index)]
    data = client.to_json()
   
    score_pred = model.predict_proba(client)[:, 1]
    data_client = data_api.copy()
   
    df_proba_pred = pd.DataFrame(model.predict_proba(data_client)[:, 1], columns=['prob_pred'],
                             index=data_client.index.tolist())
    
    df_proba_pred['Predict'] = np.where(df_proba_pred['prob_pred'] < threshold, 0, 1)
    data_client['Proba_Score'] = df_proba_pred['prob_pred']
    data_client['Predict'] = df_proba_pred['Predict']
    #  JSON format!
    data_json = data_client.to_json()
    resultat_prediction = {'credit_score_pred': score_pred[0], "json_data": data, 'Total_score': data_json}
  
    class NumpyFloatValuesEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.float32):
                return float(obj)
            return json.JSONEncoder.default(self, obj)

   
    resultat_prediction = json.dumps(resultat_prediction, cls=NumpyFloatValuesEncoder)
  
    data_client.drop(['Proba_Score', 'Predict'], axis=1)
    return resultat_prediction
    # define endpoint for Flask
app.add_url_rule('/credit_client', 'credit_client', predict)
if __name__ == '__main__':
    app.run(host='localhost', port=5002, debug=True)
