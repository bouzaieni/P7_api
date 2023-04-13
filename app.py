import flask
import pandas as pd
import json
import numpy as np
import pickle

# Load  model
model = pickle.load(open('lgb.pkl', 'rb'))
# Load data
path = 'https://raw.githubusercontent.com/bouzaieni/p7/main/data_api.csv'
data_api = pd.read_csv(path, index_col=0)
#data_api = data_api.sample(1200, random_state=42)
data_api = data_api.loc[:, ~data_api.columns.str.match('TARGET')]

# threshold 
threshold = 0.5
# defining flask pages
app = flask.Flask(__name__)
app.config["DEBUG"] = True
# home page
@app.route('/', methods=['GET'])
def home():
    return "<h1><CENTER>Home Credit Default Risk</CENTER></h1>"
# results of predictions
@app.route('/credit_client', methods=['GET'])
def predict():
    # get the index from a request, defined a client_index parameter as default
    if type(flask.request.args.get('index')) is None:
        client_index = '731'
    else:
        client_index = flask.request.args.get('index')
    # get inputs features from the data with index
    client = data_api[data_api.index == int(client_index)]
    data = client.to_json()
    # predict_proba returns a list as [0,1], 0 -> for payments accepted, 1 -> for payments refused
    # we have chosen second parameter for refused value
    score_pred = model.predict_proba(client)[:, 1]
    data_client = data_api.copy()
    # for add probabilities, used normalized dataset
    df_proba_pred = pd.DataFrame(model.predict_proba(data_client)[:, 1], columns=['prob_pred'],
                             index=data_client.index.tolist())
    # for add prediction, used threshold value
    df_proba_pred['Predict'] = np.where(df_proba_pred['prob_pred'] < threshold, 0, 1)
    data_client['Proba_Score'] = df_proba_pred['prob_pred']
    data_client['Predict'] = df_proba_pred['Predict']
    #  JSON format!
    data_json = data_client.to_json()
    resultat_prediction = {'credit_score_pred': score_pred[0], "json_data": data, 'Total_score': data_json}
    # for json format some values are categorical, however it is difficult to handle these values as float,
    # these values types are changed by using JSON encoder
    class NumpyFloatValuesEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.float32):
                return float(obj)
            return json.JSONEncoder.default(self, obj)

    # JSON format dumps method for send the data to Dashboard
    resultat_prediction = json.dumps(resultat_prediction, cls=NumpyFloatValuesEncoder)
    # Each request of dashboard, data_client dataframe adding ['Proba_Score', 'Predict'] columns,
    # so It needs to drop these columns at the end of the API
    data_client.drop(['Proba_Score', 'Predict'], axis=1)
    return resultat_prediction
    # define endpoint for Flask
app.add_url_rule('/credit_client', 'credit_client', predict)
if __name__ == '__main__':
    app.run(host='localhost', port=5002, debug=True)
