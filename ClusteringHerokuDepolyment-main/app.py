import pandas as pd
from flask import Flask, request, jsonify
import pickle
from scipy.spatial import KDTree

app = Flask(__name__)

#import model
#kdtree = pickle.load(open('cluster.pkl','rb'))

# Import dataset
df = pd.read_csv('Health_Dataset.csv')
workwith = df[['X_gps_longitude','X_gps_latitude']]

kdtree = KDTree(workwith)

@app.route('/', methods=['GET'])
def hello_word():
    return "Welcome to my clustering api"

@app.route('/predict', methods=['POST'])
def predict():
    json_ = request.json
    query_df = pd.DataFrame(json_, index=[0])
    d, i = kdtree.query(query_df, k = 10)
    for x in i:
        nearest_points = workwith.iloc[x]
    nearby = df.iloc[nearest_points.index][['facility_name','facility_type','community','X_gps_longitude','X_gps_latitude']]
    nearby.reset_index(drop=True, inplace=True)
    json_data = nearby.to_json()
    #json_data = {"Prediction": "Thank God"}
    return jsonify(json_data)

if __name__ == '__main__':
    app.run(debug = True)
