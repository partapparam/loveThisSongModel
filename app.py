import pickle

from flask import Flask
from flask import request
from flask import jsonify
import xgboost as xgb
import pandas as pd


model_file = "xgb_model.bin"

with open(model_file, "rb") as f_in:
    dv, xgb_model = pickle.load(f_in)

app = Flask(__name__)

# function to predict using the model and DictVectorizer
def prediction(data, dv, model):
    array = [data]
    df = pd.DataFrame(array)
    del df['song_title']

    y_test = df['target']
    del df['target']
    features = list(dv.get_feature_names_out())

    df_dict = df.to_dict(orient='records')

    X_test = dv.transform(df_dict)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)

    xgb_pred = model.predict(dtest)
    xgb_liked = (xgb_pred >= 0.5)
    result = {"Like probability": float(xgb_pred), "Liked": bool(xgb_liked)}

    return result


@app.route("/predict", methods=["POST"])
def predict():
    song_attributes = request.get_json()
    # features = list(dv.get_feature_names_out())
    # X_test = dv.transform([song_attributes])
    # dtest = xgb.DMatrix(X_test, feature_names=features)
    # print(song_attributes)
    result = prediction(song_attributes, dv, xgb_model)
    # xgb_liked = (xgb_pred >= 0.5)

    # result = {"Liked": float(xgb_pred), "Liked": bool()}
    print('This is the result', result)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)