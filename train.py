#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.tree import export_text
import xgboost as xgb
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc

import pickle

# parameters selected while tuning
random_state=1
output_file = 'xgb_model.bin'
np.random.seed(1)

xgb_params_final = {
    'eta': 0.3, 
    'max_depth': 3,
    'min_child_weight': 1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

# Lets process our data
df = pd.read_csv('./data.csv')
del df['Unnamed: 0']
# Our dataset is already clean, so we can proceed to setting up the train and test data
df_train, df_test = train_test_split(df, test_size=0.2, random_state=random_state)
df_full_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
# Set up our target values
y_train = df_train['target']
y_test = df_test['target']
# Remove the target values and also remove the song_title from our data. This did not help our accuracy.
del df_test['target']
del df_test['song_title']
del df_train['target']
del df_train['song_title']

train_dicts = df_train.to_dict(orient='records')
test_dicts = df_test.to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)
X_test = dv.transform(test_dicts)
features = list(dv.get_feature_names_out())

# DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)


# function to train the model
def train(df_train, y_train, xgb_params):
    train_dicts = df_train.to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dicts)
    features = list(dv.get_feature_names_out())
    # DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)

    xgb_model = xgb.train(params=xgb_params,
                    dtrain=dtrain,
                    num_boost_round=100,
                    verbose_eval=5)
    
    return dv, xgb_model

# function to predict using the model and DictVectorizer
def predict(df, dv, model):
    if df['Unnamed: 0']:
        del df['Unnamed: 0']
    if df['song_title']:
        del df['song_title']

    y_test = df['target']
    del df['target']

    features = list(dv.get_feature_names_out())


    df_dict = df.to_dict(orient='records')

    X_test = dv.transform(df_dict)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)

    xgb_pred = model.predict(dtest)
    xgb_liked = (xgb_pred >= 0.5)

    return xgb_liked


dv, xgb_model = train(df_train, y_train, xgb_params_final)

# saving the DictVectorizer and XGB model to the same file
with open(output_file, 'wb') as f_out: 
    pickle.dump((dv, xgb_model), f_out)