# -*- coding: utf-8 -*-
"""
@CREATETIME: 2018/10/23 11:47 
@AUTHOR: Chans
@VERSION: 1.0
"""
import logging
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBClassifier

from heamy.dataset import Dataset
from heamy.estimator import Regressor, Classifier
from heamy.pipeline import ModelsPipeline

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

np.random.seed(1000)

data = load_boston()
X, y = data['data'], data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=111)

# Create datasets
dataset = Dataset(X_train, y_train, X_test)


# Here we feed estimators and args of estimators
model_rf = Regressor(dataset=dataset, estimator=RandomForestRegressor,
                     parameters={'n_estimators': 50}, name='rf')
model_lr = Regressor(dataset=dataset, estimator=LinearRegression,
                     parameters={'normalize': True}, name='lr')
model_knn = Regressor(dataset=dataset, estimator=KNeighborsRegressor,
                      parameters={'n_neighbors': 15}, name='knn')
model_lgt = Regressor(dataset=dataset, estimator=LogisticRegression,
                      parameters={'penalty':'l2'}, name='lgt')
xgbclf = Classifier(dataset=dataset, estimator=XGBClassifier)

# Stack two models
# Returns new dataset with out-of-fold predictions
pipeline = ModelsPipeline(model_rf, model_lr, model_knn, xgbclf)
weights = pipeline.find_weights(mean_absolute_error)
result = pipeline.weight(weights)
stack_ds = pipeline.stack(k=10, seed=111)


# Then, train LinearRegression on stacked data
stacker = Regressor(dataset=dataset, estimator=LinearRegression)
results = stacker.predict()

results = stacker.validate(k=10,scorer=mean_absolute_error)

