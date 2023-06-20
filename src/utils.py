import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, accuracy_score
from sklearn.tree import export_graphviz
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from logger import logging

from exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            logging.info("Dumping pickle file")
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}
        logging.info("Entering function for evaluating models")
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)
            logging.info("Calculating predicted values of train and test data")

            train_model_score = accuracy_score(y_train, y_train_pred)

            test_model_score = accuracy_score(y_test, y_test_pred)
            logging.info("Calculating performance metrics")

            report[list(models.keys())[i]] = test_model_score
            logging.info("Returning list of performance metrics")

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            logging.info("Loading pickle file")
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)