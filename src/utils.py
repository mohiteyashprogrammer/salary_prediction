import os 
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from dotenv import load_dotenv
import pymysql
import pickle
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

load_dotenv()
# define variable
host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")


def read_sql_data():
    '''
    This Function Will Read Data From Data Base

    '''
    logging.info("reading SQL Data Base started")
    try:
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db = db
        )
        logging.info("Connection Establish",mydb)
        df = pd.read_sql_query("select * from payment",mydb)
        print(df.head())

        return df
        
    except Exception as e:
        raise CustomException(e,sys)
    

def save_object(filepath,obj):
    '''
    This Function Will save Pickel file

    '''
    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path,exist_ok=True)

        with open(filepath,"wb") as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise(CustomException(e,sys))
    

def model_traning(X_train,y_train,X_test,y_test,models,param):

    '''
    This Function Will Train The model and Evaluate

    '''
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            # model traning
            grid = GridSearchCV(model,para,cv=3,n_jobs=-1)
            grid.fit(X_train,y_train)

            model.set_params(**grid.best_params_)
            model.fit(X_train,y_train)

            #make prediction
            y_test_pred = model.predict(X_test)

            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e,sys)
    


def lode_object(file_path):
    '''
    This  Function Will Loab Pickel File And 
    Read In binery Mode
    
    '''
    try:
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

