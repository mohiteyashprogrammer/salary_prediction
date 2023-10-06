import os 
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from src.utils import model_traning,save_object,lode_object
import warnings
warnings.filterwarnings("ignore")

@dataclass
class ModelTraningConfig:
    traning_model_file_obj = os.path.join("artifcats","model.pkl")


class ModelTraning:

    def __init__(self):
        self.model_traning_config = ModelTraningConfig()


    def start_model_traning(self,train_array,test_array):

        '''
        This Function Will Take Hyperamaters and Tranin Model 
        Give you Best Evaluated Model

        '''

        try:
            logging.info("Model Traning Started")

            logging.info("Split Dependent And Indipendent Features")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "LinearRegression":LinearRegression(),
                "Ridge":Ridge(),
                "Lasso":Lasso(),
                "Elastic_Net":ElasticNet(),
                "KNeighborsRegressor":KNeighborsRegressor(),
                "RandomForestRegressor":RandomForestRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor(),
                "XGBRegressor":XGBRegressor(),
                "CatBoostRegressor":CatBoostRegressor(verbose=False)
            }

            params = {
                "LinearRegression":{

                },
                "Ridge":{
                    "alpha": [0.01, 0.1, 1, 10,20]
                },
                "Lasso":{
                    "alpha": [0.01, 0.1, 1, 10,20]
                },
                "Elastic_Net":{
                    "alpha": [0.01, 0.1, 10],
                    "l1_ratio": [0.4, 0.6, 0.8]
                },
                "KNeighborsRegressor":{
                    "n_neighbors":[5,8,10,15],
                    "weights":["uniform", "distance"],
                    "algorithm":["auto","ball_tree","kd_tree"]
                },
                "RandomForestRegressor":{
                    "criterion":["squared_error", "friedman_mse"],
                    'n_estimators': [ 180, 200,300],
                    'max_depth': [10,15,23],
                    'min_samples_split': [5,6,8],
                    'min_samples_leaf': [3,5,6],
                },
                "GradientBoostingRegressor":{
                    "learning_rate":[0.1,0.01,0.001,1],
                    "n_estimators": [ 180, 200,300],
                    'max_depth': [10,15,23],
                },
                "XGBRegressor":{
                    "learning_rate":[0.1,0.01,0.001,1],
                    "n_estimators": [ 180, 200,300],
                    'max_depth': [10,15,23],
                },
                "CatBoostRegressor":{
                    'iterations': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'depth': [4, 6, 8],
                }
            }

            model_report:dict  = model_traning(X_train,y_train,X_test,y_test,models,params)

            # To Get Best Model
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f"Best Model Found, Name Is : {best_model_name},Accuracy_score: {best_model_score}")
            print("*"*100)
            logging.info(f"Best Model Found, Name Is : {best_model_name},Accuracy_score: {best_model_score}")


            save_object(filepath=self.model_traning_config.traning_model_file_obj,
                        obj= best_model
                        )
            
        except Exception as e:
            logging.info("Error Occured In Model Trainig Stage")
            raise CustomException(e,sys)
