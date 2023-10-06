import os 
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import lode_object


class PredictPipline:

    def __init__(self):
        pass

    def predict(self,features):

        """
        This Function Will Predict The Output
        BAse On Inpute

        """
        try:
            preprocessor_path = os.path.join("artifcats","preprocessor.pkl")
            model_path = os.path.join("artifcats","model.pkl")

            preprocessor = lode_object(preprocessor_path)
            model = lode_object(model_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)

            return pred
        
        except Exception as e:
            logging.info("Error In Prediction PipLine")
            raise CustomException(e,sys)
        