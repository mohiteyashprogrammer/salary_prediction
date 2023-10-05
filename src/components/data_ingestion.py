import os 
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import read_sql_data
from sklearn.model_selection import train_test_split


# create configure path from here
@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join("artifcats","train.csv")
    test_data_path:str = os.path.join("artifcats","test.csv")
    raw_data_path:str = os.path.join("artifcats","raw.csv")


class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()


    def start_data_ingestion(self):

        '''
        This Method Will Fatch Data From SQL Data Base
        And Import In Pipline

        '''
        logging.info("DataIngestion Started")
        try:
            logging.info("Reading Data From mysql DataBase")

            data = read_sql_data()

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,index=False)

            logging.info("Apply Train Test Split")
            train_set,test_set = train_test_split(data,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data Ingestion Complited")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as e:
            logging.info("Error Occured In Data Ingestion Stage")
            raise CustomException(e,sys)



#run

if __name__=="__main__":
    ingestion = DataIngestion()
    train_data_path,test_data_path = ingestion.start_data_ingestion()

        
