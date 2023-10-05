import os 
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from dotenv import load_dotenv
import pymysql

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

