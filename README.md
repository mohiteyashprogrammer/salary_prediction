
# Salary Prediction Application
https://github.com/mohiteyashprogrammer/salary_prediction/assets/114924851/28070c26-34ab-4be4-b5b2-64b0be7a5c17


# DVC
'''
dvc init

dvc add artifcats/raw.csv
'''

MLFLOW_TRACKING_URI=https://dagshub.com/mohiteyashprogrammer/salary_prediction.mlflow \
MLFLOW_TRACKING_USERNAME=mohiteyashprogrammer \
MLFLOW_TRACKING_PASSWORD=e9dc540f21bfcc5064e219ee5772fd4f3b9a7a26 \
python script.py

# create Environment variable through this commands

export MLFLOW_TRACKING_URI=https://dagshub.com/mohiteyashprogrammer/salary_prediction.mlflow

export MLFLOW_TRACKING_USERNAME=mohiteyashprogrammer

export MLFLOW_TRACKING_PASSWORD=e9dc540f21bfcc5064e219ee5772fd4f3b9a7a26
