import os 
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline




@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifcats","preprocessor.pkl")


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def get_preprocessor_object(self):

        '''
        This Function Will give The Preprocessor Object

        '''

        try:
            logging.info("Data Transformation Started")

            catigorical_features = ['Gender', 'Education Level', 'Job Title']

            numerical_features = ['Age', 'Years of Experience']

            #numeric pipline
            num_pipline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
                ]
            )

            # cato_pipline
            cato_pipline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("onehote",OneHotEncoder(sparse=False,handle_unknown="ignore",drop="first")),
                ("scaler",StandardScaler(with_mean=False))
                ]
            )

            # create preprocessor object
            preprocessor = ColumnTransformer([
                ("num_pipline",num_pipline,numerical_features),
                ("cato_pipline",cato_pipline,catigorical_features)
            ])
            
            return preprocessor
        
            logging.info("Pupline Complited")
        

        except Exception as e:
            logging.info("Error Occured In Data Transformation")
            raise(CustomException(e,sys))
        


    def initated_data_transformation(self,train_path,test_path):

        '''
        This Method Will Take Preprocessor Object And Transform The Data

        '''
        try:

            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Reading Training and Testing Data Completed")
            logging.info(f'Train Dataframe Head : \n{train_data.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_data.head().to_string()}')

            # Add your data preprocessing steps here
            # Remove duplicates
            train_data.drop_duplicates(inplace=True)
            test_data.drop_duplicates(inplace=True)

            # Map values in the "Gender" column
            train_data["Gender"] = train_data["Gender"].map({"Male": "Male", "Female": "Female", "Other": "Male"})
            test_data["Gender"] = test_data["Gender"].map({"Male": "Male", "Female": "Female", "Other": "Male"})

            # Map values in the "Education Level" column
            train_data["Education Level"] = train_data["Education Level"].map({
            "Bachelor's Degree": "Bachelor's Degree",
            "Master's Degree": "Master's Degree",
            "PhD": "PhD",
            "Bachelor's": "Bachelor's Degree",
            "High School": "High School",
            "Master's": "Master's Degree",
            "phD": "PhD"
            })

            test_data["Education Level"] = test_data["Education Level"].map({
            "Bachelor's Degree": "Bachelor's Degree",
            "Master's Degree": "Master's Degree",
            "PhD": "PhD",
            "Bachelor's": "Bachelor's Degree",
            "High School": "High School",
            "Master's": "Master's Degree",
            "phD": "PhD"
            })
            
            # Drop columns "Race" and "Country"
            train_data.drop(["MyUnknownColumn","Race", "Country"], axis=1, inplace=True)
            test_data.drop(["MyUnknownColumn","Race", "Country"], axis=1, inplace=True)



            # Remove jobs with value counts less than 10
            job_title_counts = train_data['Job Title'].value_counts()
            train_data = train_data[train_data['Job Title'].map(job_title_counts) > 10]

            job_title_counts = test_data['Job Title'].value_counts()
            test_data = test_data[test_data['Job Title'].map(job_title_counts) > 10]

            logging.info("Obtaining Preprocessor Object")

            preprocessor_obj = self.get_preprocessor_object()

            target_column_name = "Salary"
            drop_columns = [target_column_name]

            # split Data In To Independent And Dependent Features
            input_feature_train_data = train_data.drop(drop_columns,axis=1)
            target_feature_train_data = train_data[target_column_name]

            # split Data In To Independent And Dependent Features
            input_feature_test_data = test_data.drop(drop_columns,axis=1)
            target_feature_test_data = test_data[target_column_name]

            # Apply Preprocessor Object
            input_features_train_arr = preprocessor_obj.fit_transform(input_feature_train_data)
            input_features_test_arr = preprocessor_obj.transform(input_feature_test_data)

            logging.info("Applyed Preprocessor Object On Train and Test Data")

            # Convert In TheArray To Fast Process
            train_array = np.c_[input_features_train_arr,np.array(target_feature_train_data)]
            test_array = np.c_[input_features_test_arr,np.array(target_feature_test_data)]

            print("Train Array Shape: ",train_array.shape)
            print("Test Array Shape: ",test_array.shape)

            # Save The Preprocessor in Pickel File
            save_object(filepath=self.data_transformation_config.preprocessor_obj_file_path,
                        obj=preprocessor_obj)
            
            logging.info("Preprocessor Object File Saved")

            return (
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info("Error Occured In Data Transformation Stage")
            raise(CustomException(e,sys))

