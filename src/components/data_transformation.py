import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_preprocessor_object_file(self):
        try:
            logging.info('Initiatig Data Transformation')


            num_cols = ['gender','age','annual_salary','credit_card_debt','net_worth','car_purchase_amount']

            num_pipeline = Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
            ])

            preprocessor = ColumnTransformer([
                ('num_pipeline',num_pipeline,num_cols)
            ])

            logging.info('Pipeline Completed')

            return preprocessor

        except Exception as e:
            logging.info('Exception occured during get_preprocessor_object_file')
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Reading train and test data')
            logging.info(f'Train data:\n{train_df.head(2).to_string()}')
            logging.info(f'Test data:\n{test_df.head(2).to_string()}')

            logging.info('Getting preprocessing object')

            preprocessor = self.get_preprocessor_object_file()

            target_column_name = 'car_purchase_amount'
            drop_columns = ['customer_name','customer_mail','country']

            train_df = train_df.drop(drop_columns, axis=1)
            test_df = test_df.drop(drop_columns, axis=1)

            train_arr = preprocessor.fit_transform(train_df)
            test_arr = preprocessor.fit_transform(test_df)

            logging.info("Applying preprocessing on dataset")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                object= preprocessor
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

           
        except Exception as e:
            logging.info('Exception occured during initiate_data_transformation')
            raise CustomException(e,sys)