import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts','raw_data.csv')

class DataIngestion():
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion")
        try:
            logging.info("Reading data from csv file")
            df = pd.read_csv(os.path.join('notebooks','car_purchasing.csv'), encoding='ISO-8859-1')

            df.rename(columns= {'customer name':'customer_name','customer e-mail':'customer_mail',
                'annual Salary':'annual_salary','credit card debt':'credit_card_debt','net worth':'net_worth',
                'car purchase amount':'car_purchase_amount'},inplace=True)

            logging.info(f'{df.head(2)}')

            logging.info('Updated column names with conventional names')

            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path,index=False)

            logging.info('Train Test Split initiated')
            train_set,test_set = train_test_split(df,test_size=0.3,random_state=42)

            train_df = train_set.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)
            test_df = test_set.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)

            logging.info('Train Test Split completed successfully')

            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("Failed to initiate data")
            raise CustomException(e,sys)