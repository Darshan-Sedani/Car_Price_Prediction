import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model_path = os.path.join('artifacts','model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)

            return pred
        
        except Exception as e:
            logging.info('Failed to Predict output')
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,gender,age,annual_salary,credit_card_debt,net_worth):
        self.gender = gender,
        self.age = age,
        self.annual_salary = monthly_salary,
        self.credit_card_debt = credit_card_debt
        self.net_worth = net_worth
    
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'gender': [self.gender],
                'age': [self.age],
                'annual_salary': [self.annual_salary],
                'credit_card_debt': [self.credit_card_debt],
                'net_worth': [self.net_worth]
            }

            df = pd.DataFrame(custom_data_input_dict)

            logging.info('DataFrame Created')
            return df
        
        except Exception as e:
            logging.info("Error getting data as dataframe")
            raise CustomException(e,sys)

