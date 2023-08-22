import os
import sys
import pickle
from src.logger import logging
from src.exception import CustomException

def save_object(file_path, object):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as f:
            pickle.dump(object,f)
            
    except Exception as e:
        logging.info("Error saving object")
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logging.info("Error loading object")
        raise CustomException(e,sys)