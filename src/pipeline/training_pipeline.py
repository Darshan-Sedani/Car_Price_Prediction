import os
import sys
sys.path.append(os.getcwd())
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

if __name__ =='__main__':
    obj = DataIngestion()
    train_path,test_path = obj.initiate_data_ingestion()
    transformation = DataTransformation()
    train_arr,test_arr,_= transformation.initiate_data_transformation(train_path,test_path)
    trainer = ModelTrainer()
    trainer.initate_model_training(train_arr,test_arr)