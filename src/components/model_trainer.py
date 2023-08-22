import numpy as np
import pandas as pd
import os
import sys
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error

@dataclass
class ModelTrainerConfig:
    model_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self, train_array, test_array):
        try:
            logging.info('Model training initiated')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            model = Sequential()
            model.add(Dense(10, activation='relu', input_dim=5))
            model.add(Dense(10, activation='relu'))
            model.add(Dense(1, activation='linear'))

            model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
            model.summary()

            history = model.fit(X_train, y_train, epochs=50, validation_split=0.2)

            y_pred = model.predict(X_test)

            model_score = r2_score(y_test, y_pred)

            print(f'R2 Score: {model_score}')
            print('\n=======================================================================')
            logging.info(f'R2 Score: {model_score}')

            save_object(
                file_path=self.model_trainer_config.model_path,
                object=model
            )

        except Exception as e:
            logging.info('Exception occurred while training model')
            raise CustomException(e, sys)