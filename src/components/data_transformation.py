import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder#, FunctionTransformer
sys.path.insert(0, 'src')
from exception import CustomException
from logger import logging
#from utils import DataTransform
import os

from utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")
            
            target_column_name="class"

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            logging.info("Seperated target column")

            logging.info("Applying one hot encoding.")
            one_hot_encoder=OneHotEncoder(drop='first',handle_unknown='ignore',sparse=False)
            input_feature_train_arr=one_hot_encoder.fit_transform(input_feature_train_df)
            input_feature_test_arr=one_hot_encoder.transform(input_feature_test_df)

            train_arr = pd.DataFrame(data=input_feature_train_arr, columns=one_hot_encoder.get_feature_names_out())
            test_arr = pd.DataFrame(data=input_feature_test_arr, columns=one_hot_encoder.get_feature_names_out())

            train_arr[target_column_name] = target_feature_train_df
            test_arr[target_column_name] = target_feature_test_df

            logging.info(f"Returning train and test data after preprocessing.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=one_hot_encoder

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)

