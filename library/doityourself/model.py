"""Model from scratch.
"""
import os

import pandas as pd


class Model(object):
    """Class Model
    """
    _params_folder = "library/doityourself/params/"

    def __init__(self):
        """Init an instance of the model.
        """
        # self._params_folder = "library/doityourself/params/"

    def predict(self, data_df):
        """Perform a prediction.

        Args:
            data_df (DataFrame): table to predict

        Return:
            prediction_df (DataFrame): prediction
        """
        # Declare prediction table
        prediction_df = pd.DataFrame()

        # Perform the prediction for the whole table
        for index, row in data_df.iterrows():
            # To be design
            input_predict = row
            # Perform prediction
            prediction_df.loc[index, "prediction"] = self.predict_record(input_predict)

        # Return prediction table
        return prediction_df

    def fit(self, data_df):
        """Perform a training.

        Args:
            data_df (DataFrame): table to use for fitting.
        Return:
            (bool): True if the training went well.
        """

    def load_parameters(self, model_version):
        """Load parameters of the model.

        Args:
            model_version (str): version of model to use for loading.
        Return:
            None
        """
        # Set folder path
        folder_path = self._params_folder + model_version

        # Create folder
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)

        # TOOO Load



    def persist_parameters(self, model_version):
        """Persist parameters of the model.

        Args:
            model_version (str): version of model to persist.
        Return:
            None
        """
        # Set folder path
        folder_path = self._params_folder + model_version

        # Create folder
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)

        # TOOO Load

    def predict_record(self, input_predict):
        """Perform a prediction of one given record.

        Args:
            input_predict (? TO BE DEFINED): Input to predict.

        Return:
            prediction (int): predicted category.
        """

