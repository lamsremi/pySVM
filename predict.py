"""Predict script.
"""
import importlib

import pandas as pd


def main(data_df=None,
         model_type=None,
         model_version=None,
         loaded_model=None):
    """Predict function.

    Args:
        data_df (DataFrame):
        model_type (str):
        model_version (str):
        loaded_model (loaded model):

    Return:
        prediction_df (DataFrame): prediction df

    """
    # If no loaded model:
    if loaded_model is None:
        # Init the model
        model = init_model(model_type)
        # Load parameters
        model.load_parameters(model_version=model_version)

    # Predict
    prediction_df = model.predict(data_df)

    # Return prediction
    return prediction_df


# @tools.debug
def load_data(data_source):
    """
    Load labeled data.
    """
    data_df = pd.read_pickle(
        "data/{}/test_data.pkl".format(data_source))
    return data_df


def init_model(model_type):
    """Init model.
    """
    # Import the library
    model_class = importlib.import_module("library.{}.model".format(model_type))

    # Inits the model instance
    model = model_class.Model()

    # Return model
    return model

if __name__ == '__main__':
    for source in ["us_election", "titanic"]:
        predict_data = load_data(source).iloc[10:20, :-1]
        for model_str in ["scikit_learn_SVC"]:
            main(data_df=predict_data,
                 model_type=model_str,
                 model_version=source,
                 loaded_model=None)
