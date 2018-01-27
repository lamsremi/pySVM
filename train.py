"""Train script.
"""
import importlib

import pandas as pd


def main(data_df=None,
         data_source=None,
         model_type=None,
         initial_version=None,
         trained_version=None):
    """Train function.

    Args:
        data_df (DataFrame)
        data_source (str)
        model_type (str)
        initial_version (str)
        trained_version (str)

    Return:
        bool

    Note:
        * estimate :
            initial_version = None
            trained_version = new_model_version
        * replace :
            initial_version = None
            trained_version = existing_model_version
        * update : [for online training]
            initial_version = existing_model_version
            trained_version = existing_model_version

    """
    # Load data if none
    if data_df is None:
        data_df = load_data(data_source)

    # Init the model
    model = init_model(model_type)

    # Load parameters
    model.load_parameters(model_version=initial_version)

    # Fit the model
    model.fit(data_df)

    # Perist parameters
    model.persist_parameters(model_version=trained_version)

    # Return
    return True


# @tools.debug
def load_data(data_source):
    """
    Load labeled data.
    """
    data_df = pd.read_pickle(
        "data/{}/train_data.pkl".format(data_source))
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
        for model_str in ["scikit_learn_SVC"]:
            main(data_df=None,
                 data_source=source,
                 model_type=model_str,
                 initial_version=None,
                 trained_version=source)
