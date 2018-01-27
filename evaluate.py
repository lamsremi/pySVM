"""Evaluate the performance of the different models.
"""
import importlib
import pandas as pd

import predict
from performance import confusion_matrix
import tools


pd.set_option('display.width', 800)


# @tools.debug
def main(data_source, model_type, model_version):
    """
    Main evaluate functions.
    Args:
        data_source (str)
        model_type (str)
        model_version (str)
    Return:
    """
    # Load labaled data
    data_df = load_labaled_data(data_source)

    # print(data_df.shape)
    # Load model
    model = init_model(model_type)

    # Load parameters
    model.load_parameters(model_version=model_version)

    # Perform the whole prediction
    prediction_df = predict.main(
        data_df=data_df.iloc[:, :-1],
        model_type=model_type,
        model_version=model_version,
        loaded_model=None)

    # Combine
    results_df = pd.concat(
        [data_df.iloc[:, -1:], prediction_df],
        axis=1,
        join='inner')
    # print(results_df)

    # Assess quantitative performance
    results = confusion_matrix.main(results_df)
    tools.print_elegant(results)


def init_model(model_type):
    """Init model.
    """
    # Import the library
    model_class = importlib.import_module("library.{}.model".format(model_type))

    # Inits the model instance
    model = model_class.Model()

    # Return model
    return model


# @tools.debug
def load_labaled_data(data_source):
    """
    Load labeled data.
    """
    data_df = pd.read_pickle("data/{}/test_data.pkl".format(data_source))
    return data_df


if __name__ == '__main__':
    for source in ["us_election", "titanic"]:
        for model_str in ["scikit_learn_SVC"]:
            main(data_source=source,
                 model_type=model_str,
                 model_version=source)
