"""Model implementation using scikit-learn framework

SVC [C-Support Vector Classification.]

http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

"""
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.externals import joblib

import tools

class Model():
    """SVM model class.
    """
    def __init__(self):
        """Init an instance of the Model.
        Attributes:
            _clf
        """
        self._params_path = "library/scikit_learn_SVC/params/"
        self._clf = None

    # @tools.debug
    def predict(self, data_df):
        """Perform a prediction.
        Args:
            data_df (DataFrame): table of input.
        Return:
            predicted_df (DataFrame): table of output.
        """
        # Set input
        X = np.array(data_df)
        # Predict
        y_pred = self._clf.predict(X)
        # Set DataFrame
        prediction_df = pd.DataFrame(y_pred, columns=["prediction"], index=data_df.index)
        # Return table
        return prediction_df

    def fit(self, data_df):
        """Perfor a training.
        Args:
            data_df (DataFrame): table with the target variable at the end.
        Return:

        Note fit method for SVC
            Args:
                X (array-like, sparse matrix): shape (n_samples, n_features)
                y (array-like): shape (n_samples,)
        """
        # Format input
        X = np.array(data_df.iloc[:, :-1])
        # Format output
        y = np.array(data_df.iloc[:, -1])
        # Fit
        self._clf.fit(X, y)

    def load_parameters(self, model_version):
        """Load instance's parameters.

        Load all the parameters required to perform a prediction.
        These are parameters of a model version.

        Args:
            model_version (str): version of the model to load the parameters.
        Return:
            None
        """
        if model_version is None:
            self._clf = SVC(verbose=True)
        else:
            # Set folder name
            folder_path = self._params_path + model_version
            self._clf = joblib.load(folder_path + "/clf.pkl")

    def persist_parameters(self, model_version):
        """Persist instance's parameters.

        Persist all the estimated parameters reauired to perform
        a prediction. These are parameters of a model version.

        Args:
            model_version (str): version of the model use for storing
        """
        # Set folder name
        folder_path = self._params_path + model_version

        # Create folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        # Persist
        joblib.dump(self._clf, folder_path + "/clf.pkl")
