
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer


def _create_binary_mappings():
    """
    Define deterministic binary mappings for consistency between training and serving.
    """
    return {
        frozenset({"Yes", "No"}): {"No": 0, "Yes": 1},
        frozenset({"Male", "Female"}): {"Female": 0, "Male": 1},
    }

def _create_ordinal_mappings():
    """
    Define deterministic ordinal mappings for consistency between training and serving.
    """
    return {
        frozenset({"Low", "Medium", "High"}): {"Low": 0, "Medium": 1, "High": 2},
        frozenset({"Bronze", "Silver", "Gold", "Platinum"}): {"Bronze": 0, "Silver": 1, "Gold": 2, "Platinum": 3},
    }
def _create_datetime_mappings():
    """
    Define deterministic datetime mappings for consistency between training and serving.
    """
    return {
        "2020-01-01": 0,
        "2020-06-01": 1,
        "2021-01-01": 2,
        "2021-06-01": 3,
    }

def _create_categorical_mappings():
    """
    Define deterministic categorical mappings for consistency between training and serving.
    """
    return {
        frozenset({"Yes", "No"}): {"No": 0, "Yes": 1},
        frozenset({"Male", "Female"}): {"Female": 0, "Male": 1},
        frozenset({"Bronze", "Silver", "Gold", "Platinum"}): {"Bronze": 0, "Silver": 1, "Gold": 2, "Platinum": 3},
        frozenset({"Low", "Medium", "High"}): {"Low": 0, "Medium": 1, "High": 2},
        frozenset({"2020-01-01", "2020-06-01", "2021-01-01", "2021-06-01"}): {"2020-01-01": 0, "2020-06-01": 1, "2021-01-01": 2, "2021-06-01": 3},
    }
def _create_feature_transformer(categorical_cols, numeric_cols, ordinal_cols, datetime_cols):
    """ 
    Create a feature transformer that will be used to encode categorical and numeric features.
    """
    # Create a list of transformers to be used in the feature transformer
    transformers = []

    # Encode categorical features
    if categorical_cols:
        transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore")))

    # Encode numeric features
    if numeric_cols:
        transformers.append(("numeric", OrdinalEncoder(handle_unknown="ignore")))

    # Encode ordinal features
    if ordinal_cols:
        transformers.append(("ordinal", OrdinalEncoder(handle_unknown="ignore")))

    # Encode datetime features
    if datetime_cols:
        transformers.append(("datetime", OrdinalEncoder(handle_unknown="ignore")))