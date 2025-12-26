
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
