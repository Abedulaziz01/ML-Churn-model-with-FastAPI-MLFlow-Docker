
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


