
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


def _map_binary_series(s: pd.Series) -> pd.Series:
    """
    Apply deterministic binary encoding to 2-category features.
    """
    vals = list(pd.Series(s.dropna().unique()).astype(str))
    valset = frozenset(vals)
    
    mappings = _create_binary_mappings()
    if valset in mappings:
        return s.map(mappings[valset]).astype("Int64")
    
    # Generic binary mapping using alphabetical order
    if len(vals) == 2:
        sorted_vals = sorted(vals)
        mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
        return s.astype(str).map(mapping).astype("Int64")
    
    return s


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features based on domain knowledge for churn prediction.
    """
    df = df.copy()
    
    # Tenure-based features
    if 'tenure' in df.columns:
        # Tenure groups (binning)
        df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72, np.inf], 
                                    labels=['0-12', '13-24', '25-48', '49-72', '73+'])
        # Convert to categorical for encoding
        df['tenure_group'] = df['tenure_group'].astype(str)
    
    # Charge-based features
    if 'MonthlyCharges' in df.columns and 'TotalCharges' in df.columns:
        # Average monthly charges (total / tenure, avoid division by zero)
        df['avg_monthly_charges'] = df['TotalCharges'] / (df['tenure'] + 1)
        
        # Charge ratio (monthly to total)
        df['monthly_to_total_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
        
        # Charge increase indicator (if tenure > 0, monthly > avg)
        df['charge_increase'] = (df['MonthlyCharges'] > df['avg_monthly_charges']).astype(int)
    
    # Service-based features (assuming common telecom columns)
    service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    existing_services = [col for col in service_cols if col in df.columns]
    if existing_services:
        # Total services count
        df['total_services'] = df[existing_services].apply(lambda row: sum(1 for val in row if val == 'Yes'), axis=1)
    
    return df


def build_features(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Apply complete feature engineering pipeline for training data using sklearn transformers.
    
    This function transforms raw customer data into ML-ready features. The transformations
    are designed to be replicated in the serving pipeline for consistency.
    """
    df = df.copy()
    print(f"🔧 Starting feature engineering on {df.shape[1]} columns...")
    
    # === STEP 1: Add Derived Features ===
    df = _add_derived_features(df)
    print(f"   ➕ Added derived features. New shape: {df.shape}")
    
    # Identify feature types
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != target_col]
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    
    print(f"   📊 Found {len(obj_cols)} categorical and {len(numeric_cols)} numeric columns")
    
    # Split categorical by cardinality
    binary_cols = [c for c in obj_cols if df[c].dropna().nunique() == 2]
    multi_cols = [c for c in obj_cols if df[c].dropna().nunique() > 2]
    
    print(f"   🔢 Binary features: {len(binary_cols)} | Multi-category features: {len(multi_cols)}")
    if binary_cols:
        print(f"      Binary: {binary_cols}")
    if multi_cols:
        print(f"      Multi-category: {multi_cols}")
    
    # Apply binary encoding manually for deterministic mappings
    for c in binary_cols:
        original_dtype = df[c].dtype
        df[c] = _map_binary_series(df[c].astype(str))
        print(f"      ✅ {c}: {original_dtype} → binary (0/1)")
    
    # Convert boolean columns to int
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)
        print(f"   🔄 Converted {len(bool_cols)} boolean columns to int: {bool_cols}")
    
    # One-hot encode multi-category features
    if multi_cols:
        print(f"   🌟 Applying one-hot encoding to {len(multi_cols)} multi-category columns...")
        original_shape = df.shape
        
        encoder = OneHotEncoder(drop='first', sparse_output=False, dtype=int)
        encoded = encoder.fit_transform(df[multi_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(multi_cols), index=df.index)
        
        df = df.drop(columns=multi_cols).join(encoded_df)
        
        new_features = df.shape[1] - original_shape[1] + len(multi_cols)
        print(f"      ✅ Created {new_features} new features from {len(multi_cols)} categorical columns")
    
    # Data type cleanup for binary columns
    for c in binary_cols:
        if pd.api.types.is_integer_dtype(df[c]):
            df[c] = df[c].fillna(0).astype(int)
    
    print(f"✅ Feature engineering complete: {df.shape[1]} final features")
    return df