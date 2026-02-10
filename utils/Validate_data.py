import great_expectations as ge
from typing import Tuple, List
def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for Telco Customer Churn dataset using Great Expectations.
    
    This function implements critical data quality checks that must pass before model training.
    It validates data integrity, business logic constraints, and statistical properties
    that the ML model expects.
    
    """
    print("🔍 Starting data validation with Great Expectations...")
    
    # Convert pandas DataFrame to Great Expectations Dataset
    ge_df = ge.dataset.PandasDataset(df)
    
    # === SCHEMA VALIDATION - ESSENTIAL COLUMNS ===
    print("   📋 Validating schema and required columns...")
    
    # Customer identifier must exist (required for business operations)  
    ge_df.expect_column_to_exist("customerID")
    ge_df.expect_column_values_to_not_be_null("customerID")