import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.features.preprocessing import load_data, clean_data


def test_load_data():
    """Test that data loads correctly."""
    df = load_data("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    assert df.shape[0] == 7043
    assert df.shape[1] == 21


def test_clean_data():
    """Test that data cleaning removes customerID and fixes TotalCharges."""
    df = load_data("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df_clean = clean_data(df)
    assert "customerID" not in df_clean.columns
    assert df_clean["TotalCharges"].isnull().sum() == 0