import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw telecom dataset."""
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Dataset loaded : {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw data."""
    logger.info("Cleaning data...")

    # Fix TotalCharges column (contains spaces)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Drop customerID (not useful for prediction)
    df.drop(columns=["customerID"], inplace=True)

    logger.info(f"Data cleaned : {df.shape[0]} rows remaining")
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features."""
    logger.info("Encoding categorical features...")

    # Binary encoding
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService",
                   "PaperlessBilling", "Churn"]
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0,
                                "Male": 1, "Female": 0})

    # Label encoding for other categorical columns
    cat_cols = df.select_dtypes(include="object").columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    logger.info("Encoding done.")
    return df


def scale_features(df: pd.DataFrame,
                   target_col: str = "Churn") -> tuple:
    """Scale numerical features."""
    logger.info("Scaling numerical features...")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    logger.info("Scaling done.")
    return X_scaled, y


def run_preprocessing(filepath: str) -> tuple:
    """Full preprocessing pipeline."""
    df = load_data(filepath)
    df = clean_data(df)
    df = encode_features(df)
    X, y = scale_features(df)
    return X, y


if __name__ == "__main__":
    X, y = run_preprocessing("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print(f"\n✅ Features shape : {X.shape}")
    print(f"✅ Target distribution :\n{y.value_counts()}")