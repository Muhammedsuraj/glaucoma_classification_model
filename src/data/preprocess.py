import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess_data(df: pd.DataFrame, target_col: str = "casetype") -> pd.DataFrame:
    """
    Basic cleaning.
    - trim column names
    - fix heightcm1 to numeric
    - simple NA handling
    """
    # tidy headers
    df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace

    # heightcm1 often has some strings in its instances -> coerce to float
    if "heightcm1" in df.columns:
        df["heightcm1"] = pd.to_numeric(df["heightcm1"], errors="coerce")

    if "casetype" in df.columns:
        le = LabelEncoder()
        df["casetype"] = le.fit_transform(df["casetype"])

    # simple NA strategy:
    # - numeric: fill with 0
    # - others: leave for encoders to handle (get_dummies ignores NaN safely)
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)
    
    return df
