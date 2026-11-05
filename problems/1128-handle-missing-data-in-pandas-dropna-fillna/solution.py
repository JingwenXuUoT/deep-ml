import pandas as pd

def solution(df):
    # Create a copy to avoid modifying the original dataframe in-place
    df_cleaned = df.copy()
    
    # 1. Drop any column whose fraction of missing values is greater than 0.5
    null_cols_fraction = df_cleaned.isna().mean()
    cols_to_keep = null_cols_fraction[null_cols_fraction <= 0.5].index
    df_cleaned = df_cleaned[cols_to_keep]
    
    # 2. After dropping those columns, drop any row whose fraction of missing values
    #    (across the remaining columns) is greater than 0.5
    if not df_cleaned.empty and len(df_cleaned.columns) > 0:
        null_rows_fraction = df_cleaned.isna().mean(axis=1)
        df_cleaned = df_cleaned[null_rows_fraction <= 0.5]
        
    # 3. Fill the remaining missing values:
    #    - For numeric columns, use the column mean.
    #    - For non-numeric columns, use the column's most frequent value (first mode).
    for col in df_cleaned.columns:
        if df_cleaned[col].isna().any():
            if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                # Fill with column mean
                col_mean = df_cleaned[col].mean()
                df_cleaned[col] = df_cleaned[col].fillna(col_mean)
            else:
                # Fill with first mode (most frequent value)
                col_mode_series = df_cleaned[col].mode()
                if not col_mode_series.empty:
                    col_mode = col_mode_series.iloc[0]
                    df_cleaned[col] = df_cleaned[col].fillna(col_mode)
                    
    # Return the resulting DataFrame with its index reset (0, 1, 2, ...)
    return df_cleaned.reset_index(drop=True)