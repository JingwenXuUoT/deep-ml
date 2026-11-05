import pandas as pd

def solution(df1, df2, df3):
    merged_inner = pd.merge(df1, df2, on='emp_id', how='inner')
    final_df = pd.merge(merged_inner, df3, on='emp_id', how='left')
    final_df = final_df[['emp_id', 'name', 'dept', 'salary']]
    return final_df.reset_index(drop=True)