import pandas as pd

def solution(df):
    filtered_df = df[df['status'] == 'completed']
    grouped = filtered_df.groupby('region', as_index=False)['amount'].sum()
    result = grouped.sort_values(by='amount', ascending=False).head(10)
    result = result.reset_index(drop=True)

    return result