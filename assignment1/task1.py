import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def preprocessdata(file_path: str) -> pd.DataFrame:
    '''
    Preprocess the data into a pandas dataframe.
    '''
    df = pd.read_csv(file_path)

    # create new column for percent
    df['tip_percent'] = df['tip'] / df['total_bill']
    return df

def get_important_data(df: pd.DataFrame, threshold: float):
    '''
    Gets data columns that have correlations in data.
    '''
    variables = []
    for variable in df.columns():
        if variable in ['tip', 'tip_percent']:
            continue # skip iteration
        
        # take the average tip percent at each variable value
        # if we see a difference that exceeds a threshold between them, we have a relationship
        values = np.array()
        for value in df[variable].unique():
            temp_df = df[df[variable] == value]
            values.append(temp_df['tip_percent'].mean())
        
        if np.mean(np.diff(values)) > threshold:
            variables.append(variable)
    
    # return the list of variables that meet a threshold of avg diff
    return variables

def plot_data(df: pd.DataFrame, variables: list) -> None:
    '''
    Plots the data that we determined have relationships.
    '''
    fig, axs = plt.subplot(len(variables), 1, figsize=(8, 12))
    for i,name in enumerate(variables):
        axs[i].plot(df[name], df['tip_percent'])
        axs[i]

if __name__ == "__main__":
    cwd = os.chdir()
    file_path = f"{cwd}/tips.csv"
    threshold = 0.02 # pct as decimal
    df = preprocessdata(file_path)
    variables = get_important_data(df, threshold)
