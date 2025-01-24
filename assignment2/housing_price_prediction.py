import pandas as pd
from sklearn.linear_model import LinearRegression
import os
import matplotlib.pyplot as plt

def preprocessdata(file_path: str) -> pd.DataFrame:
    '''
    Preprocess the data into a pandas dataframe.
    '''
    df = pd.read_csv(file_path)
    df = df.dropna()
    return df

def make_histograms(df: pd.DataFrame):
    '''
    Creates a histogram for each parameter showing data density.
    '''
    parameters = ['Price', 'Rooms', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'YearBuilt']
    for parameter in parameters:
        df[parameter].plot(kind='hist', bins=10, edgecolor='black', color='skyblue')
        plt.xlabel(parameter)
        plt.ylabel('Frequency')
        plt.title(f'{parameter} Distribution')
        plt.savefig(f"assignment2/{parameter}_Distribution.png", dpi= 500)
        plt.close()

def plot_data_and_regression(df: pd.DataFrame):
    '''
    Filter data and run linear regression through parameters and plot.
    '''
    parameters = ['Rooms', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'YearBuilt']
    for type in ['h', 'u', 't']:
        for parameter in parameters:
            temp_df = df[df['Type'] == type]
            # temp_df = temp_df[temp_df['Landsize'] < 1800]
            if type == 'h':
                year = 1960
                land = 1300
                area = 500
                type_str = "House"
            elif type == 'u':
                year = 1925
                land = 2500
                area = 200
                type_str = "Apartment"
            elif type == 't':
                year = 1960
                land = 800
                area = 300
                type_str = "Townhome"
            temp_df = temp_df[temp_df['YearBuilt'] > year]
            temp_df = temp_df[temp_df['Landsize'] < land]
            temp_df = temp_df[temp_df['BuildingArea'] < area]

            X = temp_df[[parameter]]
            Y = temp_df['Price']
            model = LinearRegression()
            model.fit(X, Y)
            temp_df['Predicted_Price'] = model.predict(X)
            r2 = model.score(X, Y)
            plt.scatter(temp_df[[parameter]], temp_df['Price'], color='blue', label='Raw Data')
            plt.plot(temp_df[parameter], temp_df['Predicted_Price'], color='green', label='Regression Line')
            plt.xlabel(f'{parameter}')
            plt.ylabel('Price $')
            plt.title(f'{type_str} - {parameter} vs Price - Linear Regression Fit')
            plt.legend()
            plt.savefig(f"assignment2/linear_regression/{type_str}_{parameter}_linear_regression.png", dpi= 500)
            plt.close()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "house_prices.csv")
    df = preprocessdata(file_path)
    make_histograms(df)
    plot_data_and_regression(df)
