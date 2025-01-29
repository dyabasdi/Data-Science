import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt

def preprocess_data(file_path: str) -> pd.DataFrame:
    '''
    Preprocess and clean the data in a pandas dataframe.
    '''
    df = pd.read_csv(file_path)
    # remove all data points that have a NaN values and also duplicates
    df = df.dropna()
    df = df.drop_duplicates()
    return df
def split_data(df, categories=['h', 'u', 't']) -> list:
    '''
    Splits data into house, apartment, townhouse.
    '''
    data = []
    for i in range(len(categories)):    
        data.append(df[df['Type'] == categories[i]])
    
    return data
def remove_outliers(df, columns, quantile=0.99) -> pd.DataFrame:
    '''
    Removes outliers from specified columns in the pandas dataframe.
    '''
    for column in columns:
        threshold = df[column].quantile(quantile)
        df = df[df[column] < threshold]
    return df
def create_regression_model(df, columns) -> dict:
    '''
    Create's regression model for a given column.
    Returns a dictionary with the model and MAE for the column.
    '''
    X_train, X_test, y_train, y_test = train_test_split(df[columns], df['Price'], test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    mae_percentage = (mae / y_test.mean()) * 100
    regression_model =  {
                        'Model': model,
                        'MAE': mae_percentage,
                        'Prediction': y_pred
                        }
    return regression_model

if __name__ == "__main__":
    column_models = [
                    'Bathroom',
                    'Bedroom2',
                    'BuildingArea',
                    ]
    outlier_models = column_models
    outlier_models.append('YearBuilt')
    outlier_models.append('Landsize')
    outlier_models.append('Rooms')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "house_prices.csv")
    df = preprocess_data(file_path)
    data = split_data(df)
    for new_df in data:
        new_df = remove_outliers(new_df, outlier_models, quantile=0.90)
        model = create_regression_model(df, column_models)
        
        # for debugging and tuning
        try:
            assert model['MAE'] < 15
            print(f" MAE is {model['MAE']:.2f}%, which is less than 15%!")
        except:
            print(f" MAE is {model['MAE']:.2f}%, which is more than 15%!")
    