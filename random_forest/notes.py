import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

def preprocess_data(file_path: str) -> pd.DataFrame:
    '''
    Preprocess and clean the data in a pandas dataframe.
    '''
    df = pd.read_csv(file_path)
    # remove all data points that have a NaN values and also duplicates
    df = df.dropna()
    df = df.drop_duplicates()

    # apply label encoder for columns with strings
    label_encoder = LabelEncoder()
    string_columns = df.select_dtypes(include=['object']).columns.tolist()
    for column in string_columns:
        df[column] = label_encoder.fit_transform(df[column])

    return df

def create_random_forest(df: pd.DataFrame) -> RandomForestRegressor:
    '''
    Creates random forest regressor and returns the model.
    '''
    mdl = RandomForestRegressor(random_state=0)
    X = df.drop(columns=['Address', 'Price', 'Propertycount'], axis=1)
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    mdl.fit(X_train, y_train)

    y_test_pred = mdl.predict(X_test)
    mae = mean_absolute_error(y_test, y_test_pred)
    print(f"Mean Absolute Error: {mae}")

    return mdl

if __name__ == "__main__":
    file_path = "files/house_prices.csv"
    df = preprocess_data(file_path)
    mdl = create_random_forest(df)
