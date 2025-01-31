import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np

class Model:
    def __init__(self, mae = 100, nodes = None, split = None):
        self.mae = mae
        self.nodes = nodes
        self.split = split

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

def optimization_sweep(df: pd.DataFrame, leaf_nodes: list, trainsplit: list) -> object:
    '''
    Returns the number of leaf nodes and train split that gives us the smallest mean absolute error.
    '''
    best_model = Model()
    for nodes in leaf_nodes:
        for split in trainsplit:
            
            mdl = DecisionTreeRegressor(max_leaf_nodes= nodes)
            
            # split data
            X = df.drop(columns=['Address', 'Price', 'Propertycount'])
            # X = df[['Rooms', 'Distance', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']]
            y = df['Price']
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split, random_state=1)

            # create fit of model
            mdl.fit(X_train, y_train)

            # predict data
            y_pred = mdl.predict(X_test)

            # get mean absolute error between test and predict
            mae_raw = mean_absolute_error(y_test, y_pred)
            mean_test = np.mean(y)
            mae = (mae_raw/mean_test) * 100
            curr_model = Model(mae, nodes, split)
            
            if curr_model.mae < best_model.mae:
                best_model = curr_model

    return best_model

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "house_prices.csv")
    df = preprocess_data(file_path)
    leaf_nodes_sweep = np.linspace(400, 700, 1).astype(int)
    trainsplit_sweep = np.linspace(0.50, 0.95, 10)
    model = optimization_sweep(df, leaf_nodes_sweep, trainsplit_sweep)
    print(f"Optimal Model:\n   Mean Absolute Error = {model.mae}%\n   Training Split = {round(model.split * 100)}% Train, {round((1 - model.split)*100)}% Test\n   Leaf Nodes = {model.nodes}")