import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def preprocess_data(file_path: str) -> pd.DataFrame:
    '''
    Preprocess and clean the data in a pandas dataframe.
    '''
    df = pd.read_csv(file_path)

    # Remove all data points that have NaN values and duplicates
    df = df.dropna().drop_duplicates()
    
    object_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    for col in object_cols:
        df[col] = df[col].astype(str)

    return df

def create_random_forest(df: pd.DataFrame) -> RandomForestClassifier:
    '''
    Creates a random forest classifier and returns the model.
    '''
    mdl = RandomForestClassifier(random_state=0)
    X = df.drop(columns=['target'], axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    mdl.fit(X_train, y_train)

    y_test_pred = mdl.predict(X_test)
    accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Accuracy: {accuracy*100:.2f}%")

    return mdl

if __name__ == "__main__":
    file_path = "files/heart_disease.csv"
    df = preprocess_data(file_path)
    mdl = create_random_forest(df)
