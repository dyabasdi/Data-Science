import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def preprocessdata(file_path: str) -> pd.DataFrame:
    '''
    Preprocess the data into a pandas dataframe.
    '''
    df = pd.read_csv(file_path)

    # Create a new column for tip percentage, and drop tip
    df["tip_percent"] = (df["tip"] / df["total_bill"]) * 100
    df = df.drop(columns=['tip'])

    # assign numerical values when there are strings
    df = pd.get_dummies(df, columns=['sex', 'smoker', 'day', 'time'], drop_first=True)

    return df

def create_model(df: pd.DataFrame):
    '''
    Takes the preprocessed dataframe and outputs a model to predict restaurant tips.
    '''
    # create x and y
    x = df.drop(columns=['tip_percent'])
    y = df['tip_percent']

    # split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # train linear regression model
    model = LinearRegression()
    model.fit(x_train, y_train)

    return model, x

def get_inputs(x) -> pd.DataFrame:
    '''
    Get user inputs for scenario to predict restraunt tips.
    '''
    # Get user input
    while True:
        try:
            total_bill = float(input("Enter total bill amount ($): "))
            if total_bill <= 0:
                print("Please enter a positive number for the total bill.")
                continue
            break  # Exit loop if input is valid
        except ValueError:
            print("Invalid input. Please enter a valid number for the total bill.")
    while True:
        try:
            size = int(input("Enter party size: "))
            if size <= 0:
                print("Party size must be greater than 0.")
                continue
            break  # Exit loop if input is valid
        except ValueError:
            print("Invalid input. Please enter a valid number for party size.")
    while True:
        sex = input("Enter sex (Male/Female): ").strip().capitalize()
        if sex in ['Male', 'Female']:
            break
        else:
            print("Invalid input. Please enter 'Male' or 'Female'.")
    while True:
        smoker = input("Is the customer a smoker? (Yes/No): ").strip().capitalize()
        if smoker in ['Yes', 'No']:
            break
        else:
            print("Invalid input. Please enter 'Yes' or 'No'.")
    while True:
        day = input("Enter day (Thur/Fri/Sat/Sun): ").strip().capitalize()
        if day in ['Thur', 'Fri', 'Sat', 'Sun']:
            break
        else:
            print("Invalid input. Please enter a valid day (Thur, Fri, Sat, or Sun).")

    # Input validation for time (Lunch/Dinner)
    while True:
        time = input("Enter time of day (Lunch/Dinner): ").strip().capitalize()
        if time in ['Lunch', 'Dinner']:
            break
        else:
            print("Invalid input. Please enter 'Lunch' or 'Dinner'.")

    # Convert user input into a DataFrame
    inputs = pd.DataFrame([{
        "total_bill": total_bill,
        "size": size,
        "sex_Male": 1 if sex == "Male" else 0,
        "smoker_Yes": 1 if smoker == "Yes" else 0,
        "day_Fri": 1 if day == "Fri" else 0,
        "day_Sat": 1 if day == "Sat" else 0,
        "day_Sun": 1 if day == "Sun" else 0,
        "time_Dinner": 1 if time == "Dinner" else 0
    }])
    
    # sanity check so we have the same columns as the main dataframe
    for col in x.columns:
        if col not in inputs:
            inputs[col] = 0
    inputs = inputs[x.columns]

    return inputs

def predict_restraunt_tip(model, inputs: pd.DataFrame) -> list[float, float]:
    '''
    Use model to predict tips for that scenario.
    '''
    predicted_tip_percent = model.predict(inputs)
    predicted_tip_amount = (predicted_tip_percent[0] / 100) * inputs["total_bill"].values[0]

    return [predicted_tip_amount, predicted_tip_percent[0]]

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "tips.csv")
    threshold = 0.02 # pct as decimal
    df = preprocessdata(file_path)
    model, x = create_model(df)
    inputs = get_inputs(x)
    predictions = predict_restraunt_tip(model, inputs)
    print(f"Predicted Tip Percentage: {predictions[1]:.2f}%")
    print(f"Predicted Tip Amount: ${predictions[0]:.2f}")