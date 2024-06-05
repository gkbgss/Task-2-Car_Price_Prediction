import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


#  1: Read the data
data = pd.read_csv("C://Users//user//Downloads//archive (3)//car data.csv")

#  2: Preprocess the data
#  Encode categorical variables
label_encoder = LabelEncoder()
data['Fuel_Type'] = label_encoder.fit_transform(data['Fuel_Type'])
data['Selling_type'] = label_encoder.fit_transform(data['Selling_type'])
data['Transmission'] = label_encoder.fit_transform(data['Transmission'])

# Define feature columns and target column
feature_columns = ['Year', 'Present_Price', 'Driven_kms', 'Fuel_Type', 'Selling_type', 'Transmission', 'Owner']
target_column = 'Selling_Price'

# Split the data into features and target
X = data[feature_columns]
y = data[target_column]

#  3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  4: Train the model
# Using RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Alternatively, you can use XGBoost
# model = xgb.XGBRegressor(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

#  5: Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

#  6: Making a prediction
# Example prediction
new_car_data = {
    'Year': [2015],
    'Present_Price': [10.0],
    'Driven_kms': [50000],
    'Fuel_Type': [1],
    'Selling_type': [0],
    'Transmission': [1],
    'Owner': [0]
}
new_car_df = pd.DataFrame(new_car_data)

predicted_price = model.predict(new_car_df)
print(f"Predicted Selling Price: {predicted_price[0]}")
