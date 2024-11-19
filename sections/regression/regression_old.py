from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression, Ridge
import matplotlib.pyplot as plt



def regression_page():
    st.header("Bienvenue")
    st.caption("Bienvenue dans le Playground de Regression")
    

# Load CSV file
df = pd.read_csv('data/diabete.csv')

# Show the first few rows of dataset
df_head = df.head()
st.write("DataFrame head:",df_head)

# Drop the "Unnamed: 0" column
df.drop(columns=['Unnamed: 0'], inplace=True)

# Show DataFrame information: data types and non-null values
df_info = df.info()
st.write( "DataFrame information :", df_info)

# Show descriptive statistics for numerical columns
df_desc = df.describe()
st.write("Descriptive statistics :", df_desc)

# Check for missing values
missing_values = df.isna().sum()
st.write("Missing values in the DataFrame:",missing_values)


### Model 1 : simple lineair Regression
st.write("***Model 1: Linear Regression")
#Separating data:
X = df.drop(columns=['target'])
y = df['target']

#Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Show the shapes of data and labels for training and testing
st.write(f"Training data shape: {X_train.shape}")
st.write(f"Test data shape: {X_test.shape}")
st.write(f"Training labels shape: {y_train.shape}")
st.write(f"Test labels shape: {y_test.shape}")

#Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Display model coefficients and intercept
st.write("Model Coefficients : ", model.coef_)
st.write("Model Intercept : ", model.intercept_)

#Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

#Evaluate model
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

#Display predictions vs true values
st.write(f"Training Mean Squared Error (MSE): {train_mse}")
st.write(f"Training Coefficient of Determination (R²): {train_r2}")
st.write(f"Test Mean Squared Error (MSE): {test_mse}")
st.write(f"Test Coefficient of Determination (R²): {test_r2}")

#Plotting predictions vs test results
fig_test, ax_test = plt.subplots()
ax_test.scatter(y_test, y_test_pred, color='blue', label="Test Predictions")
ax_test.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2, label="Perfect Prediction")
ax_test.set_xlabel('True Values')
ax_test.set_ylabel('Predictions')
ax_test.set_title('True Values vs Predictions (Test Set)')
ax_test.legend()
st.pyplot(fig_test)

# Plotting predctions vs train results
fig_train, ax_train = plt.subplots()
ax_train.scatter(y_train, y_train_pred, color='green', label="Train Predictions")
ax_train.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linewidth=2, label="Perfect Prediction")
ax_train.set_xlabel('True Values')
ax_train.set_ylabel('Predictions')
ax_train.set_title('True Values vs Predictions (Train Set)')
ax_train.legend()
st.pyplot(fig_train)



### Model 2 : Feauture enginireeinig
st.write("***Model 2:")
# Interaction features
df['age_bmi_interaction'] = df['age'] * df['bmi']
df['bp_age_interaction'] = df['bp'] * df['age']
df['sex_bmi_interaction'] = df['sex'] * df['bmi']
df['sex_age_interaction'] = df['sex'] * df['age']

#Separting X and Y
X = df.drop('target', axis=1)
y = df['target']

#Divisin and entrain model :
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

st.write(f"Training data shape: {X_train.shape}")
st.write(f"Test data shape: {X_test.shape}")
st.write(f"Training labels shape: {y_train.shape}")
st.write(f"Test labels shape: {y_test.shape}")

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Display model coeffcients and intercept :
st.write("Model Coefficients:", model.coef_)
st.write("Model Intercept:", model.intercept_)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluate model
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

#Show Metrics
st.write(f"Training Mean Squared Error (MSE): {train_mse}")
st.write(f"Training Coefficient of Determination (R²): {train_r2}")
st.write(f"Test Mean Squared Error (MSE): {test_mse}")
st.write(f"Test Coefficient of Determination (R²): {test_r2}")

#comparaison real and test
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})
st.write("Comparing Predicted vs Actual values (Test Set)")
st.write(comparison_df)

# Plotting  Predictions vs test
fig_test, ax_test = plt.subplots()
ax_test.scatter(y_test, y_test_pred, color='blue', label="Test Predictions")
ax_test.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2, label="Perfect Prediction")
ax_test.set_xlabel('True Values')
ax_test.set_ylabel('Predictions')
ax_test.set_title('True Values vs Predictions (Test Set)')
ax_test.legend()
st.pyplot(fig_test)

# Plotting predictions vs train 
fig_train, ax_train = plt.subplots()
ax_train.scatter(y_train, y_train_pred, color='green', label="Train Predictions")
ax_train.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linewidth=2, label="Perfect Prediction")
ax_train.set_xlabel('True Values')
ax_train.set_ylabel('Predictions')
ax_train.set_title('True Values vs Predictions (Train Set)')
ax_train.legend()
st.pyplot(fig_train)


#### Model 3 : Using Cross-Validation
st.write("***Model 3: Cross-Validation")

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#  Cross-validation to get MSE scores
model = LinearRegression()
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

# Convert the negative MSE values to positive 
cv_mse = -cv_scores

# Calculate the mean and the std of the MSE across the folds
mean_mse = np.mean(cv_mse)
std_mse = np.std(cv_mse)

# Calculate R² scores for cross-validation
cv_r2_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
mean_r2 = np.mean(cv_r2_scores)
std_r2 = np.std(cv_r2_scores)

# Train the model 
model.fit(X_train, y_train)

#Predictions 
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluate on the test set
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Display results
st.write("Cross-Validation Mean Squared Error (MSE) across 5 folds:", mean_mse)
st.write("Cross-Validation Std of MSE:", std_mse)
st.write("Cross-Validation R² across 5 folds:", mean_r2)
st.write("Cross-Validation Std of R²:", std_r2)
st.write(f"Training Mean Squared Error (MSE): {train_mse:.2f}")
st.write(f"Training R²: {train_r2:.2f}")
st.write(f"Test Mean Squared Error (MSE): {test_mse:.2f}")
st.write(f"Test R²: {test_r2:.2f}")

# Plotting MSE: Cross-Validation vs Test ans train results
fig_mse, ax_mse = plt.subplots(figsize=(10, 6))
ax_mse.bar(range(1, 6), cv_mse, color='blue', label='CV MSE')
ax_mse.axhline(mean_mse, color='red', linestyle='--', label=f'Mean CV MSE = {mean_mse:.2f}')
ax_mse.bar(6, test_mse, color='yellow', label=f'Test Set MSE = {test_mse:.2f}')
ax_mse.bar(7, train_mse, color='green', label=f'Train Set MSE = {train_mse:.2f}')

ax_mse.set_xticks(range(1, 8))
ax_mse.set_xticklabels([f'Fold {i}' for i in range(1, 6)] + ['Test', 'Train'])
ax_mse.set_xlabel('Fold')
ax_mse.set_ylabel('Mean Squared Error')
ax_mse.set_title('Comparison of MSE Scores: Cross-Validation, Test, and Train Sets')
ax_mse.legend()
st.pyplot(fig_mse)

# Plotting R² : Cross-Validation vs Test and train results
fig_r2, ax_r2 = plt.subplots(figsize=(10, 6))
ax_r2.bar(range(1, 6), cv_r2_scores, color='purple', label='CV R² ')
ax_r2.axhline(mean_r2, color='red', linestyle='--', label=f'Mean CV R² = {mean_r2:.2f}')
ax_r2.bar(6, test_r2, color='yellow', label=f'Test Set R² = {test_r2:.2f}')
ax_r2.bar(7, train_r2, color='green', label=f'Train Set R² = {train_r2:.2f}')

ax_r2.set_xticks(range(1, 8))
ax_r2.set_xticklabels([f'Fold {i}' for i in range(1, 6)] + ['Test', 'Train'])
ax_r2.set_xlabel('Fold')
ax_r2.set_ylabel('R²')
ax_r2.set_title('Comparison of R² Scores: Cross-Validation, Test, and Train Sets')
ax_r2.legend()
st.pyplot(fig_r2)

#### Model 4: Ridge Regression
st.write("***Model 4: Ridge Regression")

X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Show shapes data
st.write(f"Training data shape: {X_train.shape}")
st.write(f"Test data shape: {X_test.shape}")
st.write(f"Training labels shape: {y_train.shape}")
st.write(f"Test labels shape: {y_test.shape}")

#Inisialization Model Ridge 
model = Ridge(alpha=0.1)

#Train model 
model.fit(X_train, y_train)

#Show shapes coeeficients
st.write("Model Coefficients: ", model.coef_)
st.write("Model Intercept: ", model.intercept_)

#Predictions 
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

#Calculate mse and r2
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

#Displayin reuslts in streamlit 
st.write(f"Training Mean Squared Error (MSE): {train_mse}")
st.write(f"Training R²: {train_r2}")
st.write(f"Test Mean Squared Error (MSE): {test_mse}")
st.write(f"Test R²: {test_r2}")

#Visulaization of results
fig_ridge, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, y_test_pred, color='blue', label="Predictions vs True values")
ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2, label="Perfect Prediction")
ax.set_xlabel('True Values')
ax.set_ylabel('Predictions')
ax.set_title('True Values vs Predictions for Test Set')
ax.legend()
st.pyplot(fig_ridge)