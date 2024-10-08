import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# ====== 1. Data Loading and Preparation ======

# Load the dataset
df = pd.read_csv('Student_Performance.csv')

# Display the first few rows to understand the data structure
print("First 5 entries of the dataset:")
print(df.head())

# Verify that the dataset has at least 6 columns
if df.shape[1] < 6:
    raise ValueError("The dataset must contain at least 6 columns: 5 features and 1 target.")

# Define features (X) and target (y)
# Assuming:
# Column 0: Hours Studied
# Column 1: Previous Scores
# Column 2: Extracurricular Activities
# Column 3: Sleep Hours
# Column 4: Sample Question Papers Practiced
# Column 5: Performance Index (Target)

X = df.iloc[:200, 0:5].values.astype(np.float64)  # Features
y = df.iloc[:200, 5].values.astype(np.float64)    # Target

# ====== 2. Data Scaling ======

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ====== 3. Splitting the Dataset ======

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=1
)

# ====== 4. Model Training ======

knn = neighbors.KNeighborsRegressor(n_neighbors=3, p=2)
knn.fit(X_train, y_train)

# ====== 5. Model Prediction ======

y_predict = knn.predict(X_test)

# ====== 6. Model Evaluation ======

mse = mean_squared_error(y_test, y_predict)
mae = mean_absolute_error(y_test, y_predict)
rmse = np.sqrt(mse)

print(f'MSE score: {mse:.2f}')
print(f'MAE score: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')

# ====== 7. Plotting Original vs Predicted ======

plt.figure(figsize=(12, 6))
plt.plot(range(len(y_test)), y_test, 'ro', label='Original data')
plt.plot(range(len(y_predict)), y_predict, 'bo', label='Predicted data')

# Plot lines connecting original and predicted data points
for i in range(len(y_test)):
    plt.plot([i, i], [y_test[i], y_predict[i]], 'g-', linewidth=0.5)

plt.title('KNN Regression Results')
plt.xlabel('Sample Index')
plt.ylabel('Performance Index')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ====== 8. User Input and Prediction ======

def predict_student_performance(model, scaler):
    print("\nEnter the following details to predict the student's Performance Index:")
    try:
        hours_studied = float(input("Hours Studied: "))
        previous_scores = float(input("Previous Scores: "))
        extracurricular_activities = float(input("Extracurricular Activities (e.g., number of activities): "))
        sleep_hours = float(input("Sleep Hours: "))
        sample_question_papers_practiced = float(input("Sample Question Papers Practiced: "))
    except ValueError:
        print("Invalid input. Please enter numerical values.")
        return

    # Create a numpy array with the input features
    input_features = np.array([[hours_studied,
                                previous_scores,
                                extracurricular_activities,
                                sleep_hours,
                                sample_question_papers_practiced]])

    # Scale the input features using the same scaler as training data
    input_features_scaled = scaler.transform(input_features)

    # Predict the Performance Index
    predicted_performance = model.predict(input_features_scaled)

    print(f"\nPredicted Performance Index: {predicted_performance[0]:.2f}")

    return predicted_performance[0], input_features_scaled

# Call the prediction function
user_prediction, user_input_scaled = predict_student_performance(knn, scaler)

# ====== 9. Visualization of User Prediction ======

if user_prediction:
    # Update the plot to include the user prediction
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(y_test)), y_test, 'ro', label='Original data')
    plt.plot(range(len(y_predict)), y_predict, 'bo', label='Predicted data')

    # Plot lines connecting original and predicted data points
    for i in range(len(y_test)):
        plt.plot([i, i], [y_test[i], y_predict[i]], 'g-', linewidth=0.5)

    # Plot the user prediction
    plt.plot(len(y_test) + 1, user_prediction, 'ms', label='User Prediction', markersize=8)  # 'ms' = magenta square

    # Annotate the user prediction
    plt.annotate(f'User Prediction: {user_prediction:.2f}',
                 xy=(len(y_test) + 1, user_prediction),
                 xytext=(len(y_test) + 1, user_prediction + 5),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 horizontalalignment='center')

    plt.title('KNN Regression Results with User Prediction')
    plt.xlabel('Sample Index')
    plt.ylabel('Performance Index')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
