import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
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

# 4.1 KNN Regressor
knn = neighbors.KNeighborsRegressor(n_neighbors=3, p=2)
knn.fit(X_train, y_train)

# 4.2 Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# 4.3 Decision Tree Regressor
dt_reg = DecisionTreeRegressor(random_state=1)
dt_reg.fit(X_train, y_train)

# 4.4 Support Vector Regressor (SVR)
svr = SVR(kernel='rbf')
svr.fit(X_train, y_train)

# ====== 5. Model Prediction ======

# 5.1 KNN Predictions
y_predict_knn = knn.predict(X_test)

# 5.2 Linear Regression Predictions
y_predict_lr = lin_reg.predict(X_test)

# 5.3 Decision Tree Predictions
y_predict_dt = dt_reg.predict(X_test)

# 5.4 SVR Predictions
y_predict_svr = svr.predict(X_test)

# ====== 6. Model Evaluation ======

# Initialize dictionaries to store evaluation metrics
evaluation_metrics = {
    'KNN': {},
    'Linear Regression': {},
    'Decision Tree': {},
    'SVR': {}
}

# 6.1 KNN Evaluation
evaluation_metrics['KNN']['MSE'] = mean_squared_error(y_test, y_predict_knn)
evaluation_metrics['KNN']['MAE'] = mean_absolute_error(y_test, y_predict_knn)
evaluation_metrics['KNN']['RMSE'] = np.sqrt(evaluation_metrics['KNN']['MSE'])

# 6.2 Linear Regression Evaluation
evaluation_metrics['Linear Regression']['MSE'] = mean_squared_error(y_test, y_predict_lr)
evaluation_metrics['Linear Regression']['MAE'] = mean_absolute_error(y_test, y_predict_lr)
evaluation_metrics['Linear Regression']['RMSE'] = np.sqrt(evaluation_metrics['Linear Regression']['MSE'])

# 6.3 Decision Tree Evaluation
evaluation_metrics['Decision Tree']['MSE'] = mean_squared_error(y_test, y_predict_dt)
evaluation_metrics['Decision Tree']['MAE'] = mean_absolute_error(y_test, y_predict_dt)
evaluation_metrics['Decision Tree']['RMSE'] = np.sqrt(evaluation_metrics['Decision Tree']['MSE'])

# 6.4 SVR Evaluation
evaluation_metrics['SVR']['MSE'] = mean_squared_error(y_test, y_predict_svr)
evaluation_metrics['SVR']['MAE'] = mean_absolute_error(y_test, y_predict_svr)
evaluation_metrics['SVR']['RMSE'] = np.sqrt(evaluation_metrics['SVR']['MSE'])

# Print Evaluation Metrics
for model in evaluation_metrics:
    print(f"=== {model} Regression Evaluation ===")
    print(f"MSE: {evaluation_metrics[model]['MSE']:.2f}")
    print(f"MAE: {evaluation_metrics[model]['MAE']:.2f}")
    print(f"RMSE: {evaluation_metrics[model]['RMSE']:.2f}\n")

# ====== 7. Plotting Original vs Predicted ======

plt.figure(figsize=(14, 8))
plt.plot(range(len(y_test)), y_test, 'ro', label='Actual Data')

# Plot Predictions from All Models
plt.plot(range(len(y_predict_knn)), y_predict_knn, 'bo', label='KNN Predicted')
plt.plot(range(len(y_predict_lr)), y_predict_lr, 'gs', label='Linear Regression Predicted')
plt.plot(range(len(y_predict_dt)), y_predict_dt, 'm^', label='Decision Tree Predicted')  # 'm^' = magenta triangles
plt.plot(range(len(y_predict_svr)), y_predict_svr, 'cD', label='SVR Predicted')        # 'cD' = cyan diamonds

# Plot lines connecting actual and predicted data points for each model
for i in range(len(y_test)):
    # KNN
    plt.plot([i, i], [y_test[i], y_predict_knn[i]], 'b-', linewidth=0.5)
    # Linear Regression
    plt.plot([i, i], [y_test[i], y_predict_lr[i]], 'g--', linewidth=0.5)
    # Decision Tree
    plt.plot([i, i], [y_test[i], y_predict_dt[i]], 'm-.', linewidth=0.5)
    # SVR
    plt.plot([i, i], [y_test[i], y_predict_svr[i]], 'c:', linewidth=0.5)

plt.title('Actual vs Predicted Performance Index by Different Models')
plt.xlabel('Sample Index')
plt.ylabel('Performance Index')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ====== 8. Performance Comparison Bar Chart ======

# Define metrics
metrics = ['MSE', 'MAE', 'RMSE']
models = ['KNN', 'Linear Regression', 'Decision Tree', 'SVR']
colors = ['skyblue', 'lightgreen', 'salmon', 'plum']

# Prepare data for plotting
knn_scores = [evaluation_metrics['KNN'][metric] for metric in metrics]
lr_scores = [evaluation_metrics['Linear Regression'][metric] for metric in metrics]
dt_scores = [evaluation_metrics['Decision Tree'][metric] for metric in metrics]
svr_scores = [evaluation_metrics['SVR'][metric] for metric in metrics]

x = np.arange(len(metrics))  # label locations
width = 0.2  # width of the bars

fig, ax = plt.subplots(figsize=(10,7))

# Plot bars for each model
rects1 = ax.bar(x - 1.5*width, knn_scores, width, label='KNN', color=colors[0])
rects2 = ax.bar(x - 0.5*width, lr_scores, width, label='Linear Regression', color=colors[1])
rects3 = ax.bar(x + 0.5*width, dt_scores, width, label='Decision Tree', color=colors[2])
rects4 = ax.bar(x + 1.5*width, svr_scores, width, label='SVR', color=colors[3])

# Add some text for labels, title and custom x-axis tick labels
ax.set_ylabel('Scores')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

# Function to attach a text label above each bar, displaying its height
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# Apply autolabel to each set of bars
autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

fig.tight_layout()
plt.show()

# ====== 9. Box Plot of Actual vs Predictions ======

plt.figure(figsize=(12, 8))
data = [y_test, y_predict_knn, y_predict_lr, y_predict_dt, y_predict_svr]
labels = ['Actual', 'KNN Predicted', 'Linear Regression Predicted', 'Decision Tree Predicted', 'SVR Predicted']
colors_box = ['lightblue', 'lightgreen', 'salmon', 'plum', 'lightcyan']

box = plt.boxplot(data, labels=labels, patch_artist=True,
                 boxprops=dict(facecolor='lightblue'),
                 medianprops=dict(color='red'))

# Customize box colors
for patch, color in zip(box['boxes'], colors_box):
    patch.set_facecolor(color)

plt.title('Box Plot of Actual vs Predicted Performance Index by Different Models')
plt.ylabel('Performance Index')
plt.grid(True)
plt.tight_layout()
plt.show()

# ====== 10. Pie Charts of Error Distribution ======

# Define error ranges
def categorize_errors(y_true, y_pred):
    errors = np.abs(y_true - y_pred)
    categories = {
        '0-5': 0,
        '5-10': 0,
        '10-15': 0,
        '15+': 0
    }
    for error in errors:
        if error <=5:
            categories['0-5'] +=1
        elif error <=10:
            categories['5-10'] +=1
        elif error <=15:
            categories['10-15'] +=1
        else:
            categories['15+'] +=1
    return categories

# KNN Error Categories
categories_knn = categorize_errors(y_test, y_predict_knn)
labels_knn = list(categories_knn.keys())
sizes_knn = list(categories_knn.values())

# Linear Regression Error Categories
categories_lr = categorize_errors(y_test, y_predict_lr)
labels_lr = list(categories_lr.keys())
sizes_lr = list(categories_lr.values())

# Decision Tree Error Categories
categories_dt = categorize_errors(y_test, y_predict_dt)
labels_dt = list(categories_dt.keys())
sizes_dt = list(categories_dt.values())

# SVR Error Categories
categories_svr = categorize_errors(y_test, y_predict_svr)
labels_svr = list(categories_svr.keys())
sizes_svr = list(categories_svr.values())

# Plot Pie Charts Side by Side
fig, axs = plt.subplots(2, 2, figsize=(16,12))

# KNN Pie Chart
axs[0,0].pie(sizes_knn, labels=labels_knn, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
axs[0,0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
axs[0,0].set_title('KNN Prediction Error Distribution')

# Linear Regression Pie Chart
axs[0,1].pie(sizes_lr, labels=labels_lr, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
axs[0,1].axis('equal')
axs[0,1].set_title('Linear Regression Prediction Error Distribution')

# Decision Tree Pie Chart
axs[1,0].pie(sizes_dt, labels=labels_dt, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
axs[1,0].axis('equal')
axs[1,0].set_title('Decision Tree Prediction Error Distribution')

# SVR Pie Chart
axs[1,1].pie(sizes_svr, labels=labels_svr, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
axs[1,1].axis('equal')
axs[1,1].set_title('SVR Prediction Error Distribution')

plt.tight_layout()
plt.show()

# ====== 11. User Input and Prediction ======

def predict_student_performance(model, scaler, model_name='Model'):
    print(f"\nEnter the following details to predict the student's Performance Index using {model_name}:")
    try:
        hours_studied = float(input("Hours Studied: "))
        previous_scores = float(input("Previous Scores: "))
        extracurricular_activities = float(input("Extracurricular Activities (e.g., number of activities): "))
        sleep_hours = float(input("Sleep Hours: "))
        sample_question_papers_practiced = float(input("Sample Question Papers Practiced: "))
    except ValueError:
        print("Invalid input. Please enter numerical values.")
        return None, None

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

    print(f"\nPredicted Performance Index using {model_name}: {predicted_performance[0]:.2f}")

    return predicted_performance[0], input_features_scaled

# ====== 12. User Predictions for All Models ======

# Predict using KNN
user_prediction_knn, user_input_scaled_knn = predict_student_performance(knn, scaler, model_name='KNN')

# Predict using Linear Regression
user_prediction_lr, user_input_scaled_lr = predict_student_performance(lin_reg, scaler, model_name='Linear Regression')

# Predict using Decision Tree
user_prediction_dt, user_input_scaled_dt = predict_student_performance(dt_reg, scaler, model_name='Decision Tree')
    
# Predict using SVR
user_prediction_svr, user_input_scaled_svr = predict_student_performance(svr, scaler, model_name='SVR')

# ====== 13. Visualization of User
plt.figure(figsize=(14, 8))
plt.plot(range(len(y_test)), y_test, 'ro', label='Actual Data')

# Plot Predictions from All Models
plt.plot(range(len(y_predict_knn)), y_predict_knn, 'bo', label='KNN Predicted')
plt.plot(range(len(y_predict_lr)), y_predict_lr, 'gs', label='Linear Regression Predicted')
plt.plot(range(len(y_predict_dt)), y_predict_dt, 'm^', label='Decision Tree Predicted')  # 'm^' = magenta triangles
plt.plot(range(len(y_predict_svr)), y_predict_svr, 'cD', label='SVR Predicted')        # 'cD' = cyan diamonds

# Plot lines connecting actual and predicted data points for each model
for i in range(len(y_test)):
    # KNN
    plt.plot([i, i], [y_test[i], y_predict_knn[i]], 'b-', linewidth=0.5)
    # Linear Regression
    plt.plot([i, i], [y_test[i], y_predict_lr[i]], 'g--', linewidth=0.5)
    # Decision Tree
    plt.plot([i, i], [y_test[i], y_predict_dt[i]], 'm-.', linewidth=0.5)
    # SVR
    plt.plot([i, i], [y_test[i], y_predict_svr[i]], 'c:', linewidth=0.5)

# Plot the user predictions
    if (user_prediction_knn is not None and 
        user_prediction_lr is not None and 
        user_prediction_dt is not None and 
        user_prediction_svr is not None):
        user_index = len(y_test) + 1
        plt.plot(user_index, user_prediction_knn, 'ms', label='User Prediction KNN', markersize=8)  # 'ms' = magenta square
        plt.plot(user_index, user_prediction_lr, 'md', label='User Prediction LR', markersize=8)     # 'md' = magenta diamond
        plt.plot(user_index, user_prediction_dt, 'mp', label='User Prediction DT', markersize=8)     # 'mp' = magenta pentagon
        plt.plot(user_index, user_prediction_svr, 'mh', label='User Prediction SVR', markersize=8)    # 'mh' = magenta hexagon