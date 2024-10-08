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

X = df.iloc[0:, 0:5].values.astype(np.float64)  # Features
y = df.iloc[0:, 5].values.astype(np.float64)    # Target

# ====== 2. Data Scaling ======

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ====== 3. Splitting the Dataset ======

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=1
)

# ====== 4. Model Training ======

# Initialize and train models
knn = neighbors.KNeighborsRegressor(n_neighbors=3, p=2)
knn.fit(X_train, y_train)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

dt_reg = DecisionTreeRegressor(random_state=1)
dt_reg.fit(X_train, y_train)

svr = SVR(kernel='rbf')
svr.fit(X_train, y_train)

# ====== 5. Model Prediction ======

y_predict_knn = knn.predict(X_test)
y_predict_lr = lin_reg.predict(X_test)
y_predict_dt = dt_reg.predict(X_test)
y_predict_svr = svr.predict(X_test)

# ====== 6. Model Evaluation ======

# Initialize dictionaries to store evaluation metrics
evaluation_metrics = {
    'KNN': {},
    'Linear Regression': {},
    'Decision Tree': {},
    'SVR': {}
}

# Calculate evaluation metrics for each model
for model_name, y_pred in zip(evaluation_metrics.keys(), [y_predict_knn, y_predict_lr, y_predict_dt, y_predict_svr]):
    evaluation_metrics[model_name]['MSE'] = mean_squared_error(y_test, y_pred)
    evaluation_metrics[model_name]['MAE'] = mean_absolute_error(y_test, y_pred)
    evaluation_metrics[model_name]['RMSE'] = np.sqrt(evaluation_metrics[model_name]['MSE'])

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
    plt.plot([i, i], [y_test[i], y_predict_knn[i]], 'b-', linewidth=0.5)  # KNN
    plt.plot([i, i], [y_test[i], y_predict_lr[i]], 'g--', linewidth=0.5)  # Linear Regression
    plt.plot([i, i], [y_test[i], y_predict_dt[i]], 'm-.', linewidth=0.5)  # Decision Tree
    plt.plot([i, i], [y_test[i], y_predict_svr[i]], 'c:', linewidth=0.5)   # SVR

plt.title('Actual vs Predicted Performance Index by Different Models')
plt.xlabel('Sample Index')
plt.ylabel('Performance Index')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ====== 8. Performance Comparison Bar Chart ======

metrics = ['MSE', 'MAE', 'RMSE']
models = ['KNN', 'Linear Regression', 'Decision Tree', 'SVR']
colors = ['skyblue', 'lightgreen', 'salmon', 'plum']

# Prepare data for plotting
scores = {model: [evaluation_metrics[model][metric] for metric in metrics] for model in models}

x = np.arange(len(metrics))  # label locations
width = 0.2  # width of the bars

fig, ax = plt.subplots(figsize=(10, 7))

# Plot bars for each model
for i, model in enumerate(models):
    ax.bar(x + (i - 1.5) * width, scores[model], width, label=model, color=colors[i])

# Add labels, title, and custom x-axis tick labels
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
for i in range(len(models)):
    autolabel(ax.patches[i * len(metrics):(i + 1) * len(metrics)])

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
        if error <= 5:
            categories['0-5'] += 1
        elif error <= 10:
            categories['5-10'] += 1
        elif error <= 15:
            categories['10-15'] += 1
        else:
            categories['15+'] += 1
    return categories

# Generate error categories for each model
categories = {}
for model_name, y_pred in zip(['KNN', 'Linear Regression', 'Decision Tree', 'SVR'], 
                               [y_predict_knn, y_predict_lr, y_predict_dt, y_predict_svr]):
    categories[model_name] = categorize_errors(y_test, y_pred)

# Plot Pie Charts Side by Side
fig, axs = plt.subplots(2, 2, figsize=(16, 12))

# Pie chart for each model
for ax, (model_name, category) in zip(axs.flatten(), categories.items()):
    labels = list(category.keys())
    sizes = list(category.values())
    
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors_box)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.set_title(f'Error Distribution for {model_name}')

plt.suptitle('Error Distribution of Different Models', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.9)  # Adjust the title to not overlap with the pie charts
plt.show()
