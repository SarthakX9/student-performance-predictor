import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("dataset.csv")

print("Descriptive Statistics:\n")
print(df.describe())

# Remove outliers using IQR
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)

IQR = Q3 - Q1

df = df[~((df < (Q1 - 1.5 * IQR)) |
          (df > (Q3 + 1.5 * IQR))).any(axis=1)]

print("\nOutliers removed successfully!")

# Split dataset
X = df.drop("final_marks", axis=1)
y = df["final_marks"]

# Normalize data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

print("\nNormalization completed!")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train 3 models
lr = LinearRegression()
dt = DecisionTreeRegressor()
rf = RandomForestRegressor()

lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Predictions
lr_pred = lr.predict(X_test)
dt_pred = dt.predict(X_test)
rf_pred = rf.predict(X_test)

# Model comparison
print("\nModel Comparison Scores:")

print("Linear Regression:", r2_score(y_test, lr_pred))
print("Decision Tree:", r2_score(y_test, dt_pred))
print("Random Forest:", r2_score(y_test, rf_pred))

# Hyperparameter tuning for Random Forest
params = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10]
}

grid = GridSearchCV(RandomForestRegressor(), params)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("\nBest Parameters:", grid.best_params_)

# Save best model
joblib.dump(best_model, "model.pkl")

print("\nFinal optimized model saved as model.pkl")