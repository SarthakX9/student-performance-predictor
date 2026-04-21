import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


# STEP 1: Load dataset
df = pd.read_csv("dataset2.csv")

print("\nDataset Preview:\n")
print(df.head())

# Show dataset shape
print("\nDataset Shape (Rows, Columns):", df.shape)


# STEP 2: Handle missing values
df = df.fillna(df.mean())

print("\nMissing values handled successfully!")


# STEP 3: Recalculate final_marks (for synthetic dataset consistency)
df["final_marks"] = (
    df["study_hours"] * 5
    + df["attendance"] * 0.3
    + df["previous_marks"] * 0.4
    + df["assignments"] * 2
    + df["sleep_hours"] * 1
)

# Limit marks between 0–100 (realistic exam range)
df["final_marks"] = df["final_marks"].clip(0, 100)


# STEP 4: Descriptive statistics
print("\nDescriptive Statistics:\n")
print(df.describe())


# STEP 5: Handle outliers using IQR method
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)

IQR = Q3 - Q1

df = df[~((df < (Q1 - 1.5 * IQR)) |
          (df > (Q3 + 1.5 * IQR))).any(axis=1)]

print("\nOutliers removed successfully!")

# Show dataset shape after removing outliers
print("\nDataset Shape After Outlier Removal:", df.shape)


# STEP 6: Split features and target
X = df.drop("final_marks", axis=1)
y = df["final_marks"]


# STEP 7: Data normalization
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Convert normalized data into dataframe for display
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

print("\nDataset After Normalization:\n")
print(X_scaled_df.head())

# Show normalized dataset shape
print("\nNormalized Dataset Shape:", X_scaled_df.shape)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

print("\nData normalization completed!")


# STEP 8: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# STEP 9: Train 3 algorithms
lr = LinearRegression()
dt = DecisionTreeRegressor()
rf = RandomForestRegressor(random_state=42)

lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)


# STEP 10: Model comparison
lr_pred = lr.predict(X_test)
dt_pred = dt.predict(X_test)
rf_pred = rf.predict(X_test)

print("\nModel Comparison (R² Score):")

print("Linear Regression Score:", r2_score(y_test, lr_pred))
print("Decision Tree Score:", r2_score(y_test, dt_pred))
print("Random Forest Score:", r2_score(y_test, rf_pred))


# STEP 11: Hyperparameter tuning (Random Forest)
params = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10]
}

grid = GridSearchCV(RandomForestRegressor(random_state=42), params)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("\nBest Parameters after tuning:", grid.best_params_)


# STEP 12: Save final optimized model
joblib.dump(best_model, "model.pkl")

print("\nFinal optimized model saved as model.pkl")
print("Scaler saved as scaler.pkl")