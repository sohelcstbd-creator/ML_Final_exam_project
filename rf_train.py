import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


data = pd.read_csv("bangladesh_student_performance.csv")


if 'date' in data.columns:
  data.drop(columns=['date'], inplace=True)


X = data.drop('hsc_result', axis=1)
y = data['hsc_result']

numeric_features = X.select_dtypes(include=['int', 'float']).columns
categorical_features = X.select_dtypes(include=['object']).columns

num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy='median')),
    ("scaler", StandardScaler())
])


cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy='most_frequent')),
    ("encoder", OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, numeric_features),
    ("cat", cat_pipe, categorical_features)
])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


rf_model = RandomForestRegressor(
  n_estimators=200,
  max_depth=10,
  min_samples_split=2, 
  random_state=42,
  n_jobs=-1
)


rf_pipeline = Pipeline([
  ("preprocessor", preprocessor),
  ("model", rf_model)
])

rf_pipeline.fit(X_train, y_train)
y_pred = rf_pipeline.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(rmse)


with open("studen_rf_pipeline.pkl", "wb") as f:
  pickle.dump(rf_pipeline, f)






