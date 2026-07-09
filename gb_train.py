import numpy as np
import pandas as pd
 

import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

#loading dataset
df_insurance = pd.read_csv("insurance (1).csv")

X = df_insurance.drop("charges", axis=1)
y = df_insurance["charges"]

numerical_cols = X.select_dtypes(include=["int", "float"]).columns
categorical_cols = X.select_dtypes(include=["object"]).columns



# handling and scaling numerical column
num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# handling and encoding categorical column

cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# combine both column
preprocessor = ColumnTransformer([
    ("num", num_pipe, numerical_cols),
    ("cat", cat_pipe, categorical_cols)
])

# train test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




#primary model pipeline
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

gb_pipeline = Pipeline([
  ("preprocessor", preprocessor),
  ("model", gb_model)
])
#primary model fit and train

gb_pipeline.fit(X_train, y_train)
gb_pipeline_pred = gb_pipeline.predict(X_test)


rmse = np.sqrt(mean_squared_error(y_test, gb_pipeline_pred))
print(f"Root Mean Squared Error: {rmse}")


#saving model 
with open("insurance_prediction_model.pkl", "wb") as f:
  pickle.dump(gb_pipeline, f)




