import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

# Load data
df = pd.read_csv("insurance.csv")

X = df.drop("charges", axis=1)
y = df["charges"]

cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(exclude="object").columns

# Preprocess
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

# Model
model = GradientBoostingRegressor(random_state=42)

pipe = Pipeline([
    ("prep", preprocessor),
    ("model", model)
])

# Train
pipe.fit(X, y)

# Save model
joblib.dump(pipe, "model.joblib")

print(" Model saved  🔥")
