import pandas as pd
import gradio as gr

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


# =========================
# 1) Load dataset
# =========================
DATA_PATH = "insurance.csv"   
df = pd.read_csv(DATA_PATH)

TARGET = "charges"
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Identify columns
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

# =========================
# 2) Preprocess  
# =========================
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

model = GradientBoostingRegressor(random_state=42)

pipe = Pipeline(steps=[
    ("prep", preprocessor),
    ("model", model)
])

# =========================
# 3) Train model  
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipe.fit(X_train, y_train)

pred = pipe.predict(X_test)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))

MODEL_INFO = f"Model: GradientBoostingRegressor | MAE: {mae:,.0f} | RMSE: {rmse:,.0f}"


# =========================
# 4) Prediction function
# =========================
def predict_charges(age, sex, bmi, children, smoker, region):
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }])

    predicted = pipe.predict(input_df)[0]
    return f"Predicted Insurance Charges: ${predicted:,.2f}\n\n({MODEL_INFO})"


# =========================
# 5) Gradio UI
# =========================
demo = gr.Interface(
    fn=predict_charges,
    inputs=[
        gr.Number(label="Age", value=25),
        gr.Dropdown(choices=["male", "female"], label="Sex", value="male"),
        gr.Number(label="BMI", value=27.5),
        gr.Number(label="Children", value=0),
        gr.Dropdown(choices=["yes", "no"], label="Smoker", value="no"),
        gr.Dropdown(choices=["northeast", "northwest", "southeast", "southwest"], label="Region", value="southeast"),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Insurance Charges Prediction",
    description="Enter your information to predict medical insurance charges (trained on insurance.csv).",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
