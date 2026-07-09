import gradio as gr
import numpy as np
import pandas as pd 
import pickle

# 
with open("insurance_prediction_model.pkl", "rb") as f:
    model = pickle.load(f)

# Logic 
def predict_insurance_charges(age, sex, bmi, children, smoker, region):

    # columns into Dataframe
    input_df = pd.DataFrame([[age, sex, bmi, children, smoker, region]],
    columns=["age", "sex", "bmi", "children", "smoker", "region"])
  
    # Predicton
    prediction = model.predict(input_df)[0]
    
    # np.clip
    final_charges = np.clip(prediction, 0, 100000)
  
    return f"Predicted Insurance Charges: {final_charges:.2f}"


# App Interface 
inputs = [
    gr.Dropdown(label="age", value=25, choices=range(18, 65)),
    gr.Radio(["male", "female"], label="sex"), 
    gr.Slider(minimum=10, maximum=50, step=1, value=25, label="bmi"),
    gr.Dropdown(label="children", choices=[0, 1, 2, 3, 4, 5]),
    gr.Radio(["yes", "no",], label="smoker"),
    gr.Dropdown(label="region", choices=["southeast", "southwest", "northwest", "northeast"])
]
# App Launch
app = gr.Interface(
    fn=predict_insurance_charges,
    inputs=inputs,
    outputs="text",
    title="Insurance Charges  Predictor"
)

app.launch(share=True)

