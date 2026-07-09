import gradio as gr
import numpy as np
import pandas as pd 
import pickle

# 
with open("insurance_prediction_model.pkl", "rb") as file:
    model = pickle.load(file)

# Logic 
def predict_insurance_charges(
    gender, age, address, famsize,
    Pstatus, M_Edu, F_Edu, M_Job, F_Job,
    relationship, smoker, tuition_fee, time_friends, ssc_result):
  
    # columns into Dataframe
    input_df = pd.DataFrame([[
        gender, age, address, famsize,
        Pstatus, M_Edu, F_Edu, M_Job, F_Job,
        relationship, smoker, tuition_fee, time_friends, ssc_result
    ]],
    columns=[
        'gender', 'age', 'address', 'famsize',
        'Pstatus', 'M_Edu', 'F_Edu', 'M_Job', 'F_Job',
        'relationship', 'smoker', 'tuition_fee', 'time_friends', 'ssc_result'
    ])
  
    # Predicton
    prediction = model.predict(input_df)[0]
    
    # np.clip
    final_gpa = np.clip(prediction, 0, 5)
  
    return f"Predicted HSC Result: {final_gpa:.2f}"


# App Interface 
inputs = [
    gr.Radio(["M", "F"], label="gender"), 
    gr.Number(label="age", value=18),
    gr.Radio(["Urban", "Rural"], label="Address"),
    gr.Radio(["GT3", "LE3"], label="Family Size"),
    gr.Radio(["Together", "Apart"], label="Parent Status"),
    gr.Slider(0, 4, step=1, label="Mother's Edu"),
    gr.Slider(0, 4, step=1, label="Father's Edu"),
    gr.Dropdown(["At_home", "Health", "Other", "Services", "Teacher"], label="Mother's Job"),
    gr.Dropdown(["Teacher", "Other", "Services", "Health", "Business", "Farmer"], label="Father's Job"),
    gr.Radio(["Yes", "No"], label="Relationship"),
    gr.Radio(["Yes", "No"], label="Smoker"),
    gr.Number(label="Tuition Fee"),
    gr.Slider(1, 5, step=1, label="Time with Friends"),
    gr.Number(label="SSC Result (GPA)")
] 

# App Launch
app = gr.Interface(
    fn=predict_insurance_charges,
    inputs=inputs,
    outputs="text",
    title="Insurance Charges  Predictor"
)

app.launch(share=True)

