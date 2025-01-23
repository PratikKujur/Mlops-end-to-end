import streamlit as st
from src.ModelPedict import ModelPredict
import numpy as np
import yaml
import pandas as pd



def predict(model_reg_path,model_clf_path,features):
    features=np.array(pd.DataFrame(features))
    predictions=ModelPredict(model_reg_path,model_clf_path,features)

    
    return {"quality": predictions[0], "type": "Red" if predictions[1]==1 else "White"}

if __name__ == "__main__":

    params_trainer=yaml.safe_load(open("/Users/pratik.kujur/Desktop/Projects/Mlops-end-to-end/params.yaml"))['train']


    # Page configuration
    st.set_page_config(page_title="Wine Quality and Type Multiout Prediction")
    st.header("🍷 Wine Quality and Type Multiout Prediction")

    # Features dictionary to store user inputs
    features = {
        "volatile_acidity": list(),
        "citric_acid": list(),
        "residual_sugar": list(),
        "chlorides": list(),
        "free_sulfur_dioxide": list(),
        "total_sulfur_dioxide": list(),
        "density": list(),
        "pH": list(),
        "sulphates": list(),
        "alcohol": list(),
    }

    
    with st.form(key="wine_quality_form"):
        
        volatile_acidity = st.number_input(
            label="Volatile Acidity", format="%.2f", value=0.0
        )
        features["volatile_acidity"].append(volatile_acidity)
        
        citric_acid= st.number_input(
            label="Citric Acid", format="%.2f", value=0.0
        )
        features["citric_acid"].append(citric_acid)
        
        residual_sugar = st.number_input(
            label="Residual Sugar", format="%.2f", value=0.0
        )
        features["residual_sugar"].append(residual_sugar)
        
        chlorides = st.number_input(
            label="Chlorides", format="%.2f", value=0.0
        )
        features["chlorides"].append(chlorides)
         
        free_sulfur_dioxide= st.number_input(
            label="Free Sulfur Dioxide", format="%.2f", value=0.0
        )
        features["free_sulfur_dioxide"].append(free_sulfur_dioxide)
        
        total_sulfur_dioxide= st.number_input(
            label="Total Sulfur Dioxide", format="%.2f", value=0.0
        )
        features["total_sulfur_dioxide"].append(total_sulfur_dioxide)
        
        density = st.number_input(
            label="Density", format="%.6f", value=0.0
        )
        features["density"].append(density)
        
        pH = st.number_input(
            label="pH", format="%.2f", value=0.0
        )
        features["pH"].append(pH)
        
        sulphates = st.number_input(
            label="Sulphates", format="%.2f", value=0.0
        )
        features["sulphates"].append(sulphates)
        
        alcohol = st.number_input(
            label="Alcohol", format="%.2f", value=0.0
        )
        features["alcohol"].append(alcohol)

  
        submit = st.form_submit_button(label="Predict")

   
    if submit:
        prediction = predict(params_trainer['model_1_reg'],params_trainer['model_1_clf'],features)
        st.success(f"Prediction: Quality - {prediction['quality']}, Type - {prediction['type']}")
