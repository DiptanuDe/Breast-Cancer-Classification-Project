import streamlit as st
import pickle
import numpy as np
import pandas as pd

def load_model():
    with open('saved_steps.pkl','rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

RandomForest = data["model"]


def show_prediction_page():
    st.title("Breast Cancer Classification")
    
    st.write("""### Please provide the following information to predict the type of breast cancer""" )

    #binary_ans = ("Yes","No")

    #Operation_history = st.selectbox("Did you have any major operation previously?:",binary_ans)

    radius_mean = st.slider("Mean radius:",0.0,50.0,0.1)
    texture_mean = st.slider("Mean texture:",0.0,50.0,0.1)
    smoothness_mean = st.slider("Mean smoothness:",0.0,1.0,0.00001)
    compactness_mean = st.slider("Mean compactness:",0.0,1.0,0.00001)
    symmetry_mean = st.slider("Mean symmetry:",0.0,1.0,0.00001)
    fractal_dimension_mean = st.slider("Mean fractal dimension:",0.0,1.0,0.00001)
    texture_se = st.slider("texture standard error:",0.0,10.0,0.01)
    smoothness_se = st.slider("Smoothness standard error:",0.0,1.0,0.00001)
    symmetry_se = st.slider("Symmetry standard error:",0.0,1.0,0.00001)
    symmetry_worst = st.slider("Symmetry worst:",0.0,1.0,0.00001)

    predict = st.button("Identify the type of breast cancer")
    if predict:
        X = pd.DataFrame([[radius_mean,texture_mean,smoothness_mean,compactness_mean,symmetry_mean,fractal_dimension_mean,texture_se,
                          smoothness_se,symmetry_se,symmetry_worst]])
        Cancer_type = RandomForest.predict(X)
        
        if Cancer_type[0] == 'M':
            Cancer_type[0] = 'Malignant'
        elif Cancer_type[0] == 'B':
            Cancer_type[0] = 'Benign'
            
        st.subheader(f"The Cancer type is: {Cancer_type[0]}")