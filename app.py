import streamlit as st
import joblib
import numpy as np

# Load model
st.write("Loading model...")
try:
    clf = joblib.load('klasifikasi_obesitas.pkl')
    scaler = joblib.load('scaler.sav')
    st.write("Model loaded.")
except Exception as e:
    st.write(f"Error loading model: {e}")

st.title('Obesity Level Classification')

# Input pengguna
height = st.number_input('Height (in meters)', min_value=0.5, max_value=2.5, value=1.75)
weight = st.number_input('Weight (in kg)', min_value=20, max_value=200, value=70)
gender = st.selectbox('Gender', ('Female', 'Male'))

# Konversi gender ke bentuk numerik
gender_num = 1 if gender == 'Female' else 0

if st.button('Predict'):
    st.write("Button clicked.")

    # Buat array input
    input_data = np.array([[height, weight, gender_num]])

    # Debug: Tampilkan data input
    st.write(f'Input data: {input_data}')

    try:
        # Lakukan prediksi
        prediction = clf.predict(input_data)
        # Debug: Tampilkan prediksi mentah
        st.write(f'Raw prediction: {prediction}')

        # Tampilkan hasil prediksi
        obesity_level = ['Normal', 'Overweight', 'Obese']
        st.write(f'The predicted obesity level is: {obesity_level[prediction[0]]}')
    except Exception as e:
        st.write(f'Error during prediction: {e}')
