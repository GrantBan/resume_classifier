import time

import requests
import streamlit as st


st.title("Resume Classification System (BERT API)")

text_input = st.text_area("Please enter resume text:", height=300)

if st.button("Predict"):
    url = "http://127.0.0.1:8000/predict"
    start = time.time()

    try:
        response = requests.post(url, json={"text": text_input}, timeout=120)
        cost = (time.time() - start) * 1000

        if response.status_code == 200:
            result = response.json()
            st.success(f"Label: {result['label']}")
            st.info(f"Category: {result['category']}")
            st.info(f"Backend time: {result['time_ms']} ms")
            st.info(f"Total request time: {cost:.2f} ms")
        else:
            st.error(f"Request failed: {response.text}")

    except Exception as e:
        st.error(f"Request error: {str(e)}")
