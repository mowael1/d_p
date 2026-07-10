import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="EuroSAT Classification",
    page_icon="🛰️",
    layout="centered"
)

st.title("🛰️ EuroSAT Land Cover Classification")

st.write("Upload a satellite image to classify it.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):

        with st.spinner("Predicting..."):

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }

            response = requests.post(API_URL, files=files)

        if response.status_code == 200:

            result = response.json()

            st.success("Prediction Completed ✅")

            st.subheader("Prediction")

            st.write(f"**Class:** {result['predicted_class']}")

            st.write(f"**Confidence:** {result['confidence']:.2%}")

        else:

            st.error("Prediction Failed")

            st.text(response.text)