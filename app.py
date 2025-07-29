import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("ekg_model.pkl")

st.title("🩺 PulseWave Multi-EKG Abnormality Detector")
st.markdown("Brahmleen Papneja - Queen's Univeristy Faculty of Health Sciences.")
st.markdown("Upload a CSV file with **one or more EKG signals** (187 columns per row).")

uploaded_file = st.file_uploader("Choose your .csv file", type="csv")

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file, header=None)

        if data.shape[1] != 187:
            st.error(f"CSV must have exactly 187 columns (your file has {data.shape[1]}).")
            st.stop()

        st.success(f"Uploaded {data.shape[0]} EKG signals.")

        # Predict probabilities and labels
        probabilities = model.predict_proba(data)
        predictions = model.predict(data)

        # Create a results DataFrame
        results_df = pd.DataFrame({
            "Prediction": ["✅ Normal" if pred == 0 else "⚠️ Abnormal" for pred in predictions],
            "Prob_Normal (%)": (probabilities[:, 0] * 100).round(2),
            "Prob_Abnormal (%)": (probabilities[:, 1] * 100).round(2),
        })

        # Display results
        st.dataframe(results_df)

        # Show chart of all EKGs (first 5 only for simplicity)
        st.markdown("### Preview of EKG Signals (first 5):")
        for i in range(min(5, data.shape[0])):
            st.line_chart(data.iloc[i])

    except Exception as e:
        st.error(f"Error: {e}")
