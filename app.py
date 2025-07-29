import streamlit as st
import pandas as pd
import joblib
import io

# Load model
model = joblib.load("ekg_model.pkl")

st.title("🩺 EKG Abnormality Detector (Batch Mode)")
st.markdown("Brahmleen Papneja - Queen's Univeristy Faculty of Health Sciences.")
st.markdown("Upload a CSV with one or more EKG signals (each row = 1 signal).")

uploaded_file = st.file_uploader("📂 Upload CSV", type="csv")

EXPECTED_LENGTH = 187

def preprocess_signals(df):
    cleaned = []
    for _, row in df.iterrows():
        row_array = row.values.flatten()
        if len(row_array) < EXPECTED_LENGTH:
            # Pad with zeros
            row_array = list(row_array) + [0] * (EXPECTED_LENGTH - len(row_array))
        elif len(row_array) > EXPECTED_LENGTH:
            # Trim to first 187
            row_array = row_array[:EXPECTED_LENGTH]
        cleaned.append(row_array)
    return pd.DataFrame(cleaned)

if uploaded_file is not None:
    try:
        raw = pd.read_csv(uploaded_file, header=None)
        st.success(f"Uploaded {raw.shape[0]} EKG signals.")

        # Preprocess to exactly 187 columns
        data = preprocess_signals(raw)

        # Make predictions and probabilities
        probabilities = model.predict_proba(data)
        predictions = model.predict(data)

        # Build result DataFrame
        results_df = pd.DataFrame({
            "Prediction": ["✅ Normal" if p == 0 else "⚠️ Abnormal" for p in predictions],
            "Prob_Normal (%)": (probabilities[:, 0] * 100).round(2),
            "Prob_Abnormal (%)": (probabilities[:, 1] * 100).round(2),
        })

        # Add signal index for search/filtering
        results_df.index.name = "Signal #"
        st.markdown("### 🔍 Results")
        search = st.text_input("Search by prediction (type 'normal' or 'abnormal')")

        if search:
            if "normal" in search.lower():
                filtered = results_df[results_df["Prediction"] == "✅ Normal"]
            elif "abnormal" in search.lower():
                filtered = results_df[results_df["Prediction"] == "⚠️ Abnormal"]
            else:
                filtered = results_df
            st.dataframe(filtered)
        else:
            st.dataframe(results_df)

        # Download button
        csv = results_df.to_csv().encode("utf-8")
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="ekg_predictions.csv",
            mime="text/csv",
        )

        # Preview first 5 signals
        st.markdown("### 📈 First 5 EKG Signal Charts")
        for i in range(min(5, data.shape[0])):
            st.line_chart(data.iloc[i])

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
