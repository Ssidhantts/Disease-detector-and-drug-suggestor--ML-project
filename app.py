import json, re
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Disease Detector & Drug Suggestor", page_icon="🩺")

def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()

@st.cache_resource
def load_all():
    bundle = joblib.load("model.pkl")
    drug_db = json.load(open("drug_db.json"))
    metrics = json.load(open("metrics.json"))
    return bundle["model"], bundle["symptoms"], drug_db, metrics

model, symptoms, drug_db, metrics = load_all()

st.title("🩺 Disease Detector & Drug Suggestor")
st.warning("This is a demo app. It is not a substitute for professional medical advice. Please consult a doctor for any health concerns.")

# Sidebar: model ki accuracy dikhao
with st.sidebar:
    st.header("Model accuracy")
    for name, acc in metrics.items():
        st.write(f"{name}: {acc}%")
    st.caption("The dataset is synthetic, so the accuracy is not representative of real-world performance.")

# Symptoms select karo (underscore hatake readable naam dikhao)
pretty = {s: s.replace("_", " ").title() for s in symptoms}
chosen = st.multiselect(
    "Select your symptoms (please choose at least 3)",

    options=symptoms,
    format_func=lambda s: pretty[s],
)

if st.button("Detect karo"):
    if len(chosen) < 3:
        st.error("Please choose at least 3 symptoms, otherwise the result will be unreliable.")
    else:
        x = pd.DataFrame([[1 if s in chosen else 0 for s in symptoms]], columns=symptoms)
        probs = model.predict_proba(x)[0]
        top = np.argsort(probs)[::-1][:3]

        for rank, i in enumerate(top, start=1):
            disease = re.sub(r"\s+", " ", model.classes_[i]).strip()
            conf = probs[i] * 100
            with st.expander(f"{rank}. {disease} ({conf:.1f}%)", expanded=(rank == 1)):
                drugs = drug_db.get(norm(model.classes_[i]))
                if drugs:
                    st.write("**Suggested medications:**")
                    for d in drugs:
                        st.write(f"- {d}")
                else:
                    st.write("No medication data available for this disease. Please consult a doctor.")