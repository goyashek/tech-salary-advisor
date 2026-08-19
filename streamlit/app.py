import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference import (
    load_model_assets,
    predict_salary_inr,
    predict_salary_interval,
)
from src.validate_data import MAX_EXPERIENCE_YEARS

st.set_page_config(
    page_title="Tech Salary Predictor (India)", page_icon="💼", layout="centered"
)

try:
    assets = load_model_assets()
    _, metadata = assets
    mae_lpa = metadata["mae"] / 100000
except Exception as e:
    st.error(
        f"Could not load the model assets. Run `python -m src.train` first. Error: {e}"
    )
    st.stop()

st.title("Tech Salary Predictor (India) 💼")
st.write(
    "Enter a career profile to get a salary estimate in Indian Rupees (INR) and LPA."
)
st.caption(
    "This is a model estimate based on the project dataset, not an authoritative market benchmark."
)
st.write("---")

job_title = st.selectbox(
    "Job Title", metadata["job_titles"], index=4
)  # Default to Data Scientist
years_exp = st.slider(
    "Years of Experience",
    min_value=0.0,
    max_value=float(MAX_EXPERIENCE_YEARS),
    value=3.0,
    step=0.5,
)
education = st.selectbox("Education Level", metadata["education_levels"])
location = st.selectbox("Location / City", metadata["locations"])
selected_skills = st.multiselect(
    "Your Technical Skills", metadata["all_skills"], default=["Python", "SQL"]
)

st.write("")

if st.button("Predict Salary"):
    result = predict_salary_interval(
        job_title, years_exp, education, location, selected_skills, assets
    )
    prediction = result["salary_inr"]
    lpa = prediction / 100000

    st.success("### Prediction Results")
    st.metric(
        label="Predicted Salary (LPA)", value=f"₹ {lpa:.2f} LPA", help="Lakhs Per Annum"
    )
    st.write(f"**Predicted Salary in INR**: ₹ {int(prediction):,}")
    st.write(
        f"**{result['level']:.0%} calibrated prediction interval**: "
        f"₹ {result['lower_inr']:,.0f} - ₹ {result['upper_inr']:,.0f} "
        f"(₹ {result['lower_inr'] / 100000:.2f} LPA - "
        f"₹ {result['upper_inr'] / 100000:.2f} LPA)"
    )

    st.write("---")

    st.subheader("Salary Growth by Experience")
    exp_range = np.arange(0, 21.0, 1.0)
    salary_growth = []
    for e in exp_range:
        pred_e = predict_salary_inr(
            job_title, e, education, location, selected_skills, assets
        )
        salary_growth.append(pred_e / 100000)  # Convert to LPA

    chart_data = pd.DataFrame(
        {"Years of Experience": exp_range, "Salary (LPA)": salary_growth}
    ).set_index("Years of Experience")

    st.line_chart(chart_data)

    st.subheader("Skill comparison")
    missing_skills = [s for s in metadata["all_skills"] if s not in selected_skills]

    if missing_skills:
        bumps = []
        for skill in missing_skills:
            test_skills = selected_skills + [skill]
            new_pred = predict_salary_inr(
                job_title, years_exp, education, location, test_skills, assets
            )
            bump = new_pred - prediction
            if bump > 1000:
                bumps.append((skill, bump))

        bumps = sorted(bumps, key=lambda x: x[1], reverse=True)[:3]

        for skill, bump in bumps:
            st.write(
                f"📖 **{skill}**: adding this skill changes the model estimate by "
                f"**₹ {int(bump):,}** (+{bump / 100000:.2f} LPA)."
            )
        if not bumps:
            st.write("No available skill increased this estimate by more than ₹1,000.")
    else:
        st.write("All skills available in the model are already selected.")

st.write("")
st.write("---")
st.info(
    f"**Model Information:**\n"
    f"- Algorithm: {metadata['model_name']} regressor pipeline\n"
    f"- Mean Absolute Error (MAE): ₹ {metadata['mae']:,.0f} (~{mae_lpa:.2f} LPA)\n"
    f"- R² Score: {metadata['r2']:.4f}\n"
    f"- Prediction interval: {metadata['prediction_interval']['level']:.0%} "
    f"split-conformal interval\n"
    f"- Preprocessing: scaling + one-hot encoding via a scikit-learn Pipeline\n"
    f"- Dataset rows after dropping missing targets: {metadata['dataset_rows']:,}\n"
    f"- Training split: {metadata['training_rows']:,} rows"
)
