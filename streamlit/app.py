import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Tech Salary Predictor (India)",
    page_icon="💼",
    layout="centered"
)

@st.cache_resource
def load_model_assets():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(current_dir, "models/salary_model.pkl"))
    metadata = joblib.load(os.path.join(current_dir, "models/metadata.pkl"))
    return model, metadata

try:
    model, metadata = load_model_assets()
    mae_lpa = metadata['mae'] / 100000
except Exception as e:
    st.error(f"Could not load the model assets. Run `python -m src.train` first. Error: {e}")
    st.stop()

st.title("Tech Salary Predictor (India) 💼")
st.write("Enter a career profile to get a salary estimate in Indian Rupees (INR) and LPA.")
st.caption(
    "This is a model estimate based on the project dataset, not an authoritative market benchmark."
)
st.write("---")

job_title = st.selectbox("Job Title", metadata['job_titles'], index=4) # Default to Data Scientist
years_exp = st.slider("Years of Experience", min_value=0.0, max_value=20.0, value=3.0, step=0.5)
education = st.selectbox("Education Level", metadata['education_levels'])
location = st.selectbox("Location / City", metadata['locations'])
selected_skills = st.multiselect("Your Technical Skills", metadata['all_skills'], default=['Python', 'SQL'])

def predict_salary_inr(exp, title, edu, loc, skills):
    input_data = {
        'Experience_Years': exp,
        'Job_Title': title,
        'Location': loc,
        'Education_Level': edu,
        'skill_count': len(skills),
        # The user fills every field, so nothing is missing at prediction time.
        'Job_Title_missing': 0,
        'Experience_Years_missing': 0,
        'Education_Level_missing': 0,
        'Location_missing': 0,
        'Skills_missing': 0,
    }

    for skill in metadata['all_skills']:
        input_data[skill] = 1 if skill in skills else 0

    input_df = pd.DataFrame([input_data])[metadata['feature_columns']]
    return float(model.predict(input_df)[0])

st.write("")

if st.button("Predict Salary"):
    prediction = predict_salary_inr(years_exp, job_title, education, location, selected_skills)
    lpa = prediction / 100000
    mae_lpa = metadata['mae'] / 100000
    
    st.success("### Prediction Results")
    st.metric(label="Predicted Salary (LPA)", value=f"₹ {lpa:.2f} LPA", help="Lakhs Per Annum")
    st.write(f"**Predicted Salary in INR**: ₹ {int(prediction):,}")
    st.write(f"**Expected Range (±MAE)**: ₹ {(prediction - metadata['mae']):,.0f} - ₹ {(prediction + metadata['mae']):,.0f} (₹ {lpa - mae_lpa:.2f} LPA - ₹ {lpa + mae_lpa:.2f} LPA)")
    
    st.write("---")
    
    st.subheader("Salary Growth by Experience")
    exp_range = np.arange(0, 21.0, 1.0)
    salary_growth = []
    for e in exp_range:
        pred_e = predict_salary_inr(e, job_title, education, location, selected_skills)
        salary_growth.append(pred_e / 100000) # Convert to LPA
        
    chart_data = pd.DataFrame({
        "Years of Experience": exp_range,
        "Salary (LPA)": salary_growth
    }).set_index("Years of Experience")
    
    st.line_chart(chart_data)
    
    st.subheader("Skill comparison")
    missing_skills = [s for s in metadata['all_skills'] if s not in selected_skills]
    
    if missing_skills:
        bumps = []
        for skill in missing_skills:
            test_skills = selected_skills + [skill]
            new_pred = predict_salary_inr(years_exp, job_title, education, location, test_skills)
            bump = new_pred - prediction
            if bump > 1000:
                bumps.append((skill, bump))
                
        bumps = sorted(bumps, key=lambda x: x[1], reverse=True)[:3]
        
        for skill, bump in bumps:
            st.write(
                f"📖 **{skill}**: adding this skill changes the model estimate by "
                f"**₹ {int(bump):,}** (+{bump/100000:.2f} LPA)."
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
    f"- Preprocessing: scaling + one-hot encoding via a scikit-learn Pipeline\n"
    f"- Dataset rows after dropping missing targets: {metadata['dataset_rows']:,}\n"
    f"- Training split: {metadata['training_rows']:,} rows"
)
