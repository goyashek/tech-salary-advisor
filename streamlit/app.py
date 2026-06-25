import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page settings
st.set_page_config(
    page_title="Tech Salary Predictor (India)",
    page_icon="💼",
    layout="centered"
)

# Load model and metadata
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
    st.error(f"Error loading model assets. Run the Jupyter Notebook first. Error: {e}")
    st.stop()

st.title("Tech Salary Predictor (India) 💼")
st.write("Enter your career profile below to predict your expected salary in Indian Rupees (INR) and LPA.")
st.write("---")

# User inputs
job_title = st.selectbox("Job Title", metadata['job_titles'], index=4) # Default to Data Scientist
years_exp = st.slider("Years of Experience", min_value=0.0, max_value=20.0, value=3.0, step=0.5)
education = st.selectbox("Education Level", metadata['education_levels'])
location = st.selectbox("Location / City", metadata['locations'])
selected_skills = st.multiselect("Your Technical Skills", metadata['all_skills'], default=['Python', 'SQL'])

# Helper function to run model prediction using scikit-learn Pipeline
def predict_salary_inr(exp, title, edu, loc, skills):
    # Raw inputs expected by the pipeline preprocessor.
    input_data = {
        'Experience_Years': exp,
        'Job_Title': title,
        'Location': loc,
        'Education_Level': edu,
        'skill_count': len(skills),
        # The user fills every field, so nothing is missing at prediction time.
        'Experience_Years_missing': 0,
        'Education_Level_missing': 0,
        'Location_missing': 0,
        'Skills_missing': 0,
    }

    # Set the binary flags for all skills
    for skill in metadata['all_skills']:
        input_data[skill] = 1 if skill in skills else 0

    # Build a 1-row frame in the exact column order the model was trained on
    input_df = pd.DataFrame([input_data])[metadata['feature_columns']]

    # The pipeline handles preprocessing and inverts the log-target internally
    pred = model.predict(input_df)[0]
    return max(300000, pred)

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
    
    # Visual: Salary vs Experience Growth
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
    
    # Simple recommendation for learning next skills
    st.subheader("Estimated Salary Bumps for Next Skills")
    missing_skills = [s for s in metadata['all_skills'] if s not in selected_skills]
    
    if missing_skills:
        bumps = []
        for skill in missing_skills:
            test_skills = selected_skills + [skill]
            new_pred = predict_salary_inr(years_exp, job_title, education, location, test_skills)
            bump = new_pred - prediction
            if bump > 1000:
                bumps.append((skill, bump))
                
        # Sort bumps desc
        bumps = sorted(bumps, key=lambda x: x[1], reverse=True)[:3]
        
        for skill, bump in bumps:
            st.write(f"📖 **{skill}**: Learn this to add an estimated **₹ {int(bump):,}** (+{bump/100000:.2f} LPA) to your market value!")
    else:
        st.write("You possess all available skills in our list!")

st.write("")
st.write("---")
# Model metadata at bottom
st.info(
    f"**Model Information:**\n"
    f"- Algorithm: {metadata['model_name']} regressor pipeline\n"
    f"- Mean Absolute Error (MAE): ₹ {metadata['mae']:,.0f} (~{mae_lpa:.2f} LPA)\n"
    f"- R² Score: {metadata['r2']:.4f}\n"
    f"- Preprocessing: scaling + one-hot encoding via a scikit-learn Pipeline\n"
    f"- Trained on ~1.1 lakh tech salary records."
)
