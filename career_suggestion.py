import streamlit as st
import pickle
import pandas as pd

# -------------------------
# Load Model + Encoders
# -------------------------
with open("career_suggestion.pkl", "rb") as f:
    saved_objects = pickle.load(f)

model = saved_objects["model"]
label_encoder = saved_objects["label_encoder"]
feature_names = saved_objects["features"]

st.title("🎓 Career Path Suggestion System")

st.write("Fill in your details below to get the most suitable career suggestion:")

# -------------------------
# Input Features
# -------------------------
cgpa = st.slider("CGPA", 6.0, 10.0, 8.0, 0.1)
prog_skill = st.slider("Programming Skill (1-10)", 1, 10, 5)
math_skill = st.slider("Math Skill (1-10)", 1, 10, 5)
problem_solving = st.slider("Problem Solving (1-10)", 1, 10, 5)
comm_skill = st.slider("Communication Skill (1-10)", 1, 10, 5)
cybersecurity = st.slider("Cybersecurity Knowledge (1-10)", 1, 10, 5)
database = st.slider("Database Knowledge (1-10)", 1, 10, 5)
ai_ml = st.slider("AI/ML Knowledge (1-10)", 1, 10, 5)
networking = st.slider("Networking Skill (1-10)", 1, 10, 5)
creativity = st.slider("Creativity (1-10)", 1, 10, 5)
leadership = st.slider("Leadership (1-10)", 1, 10, 5)
mobile_dev = st.slider("Mobile Development Skill (1-10)", 1, 10, 5)
cloud_computing = st.slider("Cloud Computing Skill (1-10)", 1, 10, 5)
blockchain = st.slider("Blockchain Knowledge (1-10)", 1, 10, 5)
robotics = st.slider("Robotics Skill (1-10)", 1, 10, 5)
system_design = st.slider("System Design (1-10)", 1, 10, 5)
design = st.slider("Design Skill (1-10)", 1, 10, 5)

interest = st.selectbox(
    "Preferred Interest",
    ["Coding", "Analytics", "Research", "Networking", "Design", "Management", "Systems"]
)

# -------------------------
# Create Input DataFrame
# -------------------------
# Fill values for numeric features
input_dict = {
    "CGPA": cgpa,
    "Programming Skill (1-10)": prog_skill,
    "Math Skill (1-10)": math_skill,
    "Problem Solving (1-10)": problem_solving,
    "Communication Skill (1-10)": comm_skill,
    "Cybersecurity Knowledge (1-10)": cybersecurity,
    "Database Knowledge (1-10)": database,
    "AI/ML Knowledge (1-10)": ai_ml,
    "Networking Skill (1-10)": networking,
    "Creativity (1-10)": creativity,
    "Leadership (1-10)": leadership,
    "Mobile Dev Skill (1-10)": mobile_dev,
    "Cloud Computing Skill (1-10)": cloud_computing,
    "Blockchain Knowledge (1-10)": blockchain,
    "Robotics Skill (1-10)": robotics,
    "System Design (1-10)": system_design,
    "Design Skill (1-10)": design,
    "Preferred Interest": interest
}

input_df = pd.DataFrame([input_dict])

# One-hot encode Preferred Interest (must match training columns)
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=feature_names, fill_value=0)

# -------------------------
# Prediction
# -------------------------
if st.button("🔍 Suggest Career"):
    prediction = model.predict(input_df)[0]
    predicted_career = label_encoder.inverse_transform([prediction])[0]
    st.success(f"🎯 Suggested Career Path: **{predicted_career}**")
