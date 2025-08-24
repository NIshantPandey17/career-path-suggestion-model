import streamlit as st
import pickle
import pandas as pd

import os

# Set page configuration
st.set_page_config(
    page_title="Career Path Suggestion System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(15px);
    }
    
    /* Custom header styling */
    .main-header {
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Custom card styling */
    .custom-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Results styling */
    .result-card {
        background: linear-gradient(135deg, #f7fafc, #edf2f7);
        border: 2px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .career-name {
        font-weight: bold;
        font-size: 1.2rem;
        color: #2d3748;
    }
    
    .career-match {
        background: #667eea;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
    }
    
    /* Sidebar content styling */
    .sidebar-content {
        padding: 1rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Success message styling */
    .success-banner {
        background: linear-gradient(135deg, #48bb78, #38a169);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Feature item styling */
    .feature-item {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin: 0.5rem 0;
    }
    
    /* Hide default streamlit styling */
    .css-1rs6os, .css-17ziqus {
        visibility: hidden;
    }
    
    /* Custom slider styling */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #e2e8f0 0%, #667eea 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'predicted_careers' not in st.session_state:
    st.session_state.predicted_careers = []

# Load Model + Encoders
@st.cache_data
def load_model():
    try:
        if os.path.exists("career_suggestion.pkl"):
            with open("career_suggestion.pkl", "rb") as f:
                saved_objects = pickle.load(f)
            return saved_objects["model"], saved_objects["label_encoder"], saved_objects["features"]
        else:
            st.error("Model file 'career_suggestion.pkl' not found. Please ensure the file is in the correct directory.")
            return None, None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

model, label_encoder, feature_names = load_model()

# Sidebar Information Hub
with st.sidebar:
    st.markdown("# 📚 Information Hub")
    
    info_tab = st.radio("", ["🤖 How It Works", "✨ Features", "📋 Usage Guide"])
    
    if info_tab == "🤖 How It Works":
        st.markdown("""
        <div class="sidebar-content">
        <h4>AI Career Suggestion Process</h4>
        <p>Our advanced machine learning model analyzes your profile:</p>
        <ol>
        <li><strong>Data Collection:</strong> Gather technical skills, soft skills, and interests</li>
        <li><strong>Feature Processing:</strong> Apply feature engineering and encoding</li>
        <li><strong>ML Prediction:</strong> Calculate probability scores using trained algorithms</li>
        <li><strong>Results Ranking:</strong> Present top 3 careers with match percentages</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    elif info_tab == "✨ Features":
        st.markdown("""
        <div class="sidebar-content">
        <h4>System Capabilities</h4>
        </div>
        """, unsafe_allow_html=True)
        
        features_data = [
            ("🎯", "AI-Powered Analysis", "Advanced ML algorithms trained on career patterns"),
            ("📊", "Comprehensive Assessment", "Evaluates 17+ skills and attributes"),
            ("🚀", "Real-time Results", "Instant suggestions with confidence percentages"),
            ("🎨", "Intuitive Interface", "Beautiful design for enjoyable assessment")
        ]
        
        for icon, title, desc in features_data:
            st.markdown(f"""
            <div class="feature-item">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>
            <h4 style="margin-bottom: 0.5rem; color: #2d3748;">{title}</h4>
            <p style="font-size: 0.9rem; color: #4a5568; margin: 0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    elif info_tab == "📋 Usage Guide":
        st.markdown("""
        <div class="sidebar-content">
        <h4>Step-by-Step Guide</h4>
        <ol>
        <li><strong>Rate Your Skills:</strong> Use sliders honestly (1-10 scale)</li>
        <li><strong>Set CGPA:</strong> Adjust academic performance (6.0-10.0)</li>
        <li><strong>Choose Interest:</strong> Select primary area from dropdown</li>
        <li><strong>Get Suggestions:</strong> Click the suggestion button</li>
        <li><strong>Review Results:</strong> Analyze top 3 career matches</li>
        </ol>
        <p><strong>💡 Tip:</strong> Be realistic about current skills for best results.</p>
        </div>
        """, unsafe_allow_html=True)

# Main Content Area
st.markdown("""
<div class="main-header">
<h1>🎓 Career Path Suggestion System</h1>
<h5 style="
  display: inline-block;
  padding: 8px 15px;
  border-radius: 8px;
  font-weight: bold;
  color: #000080;
  background: linear-gradient(90deg, #ffd700, #ff8c00, #ffd700);
  background-size: 200% 100%;
  animation: shine 3s linear infinite;
">
  Only For Computer Science Student
</h5>

<style>
@keyframes shine {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}
</style>
<p>Discover your perfect career path with AI-powered recommendations</p>
</div>
""", unsafe_allow_html=True)

# Check if model is loaded
if model is None:
    st.error("Unable to load the machine learning model. Please check if 'career_suggestion.pkl' exists.")
    st.stop()

# Create two columns for the form
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("## 📊 Your Skills & Preferences")
    
    cgpa = st.slider("CGPA", 6.0, 10.0, 8.0, 0.1, help="Your current CGPA on a 10.0 scale")
    prog_skill = st.slider("Programming Skill (1-10)", 1, 10, 5, help="Rate your programming abilities")
    math_skill = st.slider("Math Skill (1-10)", 1, 10, 5, help="Rate your mathematical abilities")
    problem_solving = st.slider("Problem Solving (1-10)", 1, 10, 5, help="Rate your problem-solving skills")
    comm_skill = st.slider("Communication Skill (1-10)", 1, 10, 5, help="Rate your communication abilities")
    leadership = st.slider("Leadership (1-10)", 1, 10, 5, help="Rate your leadership skills")
    cybersecurity = st.slider("Cybersecurity Knowledge (1-10)", 1, 10, 5, help="Rate your cybersecurity knowledge")
    database = st.slider("Database Knowledge (1-10)", 1, 10, 5, help="Rate your database management skills")
    ai_ml = st.slider("AI/ML Knowledge (1-10)", 1, 10, 5, help="Rate your AI/Machine Learning knowledge")
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("## 🔧 Additional Skills")
    
    networking = st.slider("Networking Skill (1-10)", 1, 10, 5, help="Rate your networking abilities")
    creativity = st.slider("Creativity (1-10)", 1, 10, 5, help="Rate your creative thinking")
    mobile_dev = st.slider("Mobile Development Skill (1-10)", 1, 10, 5, help="Rate your mobile development skills")
    cloud_computing = st.slider("Cloud Computing Skill (1-10)", 1, 10, 5, help="Rate your cloud computing knowledge")
    blockchain = st.slider("Blockchain Knowledge (1-10)", 1, 10, 5, help="Rate your blockchain knowledge")
    robotics = st.slider("Robotics Skill (1-10)", 1, 10, 5, help="Rate your robotics knowledge")
    system_design = st.slider("System Design (1-10)", 1, 10, 5, help="Rate your system design skills")
    design = st.slider("Design Skill (1-10)", 1, 10, 5, help="Rate your design abilities")
    
    interest = st.selectbox(
        "Preferred Interest",
        ["Coding", "Analytics", "Research", "Networking", "Design", "Management", "Systems"],
        help="Select your primary area of interest"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

# Predict button
st.markdown('<div style="text-align: center; margin: 2rem 0;">', unsafe_allow_html=True)
if st.button("🔍 Get My Career Suggestions", key="predict_btn"):
    # Prepare input data
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
    
    try:
        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])
        
        # One-hot encode the "Preferred Interest" column
        input_df = pd.get_dummies(input_df, columns=["Preferred Interest"])
        
        # Ensure all expected columns exist (missing ones → 0)
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder columns to match model's training
        input_df = input_df[feature_names]
        
        # Make prediction
        probs = model.predict_proba(input_df)[0]
        top_indices = probs.argsort()[-3:][::-1]
        
        # Store results in session state
        st.session_state.predicted_careers = [
                (label_encoder.inverse_transform([i])[0], probs[i] * 100)
                for i in top_indices
            ]
        
        st.session_state.prediction_made = True
        st.session_state.input_dict = input_dict

        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

st.markdown('</div>', unsafe_allow_html=True)



# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'predicted_careers' not in st.session_state:
    st.session_state.predicted_careers = []
if 'clear_clicked' not in st.session_state:
    st.session_state.clear_clicked = False

    

# Display results
if st.session_state.prediction_made and st.session_state.predicted_careers:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("""
    <div class="success-banner">
    <h2>🎯 Your Career Recommendations</h2>
    <p>Based on your skills and preferences, here are your top career matches:</p>
    </div>
    """, unsafe_allow_html=True)
    
    for i, (career, match_percentage) in enumerate(st.session_state.predicted_careers):
        st.markdown(f"""
        <div class="result-card">
        <div class="career-name">{i+1}. {career}</div>
        <div class="career-match">{match_percentage:.1f}% match</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional insights
    st.markdown("---")
    st.markdown("### 📈 What this means:")
    best_career = st.session_state.predicted_careers[0][0]
    st.write(f"Your top recommendation is **{best_career}** with a {st.session_state.predicted_careers[0][1]:.1f}% compatibility match.")
    st.write("Consider exploring these career paths further by researching job requirements, salary expectations, and growth opportunities in your area.")
    
    # Save prediction to session state for potential future use
    st.session_state["predicted_career"] = best_career

    
    st.session_state.clear_clicked = True
    
    # If clear was clicked, reset relevant states immediately
    if st.session_state.clear_clicked:
        st.session_state.prediction_made = False
        st.session_state.predicted_careers = []
        if 'input_dict' in st.session_state:
            del st.session_state['input_dict']
        st.session_state.clear_clicked = False
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: white; padding: 1rem;">
<p>🚀 Powered by Machine Learning | Built with ❤️ Streamlit</p>
<p><em>Make informed career decisions based on your unique skill profile</em></p>
</div>
<!-- Bottom right corner name -->
<div style="
    position: fixed;
    bottom: 10px;
    right: 15px;
    color: rgba(255, 255, 255, 0.9);
    font-size: 1rem;
    z-index: 1000;
    padding: 5px 10px;
    border-radius: 8px;
    background: linear-gradient(270deg, #ff5f6d, #ffc371, #5ee7df, #9f5fff);
    background-size: 800% 800%;
    animation: gradientBG 10s ease infinite;
">
    Rhythm forever ❤️
</div>

<style>
@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
</style>
""", unsafe_allow_html=True)

# Add some floating animation elements (CSS only)
st.markdown("""
<style>
/* Full page animated background */
.stApp {
    height: 100%;
    background: linear-gradient(-45deg, #667eea, #764ba2, #ff5f6d, #ffc371);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
}

/* Gradient animation keyframes */
@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Optional: maintain existing blur/frosted sidebar */
.css-1d391kg {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(15px);
}

/* Floating animations (optional) */
@keyframes float {
    0%, 100% { transform: translateY(0px) rotate(0deg); }
    50% { transform: translateY(-20px) rotate(180deg); }
}

.stApp::before {
    content: '';
    position: fixed;
    top: 20%;
    right: 20%;
    width: 80px;
    height: 80px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 50%;
    animation: float 6s ease-in-out infinite;
    pointer-events: none;
    z-index: -1;
}

.stApp::after {
    content: '';
    position: fixed;
    bottom: 20%;
    right: 30%;
    width: 60px;
    height: 60px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 50%;
    animation: float 6s ease-in-out infinite 4s;
    pointer-events: none;
    z-index: -1;
}
</style>
""", unsafe_allow_html=True)
=======

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
 
