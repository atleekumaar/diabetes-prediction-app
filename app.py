import streamlit as st
import joblib
import numpy as np

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS (Dhakad Dark Theme UI)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes AI Diagnostic System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced CSS Styling
st.markdown("""
    <style>
    /* Dark Theme Backgrounds */
    .main {
        background-color: #0e1117;
    }
    
    /* Header Container */
    .header-container {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #374151;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        margin-bottom: 2rem;
        text-align: center;
    }
    .header-title {
        color: #60a5fa;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .header-subtitle {
        color: #9ca3af;
        font-size: 1.1rem;
    }

    /* Metric Cards */
    .metric-card {
        background: #1f2937;
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        margin-bottom: 1rem;
    }
    .metric-label {
        color: #9ca3af;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        color: #f3f4f6;
        font-size: 1.6rem;
        font-weight: 700;
    }

    /* Custom Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1d4ed8 0%, #1e40af 100%);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.6);
        transform: translateY(-2px);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }
    
    /* Hide Streamlit Default Menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Load Model & Scaler (With Caching for Speed)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load('diabetes_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"Error loading model files: {e}. Please ensure 'diabetes_model.pkl' and 'scaler.pkl' are in the repository root.")

# -----------------------------------------------------------------------------
# 3. Sidebar Features & Quick Guide
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/387/387561.png", width=100)
    st.title("🩺 AI Health Assistant")
    st.markdown("---")
    
    st.subheader("📌 Input Reference Ranges")
    st.info("""
    - **Glucose:** Normal < 140 mg/dL
    - **Blood Pressure:** Normal < 80 mmHg
    - **BMI:** Normal (18.5 - 24.9)
    - **Age:** 21 - 81 years
    """)
    
    st.markdown("---")
    st.markdown("### ⚙️ System Info")
    st.text("Model: Support Vector Machine")
    st.text("Preprocessing: StandardScaler")
    st.text("Accuracy: ~77.3%")
    
    st.markdown("---")
    st.caption("Developed with ❤️ using Streamlit & Scikit-Learn By Atul Shukla")

# -----------------------------------------------------------------------------
# 4. Main Page Header
# -----------------------------------------------------------------------------
st.markdown("""
    <div class="header-container">
        <div class="header-title">Diabetes Health Risk Predictor</div>
        <div class="header-subtitle">Advanced Machine Learning Diagnostics Powered by Support Vector Machines (SVM)</div>
    </div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. Interactive Form Inputs (Columns Layout)
# -----------------------------------------------------------------------------
st.subheader("📋 Enter Patient Medical Metrics")

col1, col2 = st.columns(2)

with col1:
    glucose = st.number_input('Glucose Level (mg/dL)', min_value=0, max_value=300, value=120, help="Plasma glucose concentration (2 hours in an oral glucose tolerance test)")
    blood_pressure = st.number_input('Blood Pressure (mmHg)', min_value=0, max_value=200, value=70, help="Diastolic blood pressure")
    skin_thickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=20, help="Triceps skin fold thickness")

with col2:
    insulin = st.number_input('Insulin Level (mu U/ml)', min_value=0, max_value=900, value=80, help="2-Hour serum insulin")
    bmi = st.number_input('Body Mass Index (BMI)', min_value=0.0, max_value=70.0, value=25.0, format="%.1f", help="Weight in kg/(height in m)^2")
    dpf = st.number_input('Diabetes Pedigree Function', min_value=0.000, max_value=3.000, value=0.500, format="%.3f", help="Diabetes pedigree function based on family history")
    age = st.number_input('Age (Years)', min_value=1, max_value=120, value=30)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. Prediction Engine & Output Dashboard
# -----------------------------------------------------------------------------
if st.button('🚀 Analyze Risk'):
    input_data = (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age)
    input_array = np.asarray(input_data).reshape(1, -1)
    
    # Preprocess features using loaded scaler
    std_data = scaler.transform(input_array)
    
    # Predict status
    prediction = model.predict(std_data)
    
    st.markdown("---")
    st.subheader("📊 Diagnostic Summary")
    
    if prediction[0] == 0:
        st.balloons()
        st.success("### 🎉 Result: The patient is NOT DIABETIC")
        st.markdown("""
        **Clinical Insights:**
        - The entered physiological metrics fall within low-risk thresholds.
        - Encourage maintaining a balanced diet, regular exercise, and standard health checkups.
        """)
    else:
        st.warning("### ⚠️ Result: The patient IS DIABETIC / HIGH RISK")
        st.markdown("""
        **Clinical Insights:**
        - Elevated metrics (such as high Glucose or BMI) indicate a high potential risk of diabetes.
        - **Recommendation:** Consult a certified medical practitioner for a formal HbA1c test and proper medical diagnosis.
        """)

    # Display Metrics Quick Summary Dashboard
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f'<div class="metric-card"><div class="metric-label">Glucose</div><div class="metric-value">{glucose} mg/dL</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-card"><div class="metric-label">BMI</div><div class="metric-value">{bmi}</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="metric-card"><div class="metric-label">Blood Pressure</div><div class="metric-value">{blood_pressure} mmHg</div></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="metric-card"><div class="metric-label">Age</div><div class="metric-value">{age} Yrs</div></div>', unsafe_allow_html=True)

    # Patient Report Download Feature
    report_text = f"""
    ========================================
         DIABETES AI DIAGNOSTIC REPORT
    ========================================
    Pregnancies: {pregnancies}
    Glucose Level: {glucose} mg/dL
    Blood Pressure: {blood_pressure} mmHg
    Skin Thickness: {skin_thickness} mm
    Insulin Level: {insulin} mu U/ml
    BMI: {bmi}
    Diabetes Pedigree Function: {dpf}
    Age: {age} years
    ----------------------------------------
    PREDICTION RESULT: {"DIABETIC" if prediction[0] == 1 else "NON-DIABETIC"}
    ========================================
    """
    st.download_button(
        label="📥 Download Diagnostic Report (.txt)",
        data=report_text,
        file_name=f"diabetes_report_patient_age_{age}.txt",
        mime="text/plain"
    )
