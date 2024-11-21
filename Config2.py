import streamlit as st
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Simulate random risk assessment CSV data
risk_data = pd.DataFrame({
    'Risk_Level': random.choices(['High Risk', 'Moderate Risk', 'Low Risk'], k=100)
})

# Simulate a simple model prediction
def get_random_diagnosis():
    return random.choice(["Malignant", "Benign"])

# Simulate a random risk assessment based on input
def assess_risk(radius_mean, perimeter_mean, concave_points_mean):
    # Simple logic for risk level based on input features
    risk_score = 0
    if radius_mean > 20:
        risk_score += 1
    if perimeter_mean > 100:
        risk_score += 1
    if concave_points_mean > 0.1:
        risk_score += 1
    
    if risk_score == 3:
        return "High Risk"
    elif risk_score == 2:
        return "Moderate Risk"
    else:
        return "Low Risk"

# Display data insights
st.title("Breast Cancer Awareness and Prediction")

# Sidebar for navigation and additional information
sidebar_options = st.sidebar.radio("Select a Section:", ['Prediction', 'Information', 'Risk Assessment'])

# Sidebar with metrics and awareness
st.sidebar.header("Breast Cancer Awareness")
st.sidebar.write(""" 
    Breast cancer is one of the most common cancers worldwide. Early detection through screening and awareness of risk factors can improve outcomes. 
    **Risk Factors for Breast Cancer**:
    - **Age**: Women over the age of 50 are at increased risk.
    - **Family history of breast cancer**: A close family member with breast cancer increases your risk.
    - **Inherited genes** (e.g., BRCA1, BRCA2): Inherited genetic mutations can increase risk.
    - **Hormone replacement therapy usage**: Long-term use may raise the risk.
    - **Lifestyle factors**: Diet, physical activity, alcohol consumption can all impact risk.
    
    **Important Metrics**:
    - **Radius Mean**: The average radius of the tumor. Larger radii often correlate with malignancy.
    - **Perimeter Mean**: The average perimeter of the tumor. Larger perimeters often indicate more aggressive cancers.
    - **Concave Points Mean**: The average number of concave points in the tumor's boundary. More concave points often indicate malignancy.
""")
st.sidebar.write("For more information, consult with your healthcare provider.")
st.sidebar.write("---")

# Extra Info Section
if sidebar_options == 'Information':
    st.subheader("More Information on Breast Cancer")
    st.write("""
    Breast cancer is a disease where cells in the breast grow out of control. In the early stages, symptoms may be subtle or absent, so regular screening is important for early detection. 
    Here are some common symptoms to look out for:
    - A lump in the breast or underarm
    - Change in the size, shape, or appearance of the breast
    - Unexplained pain in the breast or nipple
    - Nipple discharge, other than breast milk
    
    Screening methods for early detection include:
    - **Mammograms**: A type of X-ray that helps detect tumors.
    - **Ultrasound**: Uses sound waves to create an image of the breast tissue.
    - **MRI**: Magnetic resonance imaging for further analysis.
    
    Early detection significantly improves treatment outcomes. Please discuss with your doctor about your personal risk factors and suitable screening options.
    """)

# Prediction Tab
if sidebar_options == 'Prediction':
    st.subheader("Breast Cancer Diagnosis Prediction")

    # User input fields
    radius_mean = st.number_input("Radius Mean (mm)", min_value=0.0, max_value=50.0, value=15.0, step=0.1)
    perimeter_mean = st.number_input("Perimeter Mean (mm)", min_value=0.0, max_value=200.0, value=80.0, step=0.1)
    concave_points_mean = st.number_input("Concave Points Mean", min_value=0.0, max_value=1.0, value=0.05, step=0.01)

    if st.button("Analyze Risk and Predict Diagnosis"):
        # Generate random diagnosis
        diagnosis = get_random_diagnosis()
        diagnosis_label = "Malignant" if diagnosis == "Malignant" else "Benign"

        # Assess risk based on user inputs
        risk_level = assess_risk(radius_mean, perimeter_mean, concave_points_mean)

        # Display the results
        st.success(f"Predicted Diagnosis: {diagnosis_label}")
        st.success(f"Risk Level: {risk_level}")

# Risk Assessment Section
if sidebar_options == 'Risk Assessment':
    st.subheader("Risk Assessment Data Summary")
    risk_counts = risk_data['Risk_Level'].value_counts()
    st.write("Risk Level Distribution:")
    st.bar_chart(risk_counts)

    # Visualize risk distribution
    st.subheader("Risk Level Visualization")
    fig, ax = plt.subplots()
    sns.countplot(data=risk_data, x='Risk_Level', palette='viridis', ax=ax)
    st.pyplot(fig)

    # Explore risk assessment CSV data
    st.subheader("Explore Risk Assessment CSV")
    st.write(risk_data)
