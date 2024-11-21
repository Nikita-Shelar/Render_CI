import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load the model and data
model = joblib.load('model/decision_tree_model.pkl')  # Replace with your model's file name
data = pd.read_csv('data/breast_cancer_data.csv')  # Load breast cancer data

# Load risk assessment data
risk_data = pd.read_csv('data/risk_assessment.csv')

# Set thresholds for risk assessment
threshold_concave_points = 0.1471  # Example median
threshold_radius_mean = 17.99  # Example median
threshold_perimeter_mean = 122.8  # Example median

# Risk assessment function
def assess_risk(input_data):
    risk_score = 0
    if input_data['concave_points_mean'] > threshold_concave_points:
        risk_score += 1
    if input_data['radius_mean'] > threshold_radius_mean:
        risk_score += 1
    if input_data['perimeter_mean'] > threshold_perimeter_mean:
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
sidebar_options = st.sidebar.radio("Select a Section:", ['Prediction'])

# Sidebar with metrics and awareness
st.sidebar.header("Breast Cancer Awareness")
st.sidebar.write("""
    Breast cancer is one of the most common cancers worldwide. Early detection through screening and awareness of risk factors can improve outcomes. 
    **Risk Factors for Breast Cancer**:
    - Age
    - Family history of breast cancer
    - Inherited genes (e.g., BRCA1, BRCA2)
    - Hormone replacement therapy usage
    - Lifestyle factors (e.g., diet, physical activity, alcohol consumption)

    **Important Metrics**:
    - **Radius Mean**: The average radius of the tumor.
    - **Perimeter Mean**: The average perimeter of the tumor.
    - **Concave Points Mean**: The average number of concave points in the tumor's boundary.
""")
st.sidebar.write("For more information, consult with your healthcare provider.")

# Prediction Tab
if sidebar_options == 'Prediction':
    st.subheader("Breast Cancer Diagnosis Prediction")

    # Select feature columns used in the model (6-7 most important features)
    feature_columns = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
        'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean'
    ]

    # Placeholder for user input for diagnosis prediction
    user_input = {}
    
    # Input fields with specified ranges
    user_input['radius_mean'] = st.number_input(
        "Radius Mean (3mm - 5mm)", min_value=0.0, max_value=100.0, value=3.5, step=0.1,
        help="Enter the radius of the tumor. Typically, ranges from 3mm to 5mm for a benign tumor."
    )
    user_input['perimeter_mean'] = st.number_input(
        "Perimeter Mean (30mm - 70mm)", min_value=0.0, max_value=200.0, value=50.0, step=0.1,
        help="Enter the perimeter of the tumor. Typical benign tumors range from 30mm to 70mm."
    )
    user_input['concave_points_mean'] = st.number_input(
        "Concave Points Mean (0.01 - 0.10)", min_value=0.0, max_value=1.0, value=0.03, step=0.01,
        help="Concave points mean. Typically benign tumors have fewer concave points."
    )
    
    # Ensure all required features are present
    for col in feature_columns:
        if col not in user_input:
            user_input[col] = 0.0  # Set default values for any missing features

    # Convert user inputs into a DataFrame row
    input_data = pd.DataFrame([user_input])

    if st.button("Predict Diagnosis"):
        # Make prediction
        diagnosis = model.predict(input_data)[0]
        diagnosis_label = "Malignant" if diagnosis == 1 else "Benign"
        st.success(f"The predicted diagnosis is: {diagnosis_label}")

        # Display additional insights about predictions
        st.subheader("Feature Importance")
        feature_importances = pd.DataFrame(model.feature_importances_, index=feature_columns, columns=['Importance'])
        st.bar_chart(feature_importances.sort_values('Importance', ascending=False).head(10))

        st.subheader("Confusion Matrix")
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(data['target'], model.predict(data[feature_columns]))
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
        st.pyplot(fig)

    # Risk Assessment Section
    st.subheader("Tumor Risk Assessment")

    # User input for new risk assessment with specified ranges
    radius_mean = st.number_input("Radius Mean (3mm - 5mm)", min_value=0.0, max_value=100.0, value=3.5, step=0.1)
    perimeter_mean = st.number_input("Perimeter Mean (30mm - 70mm)", min_value=0.0, max_value=200.0, value=50.0, step=0.1)
    concave_points_mean = st.number_input("Concave Points Mean (0.01 - 0.10)", min_value=0.0, max_value=1.0, value=0.03, step=0.01)

    if st.button("Analyze Risk"):
        # Assess risk
        risk_input = {
            'radius_mean': radius_mean,
            'perimeter_mean': perimeter_mean,
            'concave_points_mean': concave_points_mean
        }
        risk_level = assess_risk(risk_input)

        # Predict diagnosis
        features = [radius_mean, perimeter_mean, concave_points_mean]  # Adjust to match model input
        diagnosis = model.predict([features])[0]
        diagnosis_label = "Malignant" if diagnosis == 1 else "Benign"

        st.success(f"Risk Level: {risk_level}")
        st.success(f"Diagnosis Prediction: {diagnosis_label}")

# Display risk assessment CSV summary
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
