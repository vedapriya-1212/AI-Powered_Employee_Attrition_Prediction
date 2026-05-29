import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(
    page_title="Employee Attrition Dashboard",
    page_icon="📊",
    layout="wide"
)

# --------------------------
# LOAD FILES
# --------------------------

model = joblib.load("attrition_model.pkl")
feature_cols = joblib.load("feature_columns.pkl")

# --------------------------
# CUSTOM CSS
# --------------------------

st.markdown("""
<style>
.main {
    background-color: #0E1117;
}

.metric-card {
    background-color: #1E293B;
    padding:20px;
    border-radius:15px;
    text-align:center;
}

.big-font {
    font-size:40px;
    font-weight:bold;
}

.small-font {
    font-size:18px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------
# TITLE
# --------------------------

st.title("📊 Employee Attrition Analytics Dashboard")
st.markdown("AI Powered Employee Attrition Prediction System")

# --------------------------
# KPI CARDS
# --------------------------

col1,col2,col3,col4 = st.columns(4)

with col1:
    st.metric("Employees",1470)

with col2:
    st.metric("Attrition Rate","16.1%")

with col3:
    st.metric("Active Employees",1233)

with col4:
    st.metric("Employees Left",237)

st.divider()

# --------------------------
# SIDEBAR
# --------------------------

st.sidebar.header("Employee Details")

age = st.sidebar.slider("Age",18,60,30)

income = st.sidebar.number_input(
    "Monthly Income",
    1000,
    50000,
    5000
)

job_satisfaction = st.sidebar.selectbox(
    "Job Satisfaction",
    [1,2,3,4]
)

total_years = st.sidebar.slider(
    "Total Working Years",
    0,
    40,
    5
)

overtime = st.sidebar.selectbox(
    "Over Time",
    ["No","Yes"]
)

# --------------------------
# CHARTS
# --------------------------

col1,col2 = st.columns(2)

with col1:

    chart_df = pd.DataFrame({
        "Department":["Sales","R&D","HR"],
        "Attrition":[133,92,12]
    })

    st.subheader("Department Attrition")

    st.bar_chart(
        chart_df.set_index("Department")
    )

with col2:

    gender_df = pd.DataFrame({
        "Gender":["Male","Female"],
        "Count":[150,87]
    })

    st.subheader("Gender Attrition")

    st.bar_chart(
        gender_df.set_index("Gender")
    )

# --------------------------
# PREDICTION
# --------------------------

st.divider()

st.subheader("Predict Employee Attrition")

if st.button("Predict Attrition"):

    sample = pd.DataFrame(
        np.zeros(
            (1,len(feature_cols))
        ),
        columns=feature_cols
    )

    # Fill important features

    sample["Age"] = age
    sample["MonthlyIncome"] = income
    sample["JobSatisfaction"] = job_satisfaction
    sample["TotalWorkingYears"] = total_years

    sample["OverTime"] = 1 if overtime=="Yes" else 0

    pred = model.predict(sample)[0]

    try:
        prob = model.predict_proba(sample)[0][1]
    except:
        prob = 0.50

    if pred == 1:

        st.error(
            f"⚠️ High Attrition Risk ({prob:.2%})"
        )

    else:

        st.success(
            f"✅ Low Attrition Risk ({prob:.2%})"
        )

# --------------------------
# MODEL COMPARISON
# --------------------------

st.divider()

st.subheader("Model Accuracy Comparison")

acc_df = pd.DataFrame({

    "Model":[
        "Random Forest",
        "Logistic Regression",
        "SVM",
        "KNN",
        "Decision Tree",
        "Naive Bayes"
    ],

    "Accuracy":[
        88,
        84,
        83,
        80,
        78,
        75
    ]
})

st.bar_chart(
    acc_df.set_index("Model")
)