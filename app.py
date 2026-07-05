# ============================================================
# ChurnShield
# AI-Powered Customer Churn Prediction Dashboard
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ChurnShield",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""

<style>

.main{
    background-color:#F8F9FA;
}

.metric-card{
    background:white;
    padding:18px;
    border-radius:12px;
    box-shadow:0px 2px 8px rgba(0,0,0,0.08);
}

h1,h2,h3{
    color:#0E4D92;
}

</style>

""",unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv("cleaned_telco.csv")

    return df

@st.cache_resource
def load_model():

    model = joblib.load("best_model.pkl")

    return model

df = load_data()

model = load_model()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.image(
    "https://img.icons8.com/color/96/artificial-intelligence.png",
    width=80
)

st.sidebar.title("ChurnShield")

page = st.sidebar.radio(

    "Navigation",

    [

        "🏠 Executive Dashboard",

        "📈 Exploratory Analysis",

        "🤖 Live Prediction",

        "📉 Model Performance",

        "📄 Business Report",

        "ℹ About"

    ]

)

# ============================================================
# EXECUTIVE DASHBOARD
# ============================================================

if page=="🏠 Executive Dashboard":

    st.title("📊 Executive Dashboard")

    st.markdown(
        "Comprehensive overview of telecom customer churn."
    )

    st.divider()

    total_customers = len(df)

    churned = df["Churn"].sum()

    active = total_customers - churned

    churn_rate = round(
        churned/total_customers*100,
        2
    )

    avg_monthly = round(
        df["MonthlyCharges"].mean(),
        2
    )

    avg_total = round(
        df["TotalCharges"].mean(),
        2
    )

    avg_tenure = round(
        df["tenure"].mean(),
        2
    )

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(
        "👥 Customers",
        f"{total_customers:,}"
    )

    c2.metric(
        "🔴 Churned",
        churned
    )

    c3.metric(
        "🟢 Active",
        active
    )

    c4.metric(
        "📈 Churn Rate",
        f"{churn_rate}%"
    )

    c5,c6,c7 = st.columns(3)

    c5.metric(
        "💳 Avg Monthly Charges",
        f"${avg_monthly}"
    )

    c6.metric(
        "💰 Avg Total Charges",
        f"${avg_total}"
    )

    c7.metric(
        "⏳ Avg Tenure",
        f"{avg_tenure} Months"
    )

    st.divider()

    left,right = st.columns(2)

    with left:

        with st.expander(
            "📌 Customer Status Distribution",
            expanded=True
        ):

            churn_df = (
                df["Churn"]
                .replace(
                    {
                        0:"Active",
                        1:"Churned"
                    }
                )
                .value_counts()
                .reset_index()
            )

            churn_df.columns=[
                "Customer Status",
                "Customers"
            ]

            fig = px.pie(

                churn_df,

                names="Customer Status",

                values="Customers",

                hole=.55,

                title="Customer Distribution"

            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with right:

        with st.expander(
            "📌 Contract Distribution",
            expanded=True
        ):

            contract = (
                df["Contract"]
                .value_counts()
                .reset_index()
            )

            contract.columns=[
                "Contract",
                "Customers"
            ]

            fig = px.bar(

                contract,

                x="Contract",

                y="Customers",

                color="Contract",

                text="Customers",

                title="Contract Types"

            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    st.divider()

    left,right = st.columns(2)

    with left:

        with st.expander(
            "📌 Monthly Charges Distribution",
            expanded=True
        ):

            fig = px.histogram(

                df,

                x="MonthlyCharges",

                color=df["Churn"].replace(
                    {
                        0:"Active",
                        1:"Churned"
                    }
                ),

                marginal="box",

                nbins=30,

                title="Monthly Charges"

            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with right:

        with st.expander(
            "📌 Customer Tenure",
            expanded=True
        ):

            fig = px.histogram(

                df,

                x="tenure",

                color=df["Churn"].replace(
                    {
                        0:"Active",
                        1:"Churned"
                    }
                ),

                marginal="box",

                nbins=30,

                title="Customer Tenure"

            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    st.divider()

    st.subheader("📌 Executive Business Insights")

    insight1,insight2=st.columns(2)

    with insight1:

        st.success("""

### Key Findings

- Majority of customers remain active.

- Month-to-month contracts exhibit the highest churn.

- Customers with shorter tenure are more likely to churn.

- High monthly charges are associated with increased churn.

""")

    with insight2:

        st.info("""

### Recommendations

- Encourage annual contracts.

- Provide loyalty discounts.

- Target new customers with onboarding campaigns.

- Improve customer support.

""")
# ============================================================
# EXPLORATORY ANALYSIS
# ============================================================

elif page == "📈 Exploratory Analysis":

    st.title("📈 Exploratory Data Analysis")

    st.markdown(
        "Analyze customer behaviour using interactive visualizations."
    )

    st.divider()

    # --------------------------------------------------------
    # SIDEBAR FILTERS
    # --------------------------------------------------------

    st.sidebar.subheader("EDA Filters")

    contract_filter = st.sidebar.multiselect(
        "Contract Type",
        options=sorted(df["Contract"].unique()),
        default=sorted(df["Contract"].unique())
    )

    internet_filter = st.sidebar.multiselect(
        "Internet Service",
        options=sorted(df["InternetService"].unique()),
        default=sorted(df["InternetService"].unique())
    )

    payment_filter = st.sidebar.multiselect(
        "Payment Method",
        options=sorted(df["PaymentMethod"].unique()),
        default=sorted(df["PaymentMethod"].unique())
    )

    churn_filter = st.sidebar.multiselect(
        "Customer Status",
        options=["Active","Churned"],
        default=["Active","Churned"]
    )

    filtered = df.copy()

    filtered = filtered[
        filtered["Contract"].isin(contract_filter)
    ]

    filtered = filtered[
        filtered["InternetService"].isin(internet_filter)
    ]

    filtered = filtered[
        filtered["PaymentMethod"].isin(payment_filter)
    ]

    filtered = filtered[
        filtered["Churn"].replace(
            {0:"Active",1:"Churned"}
        ).isin(churn_filter)
    ]

    st.success(f"Filtered Customers : {len(filtered):,}")

    st.divider()

    # --------------------------------------------------------
    # ROW 1
    # --------------------------------------------------------

    c1,c2 = st.columns(2)

    with c1:

        with st.expander(
            "📌 Churn by Contract",
            expanded=True
        ):

            temp = (

                filtered.groupby("Contract")["Churn"]

                .mean()

                .reset_index()

            )

            temp["Churn Rate"] = temp["Churn"]*100

            fig = px.bar(

                temp,

                x="Contract",

                y="Churn Rate",

                color="Contract",

                text=temp["Churn Rate"].round(1),

                title="Churn Rate by Contract"

            )

            fig.update_traces(
                textposition="outside"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with c2:

        with st.expander(
            "📌 Internet Service Analysis",
            expanded=True
        ):

            temp=(

                filtered.groupby("InternetService")["Churn"]

                .mean()

                .reset_index()

            )

            temp["Churn Rate"]=temp["Churn"]*100

            fig=px.bar(

                temp,

                x="InternetService",

                y="Churn Rate",

                color="InternetService",

                text=temp["Churn Rate"].round(1),

                title="Internet Service vs Churn"

            )

            fig.update_traces(
                textposition="outside"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    st.divider()

    # --------------------------------------------------------
    # ROW 2
    # --------------------------------------------------------

    c1,c2 = st.columns(2)

    with c1:

        with st.expander(
            "📌 Monthly Charges",
            expanded=True
        ):

            fig = px.box(

                filtered,

                x=filtered["Churn"].replace(
                    {
                        0:"Active",
                        1:"Churned"
                    }
                ),

                y="MonthlyCharges",

                color=filtered["Churn"].replace(
                    {
                        0:"Active",
                        1:"Churned"
                    }
                ),

                title="Monthly Charges"

            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with c2:

        with st.expander(
            "📌 Total Charges",
            expanded=True
        ):

            fig = px.box(

                filtered,

                x=filtered["Churn"].replace(
                    {
                        0:"Active",
                        1:"Churned"
                    }
                ),

                y="TotalCharges",

                color=filtered["Churn"].replace(
                    {
                        0:"Active",
                        1:"Churned"
                    }
                ),

                title="Total Charges"

            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    st.divider()

    # --------------------------------------------------------
    # ROW 3
    # --------------------------------------------------------

    c1,c2 = st.columns(2)

    with c1:

        with st.expander(
            "📌 Customer Tenure",
            expanded=True
        ):

            fig = px.histogram(

                filtered,

                x="tenure",

                color=filtered["Churn"].replace(
                    {
                        0:"Active",
                        1:"Churned"
                    }
                ),

                nbins=25,

                marginal="box",

                title="Customer Tenure"

            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with c2:

        with st.expander(
            "📌 Monthly Charges vs Tenure",
            expanded=True
        ):

            fig = px.scatter(

                filtered,

                x="tenure",

                y="MonthlyCharges",

                color=filtered["Churn"].replace(
                    {
                        0:"Active",
                        1:"Churned"
                    }
                ),

                hover_data=[
                    "Contract",
                    "InternetService"
                ],

                title="Tenure vs Monthly Charges"

            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    st.divider()

    # --------------------------------------------------------
    # CORRELATION
    # --------------------------------------------------------

    with st.expander(
        "📌 Correlation Heatmap",
        expanded=True
    ):

        numeric = filtered.select_dtypes(
            include=np.number
        )

        corr = numeric.corr()

        fig = px.imshow(

            corr,

            text_auto=True,

            color_continuous_scale="RdBu_r",

            title="Correlation Matrix"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # --------------------------------------------------------
    # SUNBURST
    # --------------------------------------------------------

    with st.expander(
        "📌 Customer Segmentation",
        expanded=False
    ):

        temp = filtered.copy()

        temp["Status"] = temp["Churn"].replace(
            {
                0:"Active",
                1:"Churned"
            }
        )

        fig = px.sunburst(

            temp,

            path=[
                "Contract",
                "InternetService",
                "Status"
            ],

            title="Customer Segmentation"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    st.download_button(

        "⬇ Download Filtered Dataset",

        filtered.to_csv(index=False),

        "filtered_telco.csv",

        "text/csv"

    )

    st.divider()

    # --------------------------------------------------------
    # BUSINESS INSIGHTS
    # --------------------------------------------------------

    st.subheader("📌 Business Insights")

    st.info("""

### Key Observations

• Month-to-month contracts show the highest churn.

• Fiber Optic customers are more likely to churn.

• Customers with higher monthly charges tend to leave more often.

• Long-tenure customers are highly loyal.

• Electronic Check is associated with higher churn.

• One-Year and Two-Year contracts significantly reduce churn.

""")
# ============================================================
# LIVE CHURN PREDICTION
# ============================================================

elif page == "🤖 Live Prediction":

    st.title("🤖 Live Customer Churn Prediction")

    st.markdown(
        "Enter customer information to predict churn probability."
    )

    st.divider()

    col1, col2 = st.columns(2)

    # -------------------------
    # LEFT COLUMN
    # -------------------------

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        senior = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        tenure = st.slider(
            "Tenure (Months)",
            0,
            72,
            12
        )

        phone = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )

        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

    # -------------------------
    # RIGHT COLUMN
    # -------------------------

    with col2:

        backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

        protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

        support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

        tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        paperless = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=200.0,
            value=70.0
        )

        total = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=1000.0
        )

    st.divider()

    predict = st.button(
        "🚀 Predict Churn",
        use_container_width=True
    )

    if predict:

        input_df = pd.DataFrame({

            "gender":[gender],

            "SeniorCitizen":[senior],

            "Partner":[partner],

            "Dependents":[dependents],

            "tenure":[tenure],

            "PhoneService":[phone],

            "MultipleLines":[multiple],

            "InternetService":[internet],

            "OnlineSecurity":[security],

            "OnlineBackup":[backup],

            "DeviceProtection":[protection],

            "TechSupport":[support],

            "StreamingTV":[tv],

            "StreamingMovies":[movies],

            "Contract":[contract],

            "PaperlessBilling":[paperless],

            "PaymentMethod":[payment],

            "MonthlyCharges":[monthly],

            "TotalCharges":[total],

            "TenureGroup":[
                pd.cut(
                    [tenure],
                    bins=[0,12,24,48,60,72],
                    labels=[
                        "0-12 Months",
                        "13-24 Months",
                        "25-48 Months",
                        "49-60 Months",
                        "61-72 Months"
                    ]
                )[0]
            ],

            "MonthlyChargeCategory":[
                "Low"
                if monthly < 40 else
                "Medium"
                if monthly < 80 else
                "High"
            ]

        })

        probability = model.predict_proba(input_df)[0][1]

        prediction = model.predict(input_df)[0]

        st.divider()

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Prediction",
            "Churn" if prediction == 1 else "No Churn"
        )

        c2.metric(
            "Probability",
            f"{probability:.2%}"
        )

        if probability < 0.30:
            risk = "🟢 LOW"

        elif probability < 0.70:
            risk = "🟠 MEDIUM"

        else:
            risk = "🔴 HIGH"

        c3.metric(
            "Risk Level",
            risk
        )

        st.session_state["prediction"] = prediction
        st.session_state["probability"] = probability
        st.session_state["risk"] = risk
        st.session_state["customer"] = input_df
        # ============================================================
        # PROBABILITY GAUGE
        # ============================================================

        st.divider()

        gauge = go.Figure(go.Indicator(

            mode="gauge+number",

            value=probability*100,

            number={"suffix":"%"},

            title={"text":"Churn Probability"},

            gauge={

                "axis":{"range":[0,100]},

                "bar":{"color":"darkblue"},

                "steps":[

                    {"range":[0,30],"color":"lightgreen"},

                    {"range":[30,70],"color":"gold"},

                    {"range":[70,100],"color":"tomato"}

                ]

            }

        ))

        st.plotly_chart(
            gauge,
            use_container_width=True
        )

        # ============================================================
        # PROGRESS BAR
        # ============================================================

        st.subheader("Prediction Confidence")

        st.progress(float(probability))

        st.write(f"**Model Confidence:** {probability:.2%}")

        # ============================================================
        # RETENTION RECOMMENDATIONS
        # ============================================================

        st.divider()

        st.subheader("Retention Recommendation")

        if probability >= 0.70:

            st.error("""

### 🔴 High Risk Customer

Recommended Actions

- Offer retention discount

- Contact customer immediately

- Provide premium technical support

- Offer annual contract

- Assign customer success manager

""")

        elif probability >= 0.30:

            st.warning("""

### 🟠 Medium Risk Customer

Recommended Actions

- Send promotional offers

- Loyalty rewards

- Encourage additional services

- Improve engagement

""")

        else:

            st.success("""

### 🟢 Low Risk Customer

Recommended Actions

- Continue normal engagement

- Reward loyalty

- Upsell premium plans

- Maintain satisfaction

""")

        # ============================================================
        # CUSTOMER SUMMARY
        # ============================================================

        st.divider()

        st.subheader("Customer Summary")

        summary = pd.DataFrame({

            "Attribute":[

                "Gender",

                "Senior Citizen",

                "Partner",

                "Dependents",

                "Tenure",

                "Contract",

                "Internet",

                "Monthly Charges",

                "Total Charges"

            ],

            "Value":[

                gender,

                senior,

                partner,

                dependents,

                tenure,

                contract,

                internet,

                monthly,

                total

            ]

        })

        st.dataframe(
            summary,
            use_container_width=True
        )

        # ============================================================
        # DOWNLOAD REPORT
        # ============================================================

        report = pd.DataFrame({

            "Prediction":[

                "Churn" if prediction==1 else "No Churn"

            ],

            "Probability":[

                round(probability*100,2)

            ],

            "Risk Level":[risk],

            "Contract":[contract],

            "Internet":[internet],

            "Monthly Charges":[monthly],

            "Total Charges":[total],

            "Tenure":[tenure]

        })

        st.download_button(

            "⬇ Download Prediction Report",

            report.to_csv(index=False),

            file_name="prediction_report.csv",

            mime="text/csv"

        )

        # ============================================================
        # FINAL BUSINESS SUMMARY
        # ============================================================

        st.divider()

        st.subheader("Executive Interpretation")

        if prediction == 1:

            st.info(f"""

The model predicts that this customer is **likely to churn**.

**Risk Level:** {risk}

**Estimated Probability:** {probability:.2%}

### Suggested Business Actions

• Contact customer within 48 hours.

• Offer a loyalty discount.

• Promote annual contract.

• Assign priority customer support.

• Review billing concerns.

""")

        else:

            st.success(f"""

The model predicts that this customer is **likely to remain with the company**.

**Risk Level:** {risk}

**Estimated Probability:** {probability:.2%}

### Suggested Business Actions

• Continue customer engagement.

• Offer premium upgrades.

• Maintain service quality.

• Reward long-term loyalty.

""")
# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📉 Model Performance":

    st.title("📉 Model Performance Dashboard")

    st.markdown(
        "Performance metrics of the trained Customer Churn Prediction Model."
    )

    st.divider()

    # --------------------------------------------------------
    # LOAD MODEL RESULTS
    # --------------------------------------------------------

    try:

        results = pd.read_csv("model_results.csv")

    except:

        st.error("model_results.csv not found.")

        st.stop()

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    best = results.sort_values(
        "ROC AUC",
        ascending=False
    ).iloc[0]

    accuracy = best["Accuracy"]
    precision = best["Precision"]
    recall = best["Recall"]
    f1 = best["F1"]
    auc = best["ROC AUC"]

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    c1,c2,c3,c4,c5 = st.columns(5)

    c1.metric(
        "Accuracy",
        f"{accuracy:.2%}"
    )

    c2.metric(
        "Precision",
        f"{precision:.2%}"
    )

    c3.metric(
        "Recall",
        f"{recall:.2%}"
    )

    c4.metric(
        "F1 Score",
        f"{f1:.2%}"
    )

    c5.metric(
        "ROC-AUC",
        f"{auc:.2%}"
    )

    st.divider()

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    st.subheader("Model Comparison")

    st.dataframe(
        results,
        use_container_width=True
    )

    fig = px.bar(

        results,

        x="Model",

        y=["Accuracy","Precision","Recall","F1","ROC AUC"],

        barmode="group",

        title="Model Comparison"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # RADAR CHART
    # --------------------------------------------------------

    st.subheader("Performance Radar")

    radar = go.Figure()

    radar.add_trace(

        go.Scatterpolar(

            r=[
                accuracy,
                precision,
                recall,
                f1,
                auc
            ],

            theta=[
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "ROC-AUC"
            ],

            fill="toself",

            name=best["Model"]

        )

    )

    radar.update_layout(

        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0,1]
            )
        ),

        showlegend=False

    )

    st.plotly_chart(
        radar,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.subheader("Feature Importance")

    try:

        pipeline = joblib.load("best_model.pkl")

        rf = pipeline.named_steps["model"]

        feature_names = pipeline.named_steps[
            "preprocessor"
        ].get_feature_names_out()

        importance = pd.DataFrame({

            "Feature":feature_names,

            "Importance":rf.feature_importances_

        })

        importance = importance.sort_values(

            "Importance",

            ascending=False

        ).head(20)

        fig = px.bar(

            importance,

            x="Importance",

            y="Feature",

            orientation="h",

            title="Top 20 Important Features"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    except:

        st.warning(
            "Feature importance unavailable."
        )

    st.divider()

    # --------------------------------------------------------
    # PERFORMANCE SUMMARY
    # --------------------------------------------------------

    c1,c2 = st.columns(2)

    with c1:

        st.success(f"""

### Best Model

**{best['Model']}**

Accuracy : **{accuracy:.2%}**

Precision : **{precision:.2%}**

Recall : **{recall:.2%}**

ROC-AUC : **{auc:.2%}**

""")

    with c2:

        st.info("""

### Interpretation

✔ High Accuracy

✔ Strong Recall

✔ Suitable for Customer Retention

✔ Good Generalization

✔ Ready for Deployment

""")

    st.divider()

    # --------------------------------------------------------
    # METRIC DEFINITIONS
    # --------------------------------------------------------

    with st.expander(
        "Understanding the Metrics"
    ):

        st.markdown("""

### Accuracy
Percentage of total predictions that were correct.

### Precision
Among customers predicted to churn, how many actually churned.

### Recall
Among customers who actually churned, how many were correctly identified.

### F1 Score
Balances Precision and Recall.

### ROC-AUC
Measures how well the model separates churners from non-churners.

Higher values indicate better discrimination.

""")

    st.divider()

    # --------------------------------------------------------
    # DOWNLOAD RESULTS
    # --------------------------------------------------------

    st.download_button(

        "⬇ Download Model Results",

        results.to_csv(index=False),

        file_name="model_results.csv",

        mime="text/csv"

    )
# ============================================================
# BUSINESS REPORT
# ============================================================

elif page == "📄 Business Report":

    st.title("📄 Executive Business Report")

    st.markdown(
        "Business insights generated from the Customer Churn Analysis."
    )

    st.divider()

    total_customers = len(df)

    churned = df["Churn"].sum()

    active = total_customers - churned

    churn_rate = round(churned/total_customers*100,2)

    avg_monthly = round(df["MonthlyCharges"].mean(),2)

    avg_total = round(df["TotalCharges"].mean(),2)

    avg_tenure = round(df["tenure"].mean(),2)

    st.subheader("Executive Summary")

    st.info(f"""

Total Customers : **{total_customers:,}**

Active Customers : **{active:,}**

Churned Customers : **{churned:,}**

Overall Churn Rate : **{churn_rate}%**

Average Monthly Charges : **${avg_monthly}**

Average Total Charges : **${avg_total}**

Average Customer Tenure : **{avg_tenure} Months**

""")

    st.divider()

    # ----------------------------------------------------

    st.subheader("Customer Distribution")

    c1,c2 = st.columns(2)

    with c1:

        fig = px.pie(

            df,

            names=df["Churn"].replace({

                0:"Active",

                1:"Churned"

            }),

            title="Customer Status",

            hole=.55

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    with c2:

        contract = (

            df["Contract"]

            .value_counts()

            .reset_index()

        )

        contract.columns=[

            "Contract",

            "Customers"

        ]

        fig = px.bar(

            contract,

            x="Contract",

            y="Customers",

            color="Contract",

            text="Customers"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    st.divider()

    # ----------------------------------------------------

    st.subheader("Top Business Recommendations")

    st.success("""

### Customer Retention Strategy

• Encourage Month-to-Month customers to switch to annual plans.

• Offer loyalty rewards after 12 months.

• Provide premium support for high-risk customers.

• Improve onboarding during the first six months.

• Target Fiber Optic customers with retention campaigns.

• Reduce churn by offering personalized discounts.

• Promote bundled services.

""")

    st.divider()

    # ----------------------------------------------------

    st.subheader("Key Risk Factors")

    risk = pd.DataFrame({

        "Risk Factor":[

            "Month-to-Month Contract",

            "High Monthly Charges",

            "Short Tenure",

            "Electronic Check",

            "No Tech Support",

            "No Online Security"

        ],

        "Business Impact":[

            "Very High",

            "High",

            "High",

            "Medium",

            "Medium",

            "High"

        ]

    })

    st.dataframe(

        risk,

        use_container_width=True

    )

    st.divider()

    # ----------------------------------------------------

    st.subheader("Project Deliverables")

    deliverables = pd.DataFrame({

        "Completed":[

            "✔ Data Cleaning",

            "✔ Exploratory Data Analysis",

            "✔ Feature Engineering",

            "✔ Machine Learning Model",

            "✔ Model Evaluation",

            "✔ Streamlit Dashboard",

            "✔ Business Report"

        ]

    })

    st.dataframe(

        deliverables,

        use_container_width=True

    )

    st.divider()

    report = pd.DataFrame({

        "Metric":[

            "Customers",

            "Active",

            "Churned",

            "Churn Rate",

            "Average Monthly Charges",

            "Average Total Charges",

            "Average Tenure"

        ],

        "Value":[

            total_customers,

            active,

            churned,

            churn_rate,

            avg_monthly,

            avg_total,

            avg_tenure

        ]

    })

    st.download_button(

        "⬇ Download Executive Report",

        report.to_csv(index=False),

        file_name="Executive_Report.csv",

        mime="text/csv"

    )

# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ About":

    st.title("ℹ About ChurnShield")

    st.markdown("""

## ChurnShield

An AI-powered Customer Churn Prediction Dashboard developed using Data Science and Machine Learning.

---

### Project Objectives

- Predict customer churn
- Understand customer behaviour
- Reduce churn using business insights
- Support management decision making

---

### Technology Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- Plotly
- Streamlit
- Joblib

---

### Machine Learning

- Logistic Regression
- Random Forest
- Feature Engineering
- Hyperparameter Tuning

---

### Dashboard Features

✔ Executive Dashboard

✔ Interactive EDA

✔ Live Churn Prediction

✔ Model Performance

✔ Business Report

---

### Developed By

**Vyjayanthi Punuri**

Data Science Internship Project

2026

""")

    st.success("Thank you for exploring ChurnShield! 🚀")