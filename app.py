# # ============================================
# # FINAL INTELLIGENT XAI DASHBOARD (NO API LLM)
# # ============================================

# import streamlit as st
# import pandas as pd
# import numpy as np
# import shap
# import joblib
# import matplotlib.pyplot as plt

# from lime.lime_tabular import LimeTabularExplainer

# # ============================================
# # PAGE CONFIG
# # ============================================
# st.set_page_config(layout="wide")
# st.title("🧠 Explainable AI Marketing Intelligence Dashboard")

# # ============================================
# # LOAD DATA
# # ============================================
# df = pd.read_csv("online_shoppers_intention.csv")
# df['Revenue'] = df['Revenue'].astype(int)

# df_encoded = pd.get_dummies(df, drop_first=True)

# X = df_encoded.drop("Revenue", axis=1)
# y = df_encoded["Revenue"]

# model = joblib.load("model.pkl")

# # ============================================
# # SHAP
# # ============================================
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X)

# if isinstance(shap_values, list):
#     shap_values_to_use = shap_values[1]
# else:
#     shap_values_to_use = shap_values

# # ============================================
# # LIME
# # ============================================
# lime_explainer = LimeTabularExplainer(
#     training_data=X.values,
#     feature_names=X.columns.tolist(),
#     class_names=["No Purchase", "Purchase"],
#     mode="classification"
# )

# # ============================================
# # SIDEBAR INPUT
# # ============================================
# st.sidebar.header("🎯 Simulate User Behavior")

# page_value = st.sidebar.slider("PageValues", 0.0, 300.0, 20.0)
# exit_rate = st.sidebar.slider("ExitRates", 0.0, 1.0, 0.2)
# bounce_rate = st.sidebar.slider("BounceRates", 0.0, 1.0, 0.2)
# product_duration = st.sidebar.slider("ProductRelated_Duration", 0.0, 500.0, 50.0)

# # Create input
# input_data = X.iloc[0:1].copy()
# input_data[:] = 0

# input_data["PageValues"] = page_value
# input_data["ExitRates"] = exit_rate
# input_data["BounceRates"] = bounce_rate
# input_data["ProductRelated_Duration"] = product_duration

# # ============================================
# # PREDICTION
# # ============================================
# prediction = model.predict(input_data)[0]
# prob = model.predict_proba(input_data)[0][1]

# st.subheader("🔮 Prediction")
# st.write(f"Probability of Purchase: **{prob:.2f}**")

# if prediction == 1:
#     st.success("✅ Likely to Purchase")
# else:
#     st.error("❌ Not Likely to Purchase")

# # ============================================
# # SHAP GLOBAL
# # ============================================
# st.subheader("📊 Global Feature Importance (SHAP)")
# fig1, ax1 = plt.subplots()
# shap.summary_plot(shap_values_to_use, X, plot_type="bar", show=False)
# st.pyplot(fig1)

# # ============================================
# # SHAP LOCAL (Waterfall)
# # ============================================
# st.subheader("🧍 SHAP Local Explanation")

# input_shap = explainer.shap_values(input_data)

# if isinstance(input_shap, list):
#     shap_val = input_shap[1]
#     base_val = explainer.expected_value[1]
# else:
#     shap_val = input_shap
#     base_val = explainer.expected_value

# fig2 = plt.figure()
# shap.plots.waterfall(
#     shap.Explanation(
#         values=shap_val[0],
#         base_values=base_val,
#         data=input_data.iloc[0],
#         feature_names=input_data.columns
#     ),
#     show=False
# )
# st.pyplot(fig2)

# # ============================================
# # LIME LOCAL EXPLANATION
# # ============================================
# st.subheader("🧪 LIME Explanation")

# lime_exp = lime_explainer.explain_instance(
#     input_data.values[0],
#     model.predict_proba,
#     num_features=5
# )

# lime_list = lime_exp.as_list()
# for feature, weight in lime_list:
#     st.write(f"{feature} → {weight:.3f}")

# # ============================================
# # INTELLIGENT INSIGHT ENGINE (NO LLM API)
# # ============================================
# st.subheader("💡 AI-Generated Business Insights")

# insights = []

# if page_value > 50:
#     insights.append("High Page Value indicates strong buying intent.")
# else:
#     insights.append("Low Page Value suggests weak product attractiveness.")

# if exit_rate > 0.3:
#     insights.append("High Exit Rate indicates users are dropping off early.")
# else:
#     insights.append("Low Exit Rate shows good engagement.")

# if bounce_rate > 0.3:
#     insights.append("High Bounce Rate suggests poor landing page experience.")
# else:
#     insights.append("Bounce Rate is under control.")

# if product_duration > 100:
#     insights.append("High product engagement indicates strong interest.")
# else:
#     insights.append("Low engagement suggests need for better recommendations.")

# for i in insights:
#     st.write("• " + i)

# # ============================================
# # FINAL BUSINESS CONCLUSION GENERATOR
# # ============================================
# st.subheader("📌 Final Business Decision")

# if prediction == 1:
#     st.success("""
#     ✔ Target this user with premium products  
#     ✔ Offer upselling opportunities  
#     ✔ Use personalized recommendations  
#     """)
# else:
#     st.warning("""
#     ✔ Retarget user with ads  
#     ✔ Offer discounts or coupons  
#     ✔ Improve landing page experience  
#     """)

# # ============================================
# # OVERALL CONCLUSION (AUTO GENERATED)
# # ============================================
# st.subheader("🧾 Final Conclusion")

# conclusion = f"""
# The system predicts a purchase probability of {prob:.2f}. 
# Key influencing factors include PageValues, ExitRates, and user engagement.

# Business Recommendation:
# - Optimize high-value pages
# - Reduce exit and bounce rates
# - Focus on user engagement strategies

# This Explainable AI system helps businesses make transparent and data-driven decisions.
# """

# st.write(conclusion)
# ============================================
# ULTIMATE MARKETING AI DASHBOARD (ONE SHOT)
# ============================================

# import streamlit as st
# import pandas as pd
# import numpy as np
# import shap
# import joblib
# import plotly.express as px
# import matplotlib.pyplot as plt
# from lime.lime_tabular import LimeTabularExplainer
# from statsmodels.tsa.arima.model import ARIMA
# import warnings
# warnings.filterwarnings("ignore")

# # ============================================
# # CONFIG
# # ============================================
# st.set_page_config(layout="wide")
# st.title("🧠 Explainable AI for Marketing Decision Making")

# # ============================================
# # LOAD DATA
# # ============================================
# uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv", "xlsx"])

# def load_data(file):
#     if file:
#         return pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
#     return pd.read_csv("online_shoppers_intention.csv")

# df = load_data(uploaded_file)
# df["Revenue"] = df["Revenue"].astype(int)

# # ============================================
# # SIDEBAR FILTERS (POWER BI STYLE)
# # ============================================
# st.sidebar.header("🎛 Filters")

# bounce_filter = st.sidebar.slider("Max Bounce Rate", 0.0, 1.0, 1.0)
# exit_filter = st.sidebar.slider("Max Exit Rate", 0.0, 1.0, 1.0)

# df = df[(df["BounceRates"] <= bounce_filter) &
#         (df["ExitRates"] <= exit_filter)]

# # ============================================
# # PREPROCESS
# # ============================================
# df_encoded = pd.get_dummies(df, drop_first=True)
# X = df_encoded.drop("Revenue", axis=1)
# y = df_encoded["Revenue"]

# model = joblib.load("model.pkl")

# # ============================================
# # SHAP SAMPLE
# # ============================================
# explainer = shap.TreeExplainer(model)
# sample_X = X.sample(min(200, len(X)))
# shap_values = explainer.shap_values(sample_X)
# if isinstance(shap_values, list):
#     shap_values = shap_values[1]

# # ============================================
# # TABS
# # ============================================
# tab1, tab2, tab3, tab4, tab5 = st.tabs([
#     "📊 Overview",
#     "📈 Analytics",
#     "🧠 AI Insights",
#     "🎯 Marketing",
#     "🧪 Experiments"
# ])

# # ============================================
# # 📊 OVERVIEW
# # ============================================
# with tab1:
#     st.subheader("📊 Executive Dashboard")

#     c1, c2, c3, c4 = st.columns(4)
#     c1.metric("Users", len(df))
#     c2.metric("Conversions", df["Revenue"].sum())
#     c3.metric("Conversion Rate", f"{df['Revenue'].mean()*100:.2f}%")
#     c4.metric("Avg Engagement", f"{df['ProductRelated_Duration'].mean():.2f}")

#     df["Segment"] = pd.cut(df["PageValues"],
#                            bins=[-1, 10, 50, 1000],
#                            labels=["Low", "Medium", "High"])

#     fig = px.pie(df, names="Segment", title="Customer Segments")
#     st.plotly_chart(fig, width="stretch")

# # ============================================
# # 📈 ANALYTICS
# # ============================================
# with tab2:
#     st.subheader("📈 Behavior Analysis")

#     col1, col2 = st.columns(2)

#     with col1:
#         fig = px.scatter(df, x="PageValues", y="ProductRelated_Duration",
#                          color="Revenue")
#         st.plotly_chart(fig, width="stretch")

#     with col2:
#         fig = px.histogram(df, x="BounceRates", color="Revenue")
#         st.plotly_chart(fig, width="stretch")

#     # Correlation
#     corr = df.select_dtypes(include=np.number).corr()
#     fig = px.imshow(corr, text_auto=True)
#     st.plotly_chart(fig, width="stretch")

#     # TIME SERIES (FAKE IF NO DATE)
#     st.subheader("📅 Sales Forecast")

#     if "Month" in df.columns:
#         ts = df.groupby("Month")["Revenue"].sum()
#     else:
#         df["Time"] = np.arange(len(df))
#         ts = df.groupby("Time")["Revenue"].sum()

#     model_arima = ARIMA(ts, order=(1,1,1)).fit()
#     forecast = model_arima.forecast(steps=10)

#     fig = px.line(x=range(len(ts)), y=ts, title="Actual Sales")
#     # fig.add_scatter(x=range(len(ts), len(ts)+10), y=forecast, mode='lines', name='Forecast')
#     # Convert everything properly
#     actual_x = list(range(len(ts)))
#     forecast_x = list(range(len(ts), len(ts)+len(forecast)))

#     fig = px.line(x=actual_x, y=ts, title="Actual vs Forecast")

#     fig.add_scatter(
#         x=forecast_x,
#         y=forecast,
#         mode='lines',
#         name='Forecast' 
#     )

#     st.plotly_chart(fig, width="stretch")
#     # st.plotly_chart(fig, width="stretch")

# # ============================================
# # 🧠 AI INSIGHTS
# # ============================================
# with tab3:
#     st.subheader("🧠 Explainable AI")

#     fig, ax = plt.subplots()
#     shap.summary_plot(shap_values, sample_X, plot_type="bar", show=False)
#     st.pyplot(fig)

#     st.info("""
#     - PageValues = strongest purchase driver
#     - BounceRate negatively impacts conversion
#     """)

# # ============================================
# # 🎯 MARKETING + PREDICTION
# # ============================================
# with tab4:
#     st.subheader("🎯 Marketing Decision Engine")

#     pv = st.slider("Intent Score", 0.0, 300.0, 20.0)
#     br = st.slider("Bounce Rate", 0.0, 1.0, 0.2)
#     er = st.slider("Exit Rate", 0.0, 1.0, 0.2)
#     dur = st.slider("Engagement", 0.0, 500.0, 50.0)

#     input_data = pd.DataFrame(np.zeros((1, X.shape[1])), columns=X.columns)

#     for col, val in {
#         "PageValues": pv,
#         "BounceRates": br,
#         "ExitRates": er,
#         "ProductRelated_Duration": dur
#     }.items():
#         if col in input_data.columns:
#             input_data[col] = val

#     prob = model.predict_proba(input_data)[0][1]
#     st.progress(int(prob * 100))
#     st.write(f"Purchase Probability: {prob:.2f}")

#     # SEGMENT
#     if prob > 0.7:
#         seg = "High Value"
#     elif prob > 0.4:
#         seg = "Potential"
#     else:
#         seg = "At Risk"

#     st.success(f"Segment: {seg}")

#     # 🤖 LLM STYLE
#     if seg == "High Value":
#         rec = "Upsell premium products, loyalty rewards"
#     elif seg == "Potential":
#         rec = "Offer discounts, retarget ads"
#     else:
#         rec = "Improve UX, aggressive retargeting"

#     st.info(rec)

# # ============================================
# # 🧪 A/B TESTING
# # ============================================
# with tab5:
#     st.subheader("🧪 Campaign A/B Testing")

#     st.write("Compare two marketing strategies")

#     col1, col2 = st.columns(2)

#     with col1:
#         st.write("Campaign A")
#         a_discount = st.slider("Discount A", 0, 50, 10)

#     with col2:
#         st.write("Campaign B")
#         b_discount = st.slider("Discount B", 0, 50, 20)

#     # Simulated uplift
#     conv_a = df["Revenue"].mean() + a_discount * 0.001
#     conv_b = df["Revenue"].mean() + b_discount * 0.001

#     result = pd.DataFrame({
#         "Campaign": ["A", "B"],
#         "Conversion": [conv_a, conv_b]
#     })

#     fig = px.bar(result, x="Campaign", y="Conversion",
#                  title="A/B Test Result")
#     st.plotly_chart(fig, width="stretch")

#     if conv_a > conv_b:
#         st.success("Campaign A performs better")
#     else:
#         st.success("Campaign B performs better")
import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import plotly.express as px
import matplotlib.pyplot as plt

from lime.lime_tabular import LimeTabularExplainer
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")

# ============================================
# CONFIG
# ============================================
st.set_page_config(layout="wide")
st.title("🧠 Self-Learning Explainable AI Marketing System")

# ============================================
# LOAD DATA
# ============================================
uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv", "xlsx"])

def load_data(file):
    if file:
        return pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
    return pd.read_csv("online_shoppers_intention.csv")

df = load_data(uploaded_file)
df["Revenue"] = df["Revenue"].astype(int)

# ============================================
# FILTERS
# ============================================
st.sidebar.header("🎛 Filters")

bounce_filter = st.sidebar.slider("Max Bounce Rate", 0.0, 1.0, 1.0)
exit_filter = st.sidebar.slider("Max Exit Rate", 0.0, 1.0, 1.0)

df = df[(df["BounceRates"] <= bounce_filter) &
        (df["ExitRates"] <= exit_filter)]

# ============================================
# PREPROCESS
# ============================================
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("Revenue", axis=1)
y = df_encoded["Revenue"]

# ============================================
# AUTO LEARNING
# ============================================
st.sidebar.header("🤖 Auto Learning")

retrain = st.sidebar.checkbox("Enable Auto Learning")

if retrain:
    st.sidebar.info("Training new model...")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump(model, "model.pkl")

    st.sidebar.success("Model retrained successfully!")
else:
    model = joblib.load("model.pkl")

# ============================================
# SHAP + LIME (UPDATED EVERY TIME)
# ============================================
explainer = shap.TreeExplainer(model)

sample_X = X.sample(min(200, len(X)))
shap_values = explainer.shap_values(sample_X)

if isinstance(shap_values, list):
    shap_values = shap_values[1]

lime_explainer = LimeTabularExplainer(
    training_data=X.values,
    feature_names=X.columns.tolist(),
    class_names=["No Purchase", "Purchase"],
    mode="classification"
)

# ============================================
# TABS
# ============================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "📈 Analytics",
    "🧠 Explainability",
    "🎯 Marketing AI",
    "📊 KPI Dashboard"
])

# ============================================
# OVERVIEW
# ============================================
with tab1:
    st.subheader("📊 Executive Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    total_users = len(df)
    conversions = df["Revenue"].sum()
    conv_rate = df["Revenue"].mean()

    c1.metric("Users", total_users)
    c2.metric("Conversions", conversions)
    c3.metric("Conversion Rate", f"{conv_rate*100:.2f}%")
    c4.metric("Avg Engagement", f"{df['ProductRelated_Duration'].mean():.2f}")

    fig = px.pie(df, names="Revenue", title="Conversion Split")
    st.plotly_chart(fig, width="stretch")

# ============================================
# ANALYTICS
# ============================================
with tab2:
    st.subheader("📈 Customer Behavior")

    fig = px.scatter(df, x="PageValues", y="ProductRelated_Duration", color="Revenue")
    st.plotly_chart(fig, width="stretch")

    corr = df.select_dtypes(include=np.number).corr()
    fig = px.imshow(corr, text_auto=True)
    st.plotly_chart(fig, width="stretch")

    # Forecast
    st.subheader("📅 Sales Forecast")

    df["Time"] = np.arange(len(df))
    ts = df.groupby("Time")["Revenue"].sum()

    model_arima = ARIMA(ts, order=(1,1,1)).fit()
    forecast = model_arima.forecast(steps=10)

    actual_x = list(range(len(ts)))
    forecast_x = list(range(len(ts), len(ts)+len(forecast)))

    fig = px.line(x=actual_x, y=ts, title="Actual vs Forecast")
    fig.add_scatter(x=forecast_x, y=forecast, mode='lines', name='Forecast')

    st.plotly_chart(fig, width="stretch")

# ============================================
# EXPLAINABILITY
# ============================================
with tab3:
    st.subheader("🧠 SHAP Global Importance")

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, sample_X, plot_type="bar", show=False)
    st.pyplot(fig)

    st.subheader("🧪 LIME Example")

    sample_instance = X.iloc[0]

    exp = lime_explainer.explain_instance(
        sample_instance.values,
        model.predict_proba,
        num_features=5
    )

    for f, w in exp.as_list():
        st.write(f"{f} → {w:.3f}")

# ============================================
# MARKETING AI ENGINE
# ============================================
with tab4:
    st.subheader("🎯 Smart Marketing Engine")

    pv = st.slider("Intent Score", 0.0, 300.0, 20.0)
    br = st.slider("Bounce Rate", 0.0, 1.0, 0.2)
    er = st.slider("Exit Rate", 0.0, 1.0, 0.2)
    dur = st.slider("Engagement", 0.0, 500.0, 50.0)

    input_data = pd.DataFrame(np.zeros((1, X.shape[1])), columns=X.columns)

    for col, val in {
        "PageValues": pv,
        "BounceRates": br,
        "ExitRates": er,
        "ProductRelated_Duration": dur
    }.items():
        if col in input_data.columns:
            input_data[col] = val

    prob = model.predict_proba(input_data)[0][1]

    st.progress(int(prob * 100))
    st.write(f"Purchase Probability: {prob:.2f}")

    # Segment
    if prob > 0.7:
        seg = "High Value"
        strategy = "Upsell premium products + loyalty rewards"
    elif prob > 0.4:
        seg = "Potential"
        strategy = "Discount + retargeting ads"
    else:
        seg = "At Risk"
        strategy = "Improve UX + aggressive marketing"

    st.success(f"Segment: {seg}")
    st.info(strategy)

# ============================================
# KPI DASHBOARD
# ============================================
with tab5:
    st.subheader("📊 Marketing KPIs")

    revenue = df["Revenue"].sum()
    customers = len(df)

    # Simulated KPIs
    avg_order_value = revenue / max(customers,1)
    cac = np.random.uniform(5, 20)
    roi = (revenue - cac*customers) / max(cac*customers,1)

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Order Value", f"{avg_order_value:.2f}")
    c2.metric("Customer Acquisition Cost", f"{cac:.2f}")
    c3.metric("ROI", f"{roi:.2f}")

    # KPI trend
    kpi_df = pd.DataFrame({
        "Metric": ["AOV", "CAC", "ROI"],
        "Value": [avg_order_value, cac, roi]
    })

    fig = px.bar(kpi_df, x="Metric", y="Value", title="KPI Overview")
    st.plotly_chart(fig, width="stretch")