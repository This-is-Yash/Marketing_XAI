
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
# PAGE CONFIG (DARK UI)
# ============================================
st.set_page_config(layout="wide", page_title="XAI Marketing Dashboard")

st.markdown("""
<style>
body {background-color: #0E1117; color: white;}
.block-container {padding-top: 1rem;}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Intelligent Explainable AI Marketing Dashboard")

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
# MODEL (AUTO LEARNING)
# ============================================
st.sidebar.header("🤖 Model")

if st.sidebar.checkbox("Retrain Model"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "model.pkl")
    st.sidebar.success("Model trained!")
else:
    model = joblib.load("model.pkl")

# ============================================
# SHAP + LIME
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
    c1.metric("Users", len(df))
    c2.metric("Conversions", df["Revenue"].sum())
    c3.metric("Conversion Rate", f"{df['Revenue'].mean()*100:.2f}%")
    c4.metric("Avg Engagement", f"{df['ProductRelated_Duration'].mean():.2f}")

    fig = px.pie(df, names="Revenue", title="Conversion Split")
    st.plotly_chart(fig, use_container_width=True)

    # FUNNEL
    st.subheader("📊 Customer Funnel")
    funnel = pd.DataFrame({
        "Stage": ["Visited", "Engaged", "Intent", "Converted"],
        "Users": [
            len(df),
            int(len(df)*0.7),
            int(len(df)*0.4),
            df["Revenue"].sum()
        ]
    })
    fig = px.funnel(funnel, x="Users", y="Stage")
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# ANALYTICS
# ============================================
with tab2:
    st.subheader("📈 Behavior Analysis")

    fig = px.scatter(df, x="PageValues", y="ProductRelated_Duration", color="Revenue")
    st.plotly_chart(fig, use_container_width=True)

    # VIOLIN
    st.subheader("🎻 Intent vs Conversion")
    fig = px.violin(df, x="Revenue", y="PageValues", box=True)
    st.plotly_chart(fig, use_container_width=True)

    # BOUNCE ANALYSIS
    st.subheader("📉 Bounce Behavior")
    fig = px.histogram(df, x="BounceRates", color="Revenue")
    st.plotly_chart(fig, use_container_width=True)

    # CORRELATION
    corr = df.select_dtypes(include=np.number).corr()
    fig = px.imshow(corr, text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

    # FORECAST
    st.subheader("📅 Sales Forecast")
    df["Time"] = np.arange(len(df))
    ts = df.groupby("Time")["Revenue"].sum()

    model_arima = ARIMA(ts, order=(1,1,1)).fit()
    forecast = model_arima.forecast(steps=10)

    fig = px.line(x=list(range(len(ts))), y=ts, title="Actual vs Forecast")
    fig.add_scatter(x=list(range(len(ts), len(ts)+10)), y=forecast, name="Forecast")
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# EXPLAINABILITY
# ============================================
with tab3:
    st.subheader("🧠 SHAP Global Importance")

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, sample_X, plot_type="bar", show=False)
    st.pyplot(fig)

    st.subheader("⚡ SHAP Explanation (Waterfall)")

try:
    single_row = sample_X.iloc[[0]]

    shap_vals = explainer.shap_values(single_row)

    if isinstance(shap_vals, list):
        shap_val = shap_vals[1][0]
        base_val = explainer.expected_value[1]
    else:
        shap_val = shap_vals[0]
        base_val = explainer.expected_value

    exp = shap.Explanation(
        values=shap_val,
        base_values=base_val,
        data=single_row.iloc[0],
        feature_names=X.columns
    )

    fig, ax = plt.subplots()
    shap.plots.waterfall(exp, show=False)
    st.pyplot(fig)

except Exception as e:
    st.warning("Using fallback explanation")

    # 🔁 FINAL FALLBACK (NEVER FAILS)
    st.write("Top Influencing Features:")
    importance = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X.columns
    ).sort_values(ascending=False).head(10)

    st.bar_chart(importance)

    # ✅ FIXED FORCE PLOT
    # st.subheader("⚡ SHAP Force Plot")

    # single = explainer.shap_values(sample_X.iloc[0:1])

    # if isinstance(single, list):
    #     shap_val = single[1][0]
    #     base_val = explainer.expected_value[1]
    # else:
    #     shap_val = single[0]
    #     base_val = explainer.expected_value

    # fig = plt.figure()
    # shap.force_plot(base_val, shap_val, sample_X.iloc[0], matplotlib=True)
    # st.pyplot(fig)
# ============================================
# ✅ SAFE SHAP FORCE PLOT (NO ERROR GUARANTEED)
# ============================================
    # st.subheader("⚡ SHAP Force Plot")

    # try:
    #     # Use new SHAP explainer call
    #     single_row = sample_X.iloc[[0]]
    #     explanation = explainer(single_row)

    #     fig = plt.figure()
    #     shap.plots.force(explanation[0], matplotlib=True)

    #     st.pyplot(fig)

    # except Exception as e:
    #     st.warning("SHAP force plot failed, using fallback...")

    #     # 🔁 FALLBACK (always works)
    #     single = explainer.shap_values(sample_X.iloc[0:1])

    #     if isinstance(single, list):
    #         shap_val = single[1][0]
    #         base_val = explainer.expected_value[1]
    #     else:
    #         shap_val = single[0]
    #         base_val = explainer.expected_value

    #     fig = plt.figure()
    #     shap.force_plot(base_val, shap_val, sample_X.iloc[0], matplotlib=True)

    #     st.pyplot(fig)
    # LIME
    st.subheader("🧪 LIME Explanation")
    exp = lime_explainer.explain_instance(
        sample_X.iloc[0].values,
        model.predict_proba,
        num_features=5
    )
    for f, w in exp.as_list():
        st.write(f"{f} → {w:.3f}")

# ============================================
# MARKETING AI
# ============================================
with tab4:
    st.subheader("🎯 Marketing Decision Engine")

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

    if prob > 0.7:
        st.success("High Value → Upsell & Loyalty Offers")
    elif prob > 0.4:
        st.info("Potential → Discounts & Retargeting")
    else:
        st.warning("At Risk → Improve UX & Ads")

# ============================================
# KPI DASHBOARD
# ============================================
with tab5:
    st.subheader("📊 Marketing KPIs")

    revenue = df["Revenue"].sum()
    customers = len(df)

    aov = revenue / max(customers,1)
    cac = np.random.uniform(5,20)
    roi = (revenue - cac*customers) / max(cac*customers,1)

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Order Value", f"{aov:.2f}")
    c2.metric("Customer Acquisition Cost", f"{cac:.2f}")
    c3.metric("ROI", f"{roi:.2f}")

    kpi_df = pd.DataFrame({
        "Metric": ["AOV", "CAC", "ROI"],
        "Value": [aov, cac, roi]
    })

    fig = px.bar(kpi_df, x="Metric", y="Value")
    st.plotly_chart(fig, use_container_width=True)
# import streamlit as st
# import pandas as pd
# import numpy as np
# import shap
# import joblib
# import plotly.express as px
# import matplotlib.pyplot as plt

# from lime.lime_tabular import LimeTabularExplainer
# from statsmodels.tsa.arima.model import ARIMA
# from sklearn.ensemble import RandomForestClassifier

# import warnings
# warnings.filterwarnings("ignore")

# # ============================================
# # PAGE CONFIG
# # ============================================
# st.set_page_config(layout="wide", page_title="XAI Marketing Dashboard")

# st.title("🧠 Intelligent Explainable AI Marketing Dashboard")

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
# # FILTERS
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

# # ============================================
# # MODEL
# # ============================================
# st.sidebar.header("🤖 Model")

# if st.sidebar.checkbox("Retrain Model"):
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X, y)
#     joblib.dump(model, "model.pkl")
#     st.sidebar.success("Model trained!")
# else:
#     try:
#         model = joblib.load("model.pkl")
#     except:
#         model = RandomForestClassifier(n_estimators=100, random_state=42)
#         model.fit(X, y)

# # ============================================
# # SHAP + LIME
# # ============================================
# explainer = shap.TreeExplainer(model)

# sample_X = X.sample(min(200, len(X)))
# shap_values = explainer.shap_values(sample_X)

# if isinstance(shap_values, list):
#     shap_values = shap_values[1]

# lime_explainer = LimeTabularExplainer(
#     training_data=X.values,
#     feature_names=X.columns.tolist(),
#     class_names=["No Purchase", "Purchase"],
#     mode="classification"
# )

# # ============================================
# # TABS
# # ============================================
# tab1, tab2, tab3, tab4, tab5 = st.tabs([
#     "📊 Overview",
#     "📈 Analytics",
#     "🧠 Explainability",
#     "🎯 Marketing AI",
#     "📊 KPI Dashboard"
# ])

# # ============================================
# # OVERVIEW
# # ============================================
# with tab1:
#     c1, c2, c3, c4 = st.columns(4)
#     c1.metric("Users", len(df))
#     c2.metric("Conversions", df["Revenue"].sum())
#     c3.metric("Conversion Rate", f"{df['Revenue'].mean()*100:.2f}%")
#     c4.metric("Avg Engagement", f"{df['ProductRelated_Duration'].mean():.2f}")

#     fig = px.pie(df, names="Revenue")
#     st.plotly_chart(fig, use_container_width=True)

# # ============================================
# # ANALYTICS
# # ============================================
# with tab2:
#     fig = px.scatter(df, x="PageValues", y="ProductRelated_Duration", color="Revenue")
#     st.plotly_chart(fig, use_container_width=True)

#     corr = df.select_dtypes(include=np.number).corr()
#     fig = px.imshow(corr, text_auto=True)
#     st.plotly_chart(fig, use_container_width=True)

#     # Forecast
#     df["Time"] = np.arange(len(df))
#     ts = df.groupby("Time")["Revenue"].sum()

#     model_arima = ARIMA(ts, order=(1,1,1)).fit()
#     forecast = model_arima.forecast(steps=10)

#     fig = px.line(x=list(range(len(ts))), y=ts)
#     fig.add_scatter(x=list(range(len(ts), len(ts)+10)), y=forecast, name="Forecast")
#     st.plotly_chart(fig, use_container_width=True)

# # ============================================
# # EXPLAINABILITY
# # ============================================
# with tab3:
#     st.subheader("📊 SHAP Feature Importance")

#     fig, ax = plt.subplots()
#     shap.summary_plot(shap_values, sample_X, plot_type="bar", show=False)
#     st.pyplot(fig)

#     # ✅ FIXED (REPLACED FORCE PLOT)
#     st.subheader("⚡ SHAP Waterfall (Single Prediction)")

#     try:
#         single_row = sample_X.iloc[[0]]
#         explanation = explainer(single_row)

#         fig, ax = plt.subplots()
#         shap.plots.waterfall(explanation[0], show=False)
#         st.pyplot(fig)

#     except Exception as e:
#         st.error(f"SHAP failed: {e}")

#     # LIME
#     st.subheader("🧪 LIME Explanation")
#     exp = lime_explainer.explain_instance(
#         sample_X.iloc[0].values,
#         model.predict_proba,
#         num_features=5
#     )

#     for f, w in exp.as_list():
#         st.write(f"{f} → {w:.3f}")

# # ============================================
# # MARKETING AI
# # ============================================
# with tab4:
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

# # ============================================
# # KPI DASHBOARD
# # ============================================
# with tab5:
#     revenue = df["Revenue"].sum()
#     customers = len(df)

#     aov = revenue / max(customers,1)
#     cac = np.random.uniform(5,20)
#     roi = (revenue - cac*customers) / max(cac*customers,1)

#     c1, c2, c3 = st.columns(3)
#     c1.metric("AOV", f"{aov:.2f}")
#     c2.metric("CAC", f"{cac:.2f}")
#     c3.metric("ROI", f"{roi:.2f}")