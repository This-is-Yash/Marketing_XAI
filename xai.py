
import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import os

from lime.lime_tabular import LimeTabularExplainer
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")
st.set_page_config(layout="wide", page_title="XAI Marketing Dashboard")

st.title("Explainable AI Marketing Dashboard")

@st.cache_data
def load_data(file):
    if file:
        return pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
    return pd.read_csv("online_shoppers_intention.csv")

uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv", "xlsx"])
df = load_data(uploaded_file)

df["Revenue"] = df["Revenue"].astype(int)

st.sidebar.header("🎛 Filters")

bounce_filter = st.sidebar.slider("Max Bounce Rate", 0.0, 1.0, 1.0)
exit_filter = st.sidebar.slider("Max Exit Rate", 0.0, 1.0, 1.0)

df = df[(df["BounceRates"] <= bounce_filter) &
        (df["ExitRates"] <= exit_filter)]

df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("Revenue", axis=1)
y = df_encoded["Revenue"]

@st.cache_resource
def load_model(X, y):
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = load_model(X, y)

if st.sidebar.button("Retrain Model"):
    model.fit(X, y)
    joblib.dump(model, "model.pkl")
    st.sidebar.success("Model retrained!")

@st.cache_resource
def get_shap_values(_model, X):
    explainer = shap.TreeExplainer(_model)
    sample_X = X.sample(min(200, len(X)))

    shap_vals = explainer.shap_values(sample_X)

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    shap_vals = np.array(shap_vals)

    if len(shap_vals.shape) == 3:
        shap_vals = shap_vals[:, :, 1]

    return explainer, sample_X, shap_vals

explainer, sample_X, shap_values = get_shap_values(model, X)

lime_explainer = LimeTabularExplainer(
    training_data=X.values,
    feature_names=X.columns.tolist(),
    class_names=["No Purchase", "Purchase"],
    mode="classification"
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Analytics",
    "Explainability",
    "Marketing AI",
    "KPI Dashboard"
])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Users", len(df))
    c2.metric("Conversions", df["Revenue"].sum())
    c3.metric("Conversion Rate", f"{df['Revenue'].mean()*100:.2f}%")
    c4.metric("Avg Engagement", f"{df['ProductRelated_Duration'].mean():.2f}")

    fig = px.pie(df, names="Revenue", title="Conversion Split")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = px.scatter(df, x="PageValues", y="ProductRelated_Duration", color="Revenue")
    st.plotly_chart(fig, use_container_width=True)

    corr = df.select_dtypes(include=np.number).corr()
    fig = px.imshow(corr, text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

    # Forecast
    df["Time"] = np.arange(len(df))
    ts = df.groupby("Time")["Revenue"].sum()

    model_arima = ARIMA(ts, order=(1,1,1)).fit()
    forecast = model_arima.forecast(steps=10)

    fig = px.line(x=list(range(len(ts))), y=ts, title="Forecast")
    fig.add_scatter(x=list(range(len(ts), len(ts)+10)), y=forecast)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("SHAP Feature Importance")

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, sample_X, plot_type="bar", show=False)
    st.pyplot(fig)


    st.subheader("Top Features")

    importance = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X.columns
    ).sort_values(ascending=False).head(10)

    st.bar_chart(importance)

    # LIME
    st.subheader("LIME Explanation")

    exp = lime_explainer.explain_instance(
        sample_X.iloc[0].values,
        model.predict_proba,
        num_features=5
    )

    for f, w in exp.as_list():
        st.write(f"{f} → {w:.3f}")

with tab4:
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

with tab5:
    revenue = df["Revenue"].sum()
    customers = len(df)

    aov = revenue / max(customers,1)
    cac = np.random.uniform(5,20)
    roi = (revenue - cac*customers) / max(cac*customers,1)

    c1, c2, c3 = st.columns(3)
    c1.metric("AOV", f"{aov:.2f}")
    c2.metric("CAC", f"{cac:.2f}")
    c3.metric("ROI", f"{roi:.2f}")

    fig = px.bar(
        pd.DataFrame({"Metric":["AOV","CAC","ROI"],"Value":[aov,cac,roi]}),
        x="Metric",
        y="Value"
    )
    st.plotly_chart(fig, use_container_width=True)
