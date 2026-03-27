# # # # ============================================
# # # # ALL-IN-ONE INTELLIGENT WEB ANALYTICS DASHBOARD
# # # # ============================================

# # # import streamlit as st
# # # import pandas as pd
# # # import numpy as np
# # # import shap
# # # import joblib
# # # import plotly.express as px
# # # import matplotlib.pyplot as plt
# # # from lime.lime_tabular import LimeTabularExplainer

# # # # ============================================
# # # # CONFIG
# # # # ============================================
# # # st.set_page_config(layout="wide")
# # # st.title("🧠 Intelligent Web Analytics Dashboard (Explainable AI)")

# # # # ============================================
# # # # FILE UPLOAD
# # # # ============================================
# # # st.sidebar.header("📂 Upload Dataset")

# # # uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

# # # if uploaded_file:
# # #     if uploaded_file.name.endswith(".csv"):
# # #         df = pd.read_csv(uploaded_file)
# # #     else:
# # #         df = pd.read_excel(uploaded_file)
# # # else:
# # #     df = pd.read_csv("online_shoppers_intention.csv")

# # # df['Revenue'] = df['Revenue'].astype(int)

# # # st.subheader("📊 Dataset Preview")
# # # st.dataframe(df.head())

# # # # ============================================
# # # # PREPROCESSING
# # # # ============================================
# # # df_encoded = pd.get_dummies(df, drop_first=True)
# # # X = df_encoded.drop("Revenue", axis=1)
# # # y = df_encoded["Revenue"]

# # # # ============================================
# # # # LOAD MODEL
# # # # ============================================
# # # model = joblib.load("model.pkl")

# # # # ============================================
# # # # SHAP
# # # # ============================================
# # # explainer = shap.TreeExplainer(model)
# # # shap_values = explainer.shap_values(X)

# # # if isinstance(shap_values, list):
# # #     shap_values_to_use = shap_values[1]
# # # else:
# # #     shap_values_to_use = shap_values

# # # # ============================================
# # # # KPI SECTION (LIKE POWER BI)
# # # # ============================================
# # # st.subheader("📈 Key Metrics")

# # # col1, col2, col3 = st.columns(3)

# # # col1.metric("Total Sessions", len(df))
# # # col2.metric("Conversion Rate", f"{df['Revenue'].mean()*100:.2f}%")
# # # col3.metric("Avg Page Value", f"{df['PageValues'].mean():.2f}")

# # # # ============================================
# # # # INTERACTIVE VISUALS
# # # # ============================================
# # # st.subheader("📊 Interactive Analytics")

# # # col1, col2 = st.columns(2)

# # # with col1:
# # #     fig = px.histogram(df, x="PageValues", color="Revenue", title="Page Value Distribution")
# # #     st.plotly_chart(fig)

# # # with col2:
# # #     fig = px.scatter(df, x="ExitRates", y="BounceRates", color="Revenue",
# # #                      title="Exit vs Bounce Rate")
# # #     st.plotly_chart(fig)

# # # # ============================================
# # # # USER INPUT SIMULATION
# # # # ============================================
# # # st.sidebar.header("🎯 Simulate User")

# # # page_value = st.sidebar.slider("PageValues", 0.0, 300.0, 20.0)
# # # exit_rate = st.sidebar.slider("ExitRates", 0.0, 1.0, 0.2)
# # # bounce_rate = st.sidebar.slider("BounceRates", 0.0, 1.0, 0.2)
# # # product_duration = st.sidebar.slider("ProductRelated_Duration", 0.0, 500.0, 50.0)

# # # input_data = X.iloc[0:1].copy()
# # # input_data[:] = 0

# # # input_data["PageValues"] = page_value
# # # input_data["ExitRates"] = exit_rate
# # # input_data["BounceRates"] = bounce_rate
# # # input_data["ProductRelated_Duration"] = product_duration

# # # # ============================================
# # # # PREDICTION
# # # # ============================================
# # # prediction = model.predict(input_data)[0]
# # # prob = model.predict_proba(input_data)[0][1]

# # # st.subheader("🔮 Prediction")

# # # if prediction == 1:
# # #     st.success(f"Likely to Purchase ({prob:.2f})")
# # # else:
# # #     st.error(f"Not Likely to Purchase ({prob:.2f})")

# # # # ============================================
# # # # SHAP GLOBAL
# # # # ============================================
# # # st.subheader("📊 Feature Importance (SHAP)")

# # # fig2, ax2 = plt.subplots()
# # # shap.summary_plot(shap_values_to_use, X, plot_type="bar", show=False)
# # # st.pyplot(fig2)
# # # # st.plotly_chart(fig, width="stretch")
# # # # ============================================
# # # # SHAP LOCAL
# # # # ============================================
# # # st.subheader("🧍 Individual Explanation")

# # # input_shap = explainer.shap_values(input_data)

# # # if isinstance(input_shap, list):
# # #     shap_val = input_shap[1]
# # #     base_val = explainer.expected_value[1]
# # # else:
# # #     shap_val = input_shap
# # #     base_val = explainer.expected_value

# # # fig3 = plt.figure()
# # # shap.plots.waterfall(
# # #     shap.Explanation(
# # #         values=shap_val[0],
# # #         base_values=base_val,
# # #         data=input_data.iloc[0],
# # #         feature_names=input_data.columns
# # #     ),
# # #     show=False
# # # )
# # # st.pyplot(fig3)

# # # # ============================================
# # # # LIME
# # # # ============================================
# # # st.subheader("🧪 LIME Explanation")

# # # lime_explainer = LimeTabularExplainer(
# # #     X.values,
# # #     feature_names=X.columns.tolist(),
# # #     class_names=["No", "Yes"],
# # #     mode="classification"
# # # )

# # # lime_exp = lime_explainer.explain_instance(
# # #     input_data.values[0],
# # #     model.predict_proba,
# # #     num_features=5
# # # )

# # # for f, w in lime_exp.as_list():
# # #     st.write(f"{f} → {w:.3f}")

# # # # ============================================
# # # # INTELLIGENT INSIGHTS (RULE-BASED LLM)
# # # # ============================================
# # # st.subheader("💡 AI Business Insights")

# # # insights = []

# # # if page_value > 50:
# # #     insights.append("High Page Value → strong buying intent")
# # # else:
# # #     insights.append("Improve product pages to increase value")

# # # if exit_rate > 0.3:
# # #     insights.append("Users dropping → improve UX")
# # # else:
# # #     insights.append("Good engagement")

# # # if bounce_rate > 0.3:
# # #     insights.append("Landing page issue")
# # # else:
# # #     insights.append("Landing page performing well")

# # # if product_duration > 100:
# # #     insights.append("High engagement")
# # # else:
# # #     insights.append("Increase recommendations")

# # # for i in insights:
# # #     st.write("•", i)

# # # # ============================================
# # # # FINAL DECISION
# # # # ============================================
# # # st.subheader("🎯 Recommended Action")

# # # if prediction == 1:
# # #     st.success("Target with premium offers")
# # # else:
# # #     st.warning("Retarget with ads & discounts")

# # # # ============================================
# # # # EXPORT DATA
# # # # ============================================
# # # st.subheader("📁 Export Data")

# # # output = input_data.copy()
# # # output["Prediction"] = prediction
# # # output["Probability"] = prob
# # # output["Insights"] = " | ".join(insights)

# # # # Save files
# # # output.to_csv("output.csv", index=False)
# # # output.to_excel("output.xlsx", index=False)

# # # st.download_button("Download CSV", output.to_csv(index=False), "output.csv")
# # # st.download_button("Download Excel", output.to_excel(index=False), "output.xlsx")
# # # ============================================
# # # ALL-IN-ONE INTELLIGENT WEB ANALYTICS DASHBOARD
# # # ============================================

# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # import shap
# # import joblib
# # import plotly.express as px
# # import matplotlib.pyplot as plt
# # from lime.lime_tabular import LimeTabularExplainer
# # import io

# # # ============================================
# # # CONFIG
# # # ============================================
# # st.set_page_config(layout="wide")
# # st.title("🧠 Intelligent Web Analytics Dashboard (Explainable AI)")

# # # ============================================
# # # FILE UPLOAD
# # # ============================================
# # st.sidebar.header("📂 Upload Dataset")

# # uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

# # @st.cache_data
# # def load_data(file):
# #     if file:
# #         if file.name.endswith(".csv"):
# #             return pd.read_csv(file)
# #         else:
# #             return pd.read_excel(file)
# #     return pd.read_csv("online_shoppers_intention.csv")

# # df = load_data(uploaded_file)

# # # Ensure correct type
# # df["Revenue"] = df["Revenue"].astype(int)

# # st.subheader("📊 Dataset Preview")
# # st.dataframe(df.head())

# # # ============================================
# # # PREPROCESSING
# # # ============================================
# # df_encoded = pd.get_dummies(df, drop_first=True)

# # if "Revenue" not in df_encoded.columns:
# #     st.error("❌ 'Revenue' column missing after encoding")
# #     st.stop()

# # X = df_encoded.drop("Revenue", axis=1)
# # y = df_encoded["Revenue"]

# # # ============================================
# # # LOAD MODEL
# # # ============================================
# # @st.cache_resource
# # def load_model():
# #     return joblib.load("model.pkl")

# # model = load_model()

# # # ============================================
# # # SHAP (Optimized)
# # # ============================================
# # # @st.cache_resource
# # # def get_explainer(model):
# # #     return shap.TreeExplainer(model)
# # @st.cache_resource
# # def get_explainer(_model):
# #     return shap.TreeExplainer(_model)

# # explainer = get_explainer(model)

# # @st.cache_data
# # def compute_shap(X_sample):
# #     shap_vals = explainer.shap_values(X_sample)
# #     if isinstance(shap_vals, list):
# #         return shap_vals[1]
# #     return shap_vals

# # sample_X = X.sample(min(200, len(X)))
# # shap_values = compute_shap(sample_X)

# # # ============================================
# # # KPI SECTION
# # # ============================================
# # st.subheader("📈 Key Metrics")

# # col1, col2, col3 = st.columns(3)

# # col1.metric("Total Sessions", len(df))
# # col2.metric("Conversion Rate", f"{df['Revenue'].mean()*100:.2f}%")
# # col3.metric("Avg Page Value", f"{df['PageValues'].mean():.2f}")

# # # ============================================
# # # VISUALS
# # # ============================================
# # st.subheader("📊 Interactive Analytics")

# # col1, col2 = st.columns(2)

# # with col1:
# #     fig1 = px.histogram(df, x="PageValues", color="Revenue",
# #                         title="Page Value Distribution")
# #     st.plotly_chart(fig1, width="stretch")

# # with col2:
# #     fig2 = px.scatter(df, x="ExitRates", y="BounceRates",
# #                       color="Revenue", title="Exit vs Bounce Rate")
# #     st.plotly_chart(fig2, width="stretch")

# # # ============================================
# # # USER INPUT
# # # ============================================
# # st.sidebar.header("🎯 Simulate User")

# # page_value = st.sidebar.slider("PageValues", 0.0, 300.0, 20.0)
# # exit_rate = st.sidebar.slider("ExitRates", 0.0, 1.0, 0.2)
# # bounce_rate = st.sidebar.slider("BounceRates", 0.0, 1.0, 0.2)
# # product_duration = st.sidebar.slider("ProductRelated_Duration", 0.0, 500.0, 50.0)

# # # ✅ FIXED input creation
# # input_data = pd.DataFrame(
# #     np.zeros((1, X.shape[1])),
# #     columns=X.columns
# # )

# # # Assign safely
# # for col, val in {
# #     "PageValues": page_value,
# #     "ExitRates": exit_rate,
# #     "BounceRates": bounce_rate,
# #     "ProductRelated_Duration": product_duration
# # }.items():
# #     if col in input_data.columns:
# #         input_data[col] = val

# # # ============================================
# # # PREDICTION
# # # ============================================
# # prediction = model.predict(input_data)[0]
# # prob = model.predict_proba(input_data)[0][1]

# # st.subheader("🔮 Prediction")

# # if prediction == 1:
# #     st.success(f"Likely to Purchase ({prob:.2f})")
# # else:
# #     st.error(f"Not Likely to Purchase ({prob:.2f})")

# # # ============================================
# # # SHAP GLOBAL
# # # ============================================
# # st.subheader("📊 Feature Importance (SHAP)")

# # fig3, ax = plt.subplots()
# # shap.summary_plot(shap_values, sample_X, plot_type="bar", show=False)
# # st.pyplot(fig3)

# # # ============================================
# # # SHAP LOCAL
# # # ============================================
# # st.subheader("🧍 Individual Explanation")

# # input_shap = explainer.shap_values(input_data)

# # if isinstance(input_shap, list):
# #     shap_val = input_shap[1]
# #     base_val = explainer.expected_value[1]
# # else:
# #     shap_val = input_shap
# #     base_val = explainer.expected_value

# # fig4 = plt.figure()
# # shap.plots.waterfall(
# #     shap.Explanation(
# #         values=shap_val[0],
# #         base_values=base_val,
# #         data=input_data.iloc[0],
# #         feature_names=input_data.columns
# #     ),
# #     show=False
# # )
# # st.pyplot(fig4)

# # # ============================================
# # # LIME
# # # ============================================
# # st.subheader("🧪 LIME Explanation")

# # lime_explainer = LimeTabularExplainer(
# #     X.values,
# #     feature_names=X.columns.tolist(),
# #     class_names=["No", "Yes"],
# #     mode="classification"
# # )

# # lime_exp = lime_explainer.explain_instance(
# #     input_data.values[0],
# #     model.predict_proba,
# #     num_features=5
# # )

# # for f, w in lime_exp.as_list():
# #     st.write(f"{f} → {w:.3f}")

# # # ============================================
# # # INSIGHTS
# # # ============================================
# # st.subheader("💡 AI Business Insights")

# # insights = []

# # insights.append("High buying intent" if page_value > 50 else "Improve product pages")
# # insights.append("Improve UX" if exit_rate > 0.3 else "Good engagement")
# # insights.append("Landing page issue" if bounce_rate > 0.3 else "Landing page OK")
# # insights.append("High engagement" if product_duration > 100 else "Increase interaction")

# # for i in insights:
# #     st.write("•", i)

# # # ============================================
# # # FINAL DECISION
# # # ============================================
# # st.subheader("🎯 Recommended Action")

# # if prediction == 1:
# #     st.success("Target with premium offers")
# # else:
# #     st.warning("Retarget with ads & discounts")

# # # ============================================
# # # EXPORT
# # # ============================================
# # st.subheader("📁 Export Data")

# # output = input_data.copy()
# # output["Prediction"] = prediction
# # output["Probability"] = prob
# # output["Insights"] = " | ".join(insights)

# # # CSV download
# # st.download_button(
# #     "Download CSV",
# #     output.to_csv(index=False),
# #     file_name="output.csv"
# # )

# # # ✅ FIXED Excel download
# # buffer = io.BytesIO()
# # with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
# #     output.to_excel(writer, index=False)

# # st.download_button(
# #     "Download Excel",
# #     buffer.getvalue(),
# #     file_name="output.xlsx"
# # )
# # ============================================
# # ALL-IN-ONE INTELLIGENT WEB ANALYTICS DASHBOARD
# # ============================================

# import streamlit as st
# import pandas as pd
# import numpy as np
# import shap
# import joblib
# import plotly.express as px
# import matplotlib.pyplot as plt
# from lime.lime_tabular import LimeTabularExplainer
# import io

# # ============================================
# # CONFIG
# # ============================================
# st.set_page_config(layout="wide")
# st.title("🧠 Intelligent Web Analytics Dashboard (Explainable AI)")

# # ============================================
# # LOAD DATA
# # ============================================
# st.sidebar.header("📂 Upload Dataset")

# uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

# def load_data(file):
#     if file:
#         if file.name.endswith(".csv"):
#             return pd.read_csv(file)
#         else:
#             return pd.read_excel(file)
#     return pd.read_csv("online_shoppers_intention.csv")

# df = load_data(uploaded_file)

# # Ensure Revenue exists
# if "Revenue" not in df.columns:
#     st.error("❌ Dataset must contain 'Revenue' column")
#     st.stop()

# df["Revenue"] = df["Revenue"].astype(int)

# st.subheader("📊 Dataset Preview")
# st.dataframe(df.head())

# # ============================================
# # PREPROCESSING
# # ============================================
# df_encoded = pd.get_dummies(df, drop_first=True)

# if "Revenue" not in df_encoded.columns:
#     st.error("❌ 'Revenue' missing after encoding")
#     st.stop()

# X = df_encoded.drop("Revenue", axis=1)
# y = df_encoded["Revenue"]

# # ============================================
# # LOAD MODEL
# # ============================================
# model = joblib.load("model.pkl")

# # ============================================
# # SHAP (NO CACHE = NO ERROR)
# # ============================================
# explainer = shap.TreeExplainer(model)

# # Sample for speed
# sample_X = X.sample(min(200, len(X)))

# shap_values = explainer.shap_values(sample_X)
# if isinstance(shap_values, list):
#     shap_values = shap_values[1]

# # ============================================
# # KPI
# # ============================================
# st.subheader("📈 Key Metrics")

# col1, col2, col3 = st.columns(3)

# col1.metric("Total Sessions", len(df))
# col2.metric("Conversion Rate", f"{df['Revenue'].mean()*100:.2f}%")
# col3.metric("Avg Page Value", f"{df['PageValues'].mean():.2f}")

# # ============================================
# # VISUALS
# # ============================================
# st.subheader("📊 Interactive Analytics")

# col1, col2 = st.columns(2)

# with col1:
#     fig1 = px.histogram(df, x="PageValues", color="Revenue")
#     st.plotly_chart(fig1, width="stretch")

# with col2:
#     fig2 = px.scatter(df, x="ExitRates", y="BounceRates", color="Revenue")
#     st.plotly_chart(fig2, width="stretch")

# # ============================================
# # USER INPUT
# # ============================================
# st.sidebar.header("🎯 Simulate User")

# page_value = st.sidebar.slider("PageValues", 0.0, 300.0, 20.0)
# exit_rate = st.sidebar.slider("ExitRates", 0.0, 1.0, 0.2)
# bounce_rate = st.sidebar.slider("BounceRates", 0.0, 1.0, 0.2)
# product_duration = st.sidebar.slider("ProductRelated_Duration", 0.0, 500.0, 50.0)

# # FIXED INPUT
# input_data = pd.DataFrame(
#     np.zeros((1, X.shape[1])),
#     columns=X.columns
# )

# for col, val in {
#     "PageValues": page_value,
#     "ExitRates": exit_rate,
#     "BounceRates": bounce_rate,
#     "ProductRelated_Duration": product_duration
# }.items():
#     if col in input_data.columns:
#         input_data[col] = val

# # ============================================
# # PREDICTION
# # ============================================
# prediction = model.predict(input_data)[0]
# prob = model.predict_proba(input_data)[0][1]

# st.subheader("🔮 Prediction")

# if prediction == 1:
#     st.success(f"Likely to Purchase ({prob:.2f})")
# else:
#     st.error(f"Not Likely to Purchase ({prob:.2f})")

# # ============================================
# # SHAP GLOBAL
# # ============================================
# st.subheader("📊 Feature Importance (SHAP)")

# fig, ax = plt.subplots()
# shap.summary_plot(shap_values, sample_X, plot_type="bar", show=False)
# st.pyplot(fig)

# # ============================================
# # SHAP LOCAL
# # ============================================
# st.subheader("🧍 Individual Explanation")

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
# # LIME
# # ============================================
# st.subheader("🧪 LIME Explanation")

# lime_explainer = LimeTabularExplainer(
#     X.values,
#     feature_names=X.columns.tolist(),
#     class_names=["No", "Yes"],
#     mode="classification"
# )

# lime_exp = lime_explainer.explain_instance(
#     input_data.values[0],
#     model.predict_proba,
#     num_features=5
# )

# for f, w in lime_exp.as_list():
#     st.write(f"{f} → {w:.3f}")

# # ============================================
# # INSIGHTS
# # ============================================
# st.subheader("💡 AI Business Insights")

# insights = []

# insights.append("High buying intent" if page_value > 50 else "Improve product pages")
# insights.append("Improve UX" if exit_rate > 0.3 else "Good engagement")
# insights.append("Landing page issue" if bounce_rate > 0.3 else "Landing page OK")
# insights.append("High engagement" if product_duration > 100 else "Increase interaction")

# for i in insights:
#     st.write("•", i)

# # ============================================
# # FINAL DECISION
# # ============================================
# st.subheader("🎯 Recommended Action")

# if prediction == 1:
#     st.success("Target with premium offers")
# else:
#     st.warning("Retarget with ads & discounts")

# # ============================================
# # EXPORT
# # ============================================
# st.subheader("📁 Export Data")

# output = input_data.copy()
# output["Prediction"] = prediction
# output["Probability"] = prob
# output["Insights"] = " | ".join(insights)

# # CSV
# st.download_button(
#     "Download CSV",
#     output.to_csv(index=False),
#     file_name="output.csv"
# )

# # Excel (FIXED)
# buffer = io.BytesIO()
# with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
#     output.to_excel(writer, index=False)

# st.download_button(
#     "Download Excel",
#     buffer.getvalue(),
#     file_name="output.xlsx"
# )
# ============================================
# EXPLAINABLE AI FOR MARKETING DECISION SYSTEM
# WITH LOCAL LLM-STYLE RECOMMENDER
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import io

# ============================================
# CONFIG
# ============================================
st.set_page_config(layout="wide")
st.title("🧠 Marketing Decision Intelligence System (Explainable AI + LLM)")

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
# PREPROCESS
# ============================================
df_encoded = pd.get_dummies(df, drop_first=True)
X = df_encoded.drop("Revenue", axis=1)
y = df_encoded["Revenue"]

model = joblib.load("model.pkl")

# ============================================
# KPI SECTION (MARKETING)
# ============================================
st.subheader("📊 Marketing KPIs")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Visitors", len(df))
col2.metric("Conversion Rate", f"{y.mean()*100:.2f}%")
col3.metric("Avg Intent Score", f"{df['PageValues'].mean():.2f}")
col4.metric("Avg Engagement Time", f"{df['ProductRelated_Duration'].mean():.2f}")

# ============================================
# ADVANCED VISUALS
# ============================================
st.subheader("📈 Marketing Analytics Dashboard")

c1, c2 = st.columns(2)

with c1:
    fig = px.box(df, x="Revenue", y="PageValues",
                 title="Intent Score vs Conversion")
    st.plotly_chart(fig, width="stretch")

with c2:
    fig = px.violin(df, y="BounceRates", x="Revenue",
                    title="Bounce Behavior Analysis")
    st.plotly_chart(fig, width="stretch")

# Funnel-style chart
st.subheader("🛒 Conversion Funnel Approximation")

funnel_data = pd.DataFrame({
    "Stage": ["Visited", "Engaged", "High Intent", "Converted"],
    "Users": [
        len(df),
        len(df[df["BounceRates"] < 0.5]),
        len(df[df["PageValues"] > 20]),
        len(df[df["Revenue"] == 1])
    ]
})

fig = px.funnel(funnel_data, x="Users", y="Stage")
st.plotly_chart(fig, width="stretch")

# ============================================
# USER INPUT
# ============================================
st.sidebar.header("🎯 Simulate Customer")

page_value = st.sidebar.slider("Intent Score", 0.0, 300.0, 20.0)
exit_rate = st.sidebar.slider("Exit Rate", 0.0, 1.0, 0.2)
bounce_rate = st.sidebar.slider("Bounce Rate", 0.0, 1.0, 0.2)
duration = st.sidebar.slider("Engagement Time", 0.0, 500.0, 50.0)

input_data = pd.DataFrame(np.zeros((1, X.shape[1])), columns=X.columns)

for col, val in {
    "PageValues": page_value,
    "ExitRates": exit_rate,
    "BounceRates": bounce_rate,
    "ProductRelated_Duration": duration
}.items():
    if col in input_data.columns:
        input_data[col] = val

# ============================================
# PREDICTION
# ============================================
prob = model.predict_proba(input_data)[0][1]

st.subheader("🔮 Purchase Probability")
st.progress(int(prob * 100))
st.write(f"Probability: {prob:.2f}")

# ============================================
# SEGMENTATION
# ============================================
if prob > 0.75:
    segment = "High Value"
elif prob > 0.4:
    segment = "Nurture Lead"
else:
    segment = "At Risk"

st.info(f"Customer Segment: {segment}")

# ============================================
# SHAP
# ============================================
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X.sample(100))

st.subheader("📊 Conversion Drivers")

fig, ax = plt.subplots()
shap.summary_plot(shap_values[1] if isinstance(shap_values, list) else shap_values,
                  X.sample(100), plot_type="bar", show=False)
st.pyplot(fig)

# ============================================
# LIME
# ============================================
st.subheader("🧪 Why this user?")

lime_exp = LimeTabularExplainer(
    X.values,
    feature_names=X.columns.tolist(),
    class_names=["No", "Yes"],
    mode="classification"
)

exp = lime_exp.explain_instance(input_data.values[0],
                                model.predict_proba,
                                num_features=5)

for f, w in exp.as_list():
    st.write(f"{f}: {w:.3f}")

# ============================================
# 🧠 LOCAL LLM-STYLE RECOMMENDER (NO API)
# ============================================

def marketing_llm(segment, prob, page_value, bounce_rate, duration):
    """Simulates LLM reasoning using dynamic templates"""

    if segment == "High Value":
        return f"""
        🎯 Strategy:
        - Target with premium bundles
        - Upsell high-margin products
        - Offer loyalty rewards

        📊 Insight:
        High intent score ({page_value}) and strong engagement ({duration}) indicate readiness to purchase.

        💡 Action:
        Use personalized email campaigns and exclusive offers.
        """

    elif segment == "Nurture Lead":
        return f"""
        🎯 Strategy:
        - Provide limited-time discounts
        - Show product recommendations
        - Use retargeting ads

        📊 Insight:
        Moderate intent but bounce rate ({bounce_rate}) suggests hesitation.

        💡 Action:
        Improve landing page clarity and trust signals.
        """

    else:
        return f"""
        🎯 Strategy:
        - Retarget aggressively
        - Use push notifications
        - Improve UX/UI

        📊 Insight:
        High bounce ({bounce_rate}) and low engagement ({duration}) show weak interest.

        💡 Action:
        Optimize page speed and simplify navigation.
        """

# ============================================
# RECOMMENDATIONS
# ============================================
st.subheader("🤖 AI Marketing Recommendations")

recommendation = marketing_llm(segment, prob, page_value, bounce_rate, duration)
st.success(recommendation)

# ============================================
# EXPORT
# ============================================
output = input_data.copy()
output["Probability"] = prob
output["Segment"] = segment
output["Recommendation"] = recommendation

st.download_button("Download CSV", output.to_csv(index=False), "marketing_report.csv")