import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="🌸 Iris Classifier", layout="wide")

# Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
target_names = iris.target_names

# Sidebar
st.sidebar.title("🔧 Mode Selection")
mode = st.sidebar.radio("Choose mode", ("Prediction", "Explore Data"))

if st.sidebar.checkbox("Show raw data", False):
    st.sidebar.dataframe(df)

if mode == "Explore Data":
    st.markdown("## 📊 Iris Dataset - Data Exploration")
    st.markdown("Use the widgets below to explore features with **histograms** and **scatter plots**")

    # Histogram
    feature = st.selectbox("Select Feature for Histogram", df.columns[:-1])
    bins = st.slider("Bins", 5, 50, 20)
    fig1, ax1 = plt.subplots()
    ax1.hist(df[feature], bins=bins, color="#ff9999", edgecolor="black")
    ax1.set_xlabel(feature)
    ax1.set_title(f"Histogram of {feature}")
    st.pyplot(fig1)

    # Scatter plot
    x_feat = st.selectbox("X axis (scatter)", df.columns[:-1], index=0)
    y_feat = st.selectbox("Y axis (scatter)", df.columns[:-1], index=1)
    fig2, ax2 = plt.subplots()
    scatter = ax2.scatter(df[x_feat], df[y_feat], c=df['target'], cmap="viridis", s=100, alpha=0.7)
    ax2.set_xlabel(x_feat)
    ax2.set_ylabel(y_feat)
    ax2.set_title(f"{y_feat} vs {x_feat}")
    st.pyplot(fig2)

    st.markdown("**Legend:** 0=setosa 🌱, 1=versicolor 🌼, 2=virginica 🌸")
# Prediction 
else:
    st.markdown("## 🌸 Iris Species Prediction")
    st.markdown("Enter the flower measurements and click **Predict** to see the species.")

    col1, col2 = st.columns(2)

    sepal_length = col1.slider("Sepal length (cm)", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()), float(df['sepal length (cm)'].mean()), step=0.1)
    sepal_width = col2.slider("Sepal width (cm)", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()), float(df['sepal width (cm)'].mean()), step=0.1)
    petal_length = col1.slider("Petal length (cm)", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()), float(df['petal length (cm)'].mean()), step=0.1)
    petal_width = col2.slider("Petal width (cm)", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()), float(df['petal width (cm)'].mean()), step=0.1)
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Load model
    try:
        model = joblib.load("iris_rf_model.joblib")
    except Exception as e:
        st.error("Model file not found. Run train_model.py first to create iris_rf_model.joblib")
        st.stop()
    if st.button("Predict 🌟"):
        with st.spinner("Predicting the species..."):
            time.sleep(1.5)  # simulate processing time
            pred = model.predict(input_data)[0]
            pred_name = target_names[pred]
            proba = model.predict_proba(input_data)[0]
        if pred == 0:
           st.success(f"🌱 Predicted species: {pred_name} (Setosa)")
        elif pred == 1:
            st.info(f"🌼 Predicted species: {pred_name} (Versicolor)")
        else:
            st.warning(f"🌸 Predicted species: {pred_name} (Virginica)")
        # Show prediction probabilities as bar chart
        prob_df = pd.DataFrame([proba], columns=target_names)
        st.markdown("### 🔹 Prediction Probabilities")
        st.bar_chart(prob_df.T.rename(columns={0: "probability"}))

    # Option to show the sample rows
    if st.checkbox("Show sample rows from dataset"):
        st.write(df.head())
