import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from io import BytesIO

st.set_page_config(
    page_title="Data Analysis Dashboard",
    layout="wide",
    page_icon="📊"
)

st.title("📊Data Analysis Dashboard")
st.write("A powerful interactive dashboard for exploring and analyzing your dataset.")

uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
if uploaded:
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    st.success("Dataset loaded successfully!")

    # SIDEBAR NAVIGATION
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio("Go to:", [
        "📁 Data Preview",
        "📊 Statistics",
        "📈 Visualizations",
        "📉 Outlier Detection",
        "🔥 Correlation Explorer",
        "📅 Trend Detection",
        "🧹 Clean & Export",
    ])

    # ---------------------------------------------------------
    # PAGE 1 — DATA PREVIEW
    # ---------------------------------------------------------
    if page == "📁 Data Preview":
        st.header("📁 Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        st.subheader("Dataset Information")
        st.write(f"**Rows:** {df.shape[0]}")
        st.write(f"**Columns:** {df.shape[1]}")
        st.write("**Columns List:**")
        st.write(df.columns.tolist())

        st.subheader("Missing Values")
        st.write(df.isnull().sum())

    # ---------------------------------------------------------
    # PAGE 2 — STATISTICS
    # ---------------------------------------------------------
    elif page == "📊 Statistics":
        st.header("📊 Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)

        st.subheader("Categorical Breakdown")
        categorical = df.select_dtypes(include=["object"]).columns
        if len(categorical):
            for col in categorical:
                st.write(f"🔹 {col}")
                st.write(df[col].value_counts())
        else:
            st.info("No categorical columns found.")

    # ---------------------------------------------------------
    # PAGE 3 — VISUALIZATIONS
    # ---------------------------------------------------------
    elif page == "📈 Visualizations":
        st.header("📈 Interactive Visualizations")

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

        chart_type = st.selectbox("Choose Visualization Type", [
            "Histogram",
            "Bar Chart",
            "Line Chart",
            "Scatter Plot",
            "Box Plot"
        ])

        if chart_type == "Histogram":
            col = st.selectbox("Select column", numeric_cols)
            fig = px.histogram(df, x=col, nbins=30)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Bar Chart":
            col = st.selectbox("Select column", categorical_cols)
            fig = px.bar(df[col].value_counts())
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Line Chart":
            col = st.selectbox("Select column", numeric_cols)
            st.line_chart(df[col])

        elif chart_type == "Scatter Plot":
            x = st.selectbox("X-axis", numeric_cols)
            y = st.selectbox("Y-axis", numeric_cols)
            fig = px.scatter(df, x=x, y=y)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Box Plot":
            col = st.selectbox("Select column", numeric_cols)
            fig = px.box(df, y=col)
            st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # PAGE 4 — OUTLIER DETECTION
    # ---------------------------------------------------------
    elif page == "📉 Outlier Detection":
        st.header("📉 Outlier Detection (IQR Method)")

        num_cols = df.select_dtypes(include="number").columns.tolist()
        col = st.selectbox("Select numeric column", num_cols)

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower) | (df[col] > upper)]

        st.write(f"Total Outliers in **{col}**: {outliers.shape[0]}")

        st.dataframe(outliers, use_container_width=True)

        # Box plot
        fig = px.box(df, y=col)
        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # PAGE 5 — CORRELATION
    # ---------------------------------------------------------
    elif page == "🔥 Correlation Explorer":
        st.header("🔥 Correlation Heatmap")

        num_cols = df.select_dtypes(include="number")
        corr = num_cols.corr()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        st.pyplot(fig)

    # ---------------------------------------------------------
    # PAGE 6 — TREND DETECTION
    # ---------------------------------------------------------
    elif page == "📅 Trend Detection":
        st.header("📅 Trend Detection")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        col = st.selectbox("Select numeric column", numeric_cols)

        st.line_chart(df[col])

        st.subheader("Basic Trend Analysis")
        diff = df[col].diff().mean()

        if diff > 0:
            st.success("📈 The trend is **increasing**.")
        elif diff < 0:
            st.error("📉 The trend is **decreasing**.")
        else:
            st.info("➖ No clear trend detected.")

    # ---------------------------------------------------------
    # PAGE 7 — CLEAN & EXPORT
    # ---------------------------------------------------------
    elif page == "🧹 Clean & Export":
        st.header("🧹 Clean Data")
        method = st.radio("Choose clean method:", [
            "Drop rows with nulls",
            "Drop columns with nulls",
            "Fill nulls with mean",
            "Fill nulls with median",
            "Fill nulls with 0"
        ])

        df_clean = df.copy()

        if st.button("Apply Cleaning"):
            if method == "Drop rows with nulls":
                df_clean = df_clean.dropna()
            elif method == "Drop columns with nulls":
                df_clean = df_clean.dropna(axis=1)
            elif method == "Fill nulls with mean":
                df_clean = df_clean.fillna(df_clean.mean())
            elif method == "Fill nulls with median":
                df_clean = df_clean.fillna(df_clean.median())
            elif method == "Fill nulls with 0":
                df_clean = df_clean.fillna(0)

            st.success("Cleaning applied!")
            st.dataframe(df_clean.head(), use_container_width=True)

            csv = df_clean.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download Cleaned CSV", csv, "cleaned_data.csv")

else:
    st.info("Upload a CSV or Excel file to begin.")
