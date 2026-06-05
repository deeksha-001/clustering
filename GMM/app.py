import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("../data/Mall_Customers.csv")

# -----------------------------
# Load Model
# -----------------------------

gmm = joblib.load("models/gmm_model.pkl")
scaler = joblib.load("models/gmm_scaler.pkl")

# -----------------------------
# Prepare Data
# -----------------------------

X = df[
    ["Annual Income (k$)",
     "Spending Score (1-100)"]
]

X_scaled = scaler.transform(X)

df["Cluster"] = gmm.predict(X_scaled)

centers = scaler.inverse_transform(
    gmm.means_
)

# -----------------------------
# Page Config
# -----------------------------

st.set_page_config(
    page_title="GMM Customer Segmentation",
    page_icon="🎯",
    layout="wide"
)

# -----------------------------
# Title
# -----------------------------

st.title("🎯 Customer Segmentation using Gaussian Mixture Model")

st.markdown(
    """
    This application segments customers using a **Gaussian Mixture Model (GMM)**.

    Unlike K-Means, GMM performs **soft clustering** and provides
    probabilities for each cluster.
    """
)

# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.header("Model Information")

    st.metric(
        "Silhouette Score",
        "0.5537"
    )

    st.metric(
        "Clusters",
        "5"
    )

    st.metric(
        "Algorithm",
        "GMM"
    )

    st.markdown("---")

    st.write("### Dataset")

    st.write(
        """
        Mall Customers Dataset

        Features Used:
        - Annual Income (k$)
        - Spending Score (1-100)
        """
    )

# -----------------------------
# Inputs
# -----------------------------

col1, col2 = st.columns(2)

with col1:

    income = st.number_input(
        "Annual Income (k$)",
        min_value=0,
        max_value=200,
        value=50
    )

with col2:

    score = st.slider(
        "Spending Score (1-100)",
        min_value=1,
        max_value=100,
        value=50
    )

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Segment"):

    customer = pd.DataFrame(
        [[income, score]],
        columns=[
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ]
    )

    customer_scaled = scaler.transform(
        customer
    )

    cluster = gmm.predict(
        customer_scaled
    )[0]

    probabilities = gmm.predict_proba(
        customer_scaled
    )[0]

    confidence = np.max(
        probabilities
    )

    cluster_info = {
        0: "🙂 Average Income - Average Spending Customers",
        1: "💎 High Income - High Spending Customers",
        2: "🛒 Low Income - High Spending Customers",
        3: "🏦 High Income - Low Spending Customers",
        4: "💸 Low Income - Low Spending Customers"
    }

    # -----------------------------
    # Results
    # -----------------------------

    st.success(
        f"Customer belongs to Cluster {cluster}"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Segment")

        st.info(
            cluster_info[cluster]
        )

    with col2:

        st.subheader("Confidence")

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2%}"
        )

        st.progress(
            float(confidence)
        )

    # -----------------------------
    # Probability Chart
    # -----------------------------

    st.subheader(
        "Cluster Membership Probabilities"
    )

    prob_df = pd.DataFrame({
        "Cluster": [
            "Cluster 0",
            "Cluster 1",
            "Cluster 2",
            "Cluster 3",
            "Cluster 4"
        ],
        "Probability": probabilities
    })

    st.bar_chart(
        prob_df.set_index(
            "Cluster"
        )
    )

    # -----------------------------
    # Scatter Plot
    # -----------------------------

    st.subheader(
        "Customer Position in Dataset"
    )

    fig, ax = plt.subplots(
        figsize=(10,6)
    )

    scatter = ax.scatter(
        df["Annual Income (k$)"],
        df["Spending Score (1-100)"],
        c=df["Cluster"],
        alpha=0.7
    )

    ax.scatter(
        centers[:,0],
        centers[:,1],
        marker="X",
        s=250,
        label="Cluster Centers"
    )

    ax.scatter(
        income,
        score,
        marker="X",
        s=350,
        label="New Customer"
    )

    ax.set_xlabel(
        "Annual Income (k$)"
    )

    ax.set_ylabel(
        "Spending Score"
    )

    ax.set_title(
        "Customer Segmentation"
    )

    ax.legend()

    st.pyplot(fig)

    # -----------------------------
    # Interpretation
    # -----------------------------

    st.subheader(
        "Interpretation"
    )

    if confidence >= 0.90:

        st.success(
            "The customer strongly belongs to this segment."
        )

    elif confidence >= 0.70:

        st.info(
            "The customer mostly belongs to this segment but shares characteristics with nearby segments."
        )

    else:

        st.warning(
            "The customer lies near the boundary between multiple segments."
        )

# -----------------------------
# Dataset Preview
# -----------------------------

with st.expander(
    "View Dataset Sample"
):

    st.dataframe(
        df.head(10)
    )

# -----------------------------
# Footer
# -----------------------------

st.markdown("---")

st.markdown(
    """
    **Model:** Gaussian Mixture Model (GMM)

    **Clustering Type:** Probabilistic / Soft Clustering

    **Features Used:** Annual Income (k$), Spending Score (1-100)
    """
)