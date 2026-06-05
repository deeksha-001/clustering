import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ----------------------------------
# Load Dataset
# ----------------------------------
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR.parent / "data" / "Mall_Customers.csv"

df = pd.read_csv(DATA_PATH)

# ----------------------------------
# Load Saved Files
# ----------------------------------

kmeans = joblib.load(
    "models/kmeans_model.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)

clustered_df = joblib.load(
    "models/kmeans_df.pkl"
)

# ----------------------------------
# Cluster Centers
# ----------------------------------

centers = scaler.inverse_transform(
    kmeans.cluster_centers_
)

# ----------------------------------
# Page Config
# ----------------------------------

st.set_page_config(
    page_title="K-Means Customer Segmentation",
    page_icon="📊",
    layout="wide"
)

# ----------------------------------
# Title
# ----------------------------------

st.title("📊 Customer Segmentation using K-Means")

st.markdown(
    """
    This application segments customers using the
    **K-Means Clustering Algorithm**.

    Customers are grouped based on:

    - Annual Income (k$)
    - Spending Score (1-100)

    The model identifies five distinct customer segments.
    """
)

# ----------------------------------
# Sidebar
# ----------------------------------

with st.sidebar:

    st.header("📈 Model Information")

    st.metric(
        "Silhouette Score",
        "0.5547"
    )

    st.metric(
        "Clusters",
        "5"
    )

    st.metric(
        "Algorithm",
        "K-Means"
    )

    st.markdown("---")

    st.write(
        """
        Features Used

        • Annual Income (k$)

        • Spending Score (1-100)

        Parameters

        • K = 5

        • init = k-means++

        • random_state = 42
        """
    )

# ----------------------------------
# Inputs
# ----------------------------------

st.subheader("Customer Information")

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

# ----------------------------------
# Predict
# ----------------------------------

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

    cluster = kmeans.predict(
        customer_scaled
    )[0]

    cluster_info = {
        0: "🙂 Average Income - Average Spending Customers",
        1: "💎 High Income - High Spending Customers",
        2: "🛒 Low Income - High Spending Customers",
        3: "🏦 High Income - Low Spending Customers",
        4: "💸 Low Income - Low Spending Customers"
    }

    distances = np.linalg.norm(
        centers - np.array([income, score]),
        axis=1
    )

    nearest_distance = distances.min()

    # ----------------------------------
    # Results
    # ----------------------------------

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

        st.subheader("Distance")

        st.metric(
            "Distance to Centroid",
            f"{nearest_distance:.2f}"
        )

    # ----------------------------------
    # Customer Details
    # ----------------------------------

    st.subheader("Customer Details")

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Annual Income",
            f"{income} k$"
        )

    with c2:

        st.metric(
            "Spending Score",
            score
        )

    # ----------------------------------
    # Visualization
    # ----------------------------------

    st.subheader(
        "Customer Position in Dataset"
    )

    fig, ax = plt.subplots(
        figsize=(10,6)
    )

    scatter = ax.scatter(
        clustered_df["Annual Income (k$)"],
        clustered_df["Spending Score (1-100)"],
        c=clustered_df["Cluster"],
        alpha=0.7,
        s=60
    )

    # Centroids

    ax.scatter(
        centers[:,0],
        centers[:,1],
        marker='X',
        s=250,
        label='Centroids'
    )

    # New Customer

    ax.scatter(
        income,
        score,
        marker='*',
        s=400,
        label='New Customer'
    )

    ax.set_xlabel(
        "Annual Income (k$)"
    )

    ax.set_ylabel(
        "Spending Score (1-100)"
    )

    ax.set_title(
        "K-Means Customer Segmentation"
    )

    ax.legend()

    st.pyplot(fig)

    # ----------------------------------
    # Interpretation
    # ----------------------------------

    st.subheader("Interpretation")

    if nearest_distance < 10:

        st.success(
            "This customer strongly matches the predicted segment."
        )

    elif nearest_distance < 25:

        st.info(
            "This customer reasonably fits the predicted segment."
        )

    else:

        st.warning(
            "This customer lies farther from the centroid and may share characteristics with other segments."
        )

# ----------------------------------
# Cluster Centers
# ----------------------------------

with st.expander(
    "View Cluster Centers"
):

    centers_df = pd.DataFrame(
        centers,
        columns=[
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ]
    )

    st.dataframe(
        centers_df
    )

# ----------------------------------
# Dataset Sample
# ----------------------------------

with st.expander(
    "View Dataset Sample"
):

    st.dataframe(
        df.head(10)
    )

# ----------------------------------
# Footer
# ----------------------------------

st.markdown("---")

st.markdown(
    """
    **Algorithm:** K-Means Clustering

    **Clustering Type:** Centroid-Based Clustering

    **Features Used:** Annual Income (k$), Spending Score (1-100)

    **Model Selection:** Elbow Method + Silhouette Analysis
    """
)