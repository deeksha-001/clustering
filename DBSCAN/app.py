import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist

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

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR.parent / "models"

cluster_centers = joblib.load(
    MODELS_DIR / "cluster_centers.pkl"
)

clustered_df = joblib.load(
    MODELS_DIR / "dbscan_clustered_df.pkl"
)

# ----------------------------------
# Page Config
# ----------------------------------

st.set_page_config(
    page_title="DBSCAN Customer Segmentation",
    page_icon="📍",
    layout="wide"
)

# ----------------------------------
# Title
# ----------------------------------

st.title("📍 Customer Segmentation using DBSCAN")

st.markdown(
    """
    This application uses **DBSCAN (Density-Based Spatial Clustering)** 
    to segment customers based on:

    - Annual Income (k$)
    - Spending Score (1-100)

    DBSCAN can automatically identify **outliers (noise points)**,
    making it different from K-Means and Hierarchical Clustering.
    """
)

# ----------------------------------
# Sidebar
# ----------------------------------

with st.sidebar:

    st.header("📊 Model Information")

    st.metric(
        "Silhouette Score",
        "0.4133"
    )

    st.metric(
        "Clusters Found",
        "4"
    )

    st.metric(
        "Noise Points",
        "15"
    )

    st.markdown("---")

    st.write(
        """
        Parameters

        • eps = 0.4

        • min_samples = 5

        Features

        • Annual Income (k$)

        • Spending Score (1-100)
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
# Prediction
# ----------------------------------

if st.button("Predict Segment"):

    customer = np.array([
        [income, score]
    ])

    distances = cdist(
        customer,
        cluster_centers.values
    )

    min_distance = distances.min()

    cluster = np.argmin(
        distances
    )

    cluster_info = {
        0: "🙂 Average Income - Average Spending Customers",
        1: "💸 Low Income - Low Spending Customers",
        2: "💎 High Income - High Spending Customers",
        3: "🏦 High Income - Low Spending Customers"
    }

    # ----------------------------------
    # Outlier Detection
    # ----------------------------------

    st.subheader("Prediction Result")

    if min_distance > 35:

        st.warning(
            "⚠️ This customer appears to be an outlier and does not closely match any known customer segment."
        )

        outlier = True

    else:

        st.success(
            f"Customer belongs to Cluster {cluster}"
        )

        st.info(
            cluster_info[cluster]
        )

        outlier = False

    # ----------------------------------
    # Metrics
    # ----------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Nearest Cluster",
            cluster
        )

    with col2:

        st.metric(
            "Distance",
            f"{min_distance:.2f}"
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

    # Plot clusters

    scatter = ax.scatter(
        clustered_df["Annual Income (k$)"],
        clustered_df["Spending Score (1-100)"],
        c=clustered_df["Cluster"],
        s=60,
        alpha=0.7
    )

    # Cluster centers

    ax.scatter(
        cluster_centers.iloc[:,0],
        cluster_centers.iloc[:,1],
        marker="X",
        s=250,
        label="Cluster Centers"
    )

    # Customer

    ax.scatter(
        income,
        score,
        marker="*",
        s=400,
        label="New Customer"
    )

    ax.set_xlabel(
        "Annual Income (k$)"
    )

    ax.set_ylabel(
        "Spending Score (1-100)"
    )

    ax.set_title(
        "DBSCAN Customer Segmentation"
    )

    ax.legend()

    st.pyplot(fig)

    # ----------------------------------
    # Interpretation
    # ----------------------------------

    st.subheader("Interpretation")

    if outlier:

        st.warning(
            """
            This customer lies far from all known clusters.
            DBSCAN would likely consider this customer a noise point
            or potential outlier.
            """
        )

    elif min_distance < 10:

        st.success(
            "This customer closely matches an existing segment."
        )

    elif min_distance < 25:

        st.info(
            "This customer generally fits the predicted segment."
        )

    else:

        st.warning(
            "This customer shows characteristics of multiple segments."
        )

# ----------------------------------
# Cluster Statistics
# ----------------------------------

with st.expander(
    "View Cluster Statistics"
):

    stats = clustered_df[
        clustered_df["Cluster"] != -1
    ].groupby("Cluster")[
        ["Annual Income (k$)",
         "Spending Score (1-100)"]
    ].mean()

    st.dataframe(stats)

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
    **Algorithm:** DBSCAN

    **Type:** Density-Based Clustering

    **Key Advantage:** Detects outliers automatically

    **Features Used:** Annual Income (k$), Spending Score (1-100)
    """
)