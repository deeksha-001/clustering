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

cluster_centers = joblib.load(
    "models/cluster_centers.pkl"
)

clustered_df = joblib.load(
    "models/hierarchical_clustered_df.pkl"
)

# ----------------------------------
# Page Config
# ----------------------------------

st.set_page_config(
    page_title="Hierarchical Customer Segmentation",
    page_icon="🌳",
    layout="wide"
)

# ----------------------------------
# Title
# ----------------------------------

st.title("🌳 Hierarchical Customer Segmentation")

st.markdown(
    """
    This application segments customers using
    **Agglomerative Hierarchical Clustering**.

    Customers are grouped based on:

    - Annual Income (k$)
    - Spending Score (1-100)

    The model uses **Ward Linkage** and achieved a
    **Silhouette Score of 0.5538**.
    """
)

# ----------------------------------
# Sidebar
# ----------------------------------

with st.sidebar:

    st.header("📊 Model Information")

    st.metric(
        "Silhouette Score",
        "0.5538"
    )

    st.metric(
        "Clusters",
        "5"
    )

    st.metric(
        "Linkage",
        "Ward"
    )

    st.markdown("---")

    st.write(
        """
        Features Used:

        • Annual Income (k$)

        • Spending Score (1-100)

        Algorithm:

        • Agglomerative Hierarchical Clustering
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

    customer = np.array([
        [income, score]
    ])

    distances = cdist(
        customer,
        cluster_centers.values
    )

    cluster = np.argmin(
        distances
    )

    min_distance = distances.min()

    cluster_info = {
        0: "🏦 High Income - Low Spending Customers",
        1: "💎 High Income - High Spending Customers",
        2: "🙂 Average Income - Average Spending Customers",
        3: "🛒 Low Income - High Spending Customers",
        4: "💸 Low Income - Low Spending Customers"
    }

    # ----------------------------------
    # Result
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
            "Nearest Cluster Distance",
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
    # Cluster Visualization
    # ----------------------------------

    st.subheader(
        "Customer Position in Dataset"
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    scatter = ax.scatter(
        clustered_df["Annual Income (k$)"],
        clustered_df["Spending Score (1-100)"],
        c=clustered_df["Cluster"],
        alpha=0.7,
        s=60
    )

    # Cluster Centers

    ax.scatter(
        cluster_centers.iloc[:, 0],
        cluster_centers.iloc[:, 1],
        marker="X",
        s=250,
        label="Cluster Centers"
    )

    # New Customer

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
        "Hierarchical Customer Segmentation"
    )

    ax.legend()

    st.pyplot(fig)

    # ----------------------------------
    # Interpretation
    # ----------------------------------

    st.subheader("Interpretation")

    if min_distance < 10:

        st.success(
            "This customer closely matches an existing customer segment."
        )

    elif min_distance < 25:

        st.info(
            "This customer generally fits the predicted segment."
        )

    else:

        st.warning(
            "This customer lies farther from the segment center and may have mixed characteristics."
        )

# ----------------------------------
# Dendrogram
# ----------------------------------

try:

    st.subheader(
        "Hierarchical Clustering Dendrogram"
    )

    st.image(
        "models/dendrogram.png"
    )

except:

    pass

# ----------------------------------
# Cluster Centers
# ----------------------------------

with st.expander(
    "View Cluster Centers"
):

    st.dataframe(
        cluster_centers
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
    **Algorithm:** Agglomerative Hierarchical Clustering

    **Linkage Method:** Ward Linkage

    **Features Used:** Annual Income (k$), Spending Score (1-100)

    **Clustering Type:** Hierarchical / Tree-Based Clustering
    """
)