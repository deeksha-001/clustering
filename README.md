# Customer Segmentation using Clustering Algorithms

## Overview

This project explores multiple unsupervised machine learning techniques for customer segmentation using the Mall Customers dataset.

The objective is to group customers based on their spending behavior and annual income, helping businesses identify valuable customer segments for targeted marketing strategies.

The project compares four popular clustering algorithms:

* K-Means Clustering
* Hierarchical Clustering
* DBSCAN
* Gaussian Mixture Model (GMM)

Interactive Streamlit applications were developed for each algorithm to visualize customer segments and predict customer group membership.

---

## Dataset

**Mall Customers Dataset**

Features used:

* Annual Income (k$)
* Spending Score (1-100)

Target labels are not available because clustering is an unsupervised learning problem.

---

## Project Structure

```text
clustering/
│
├── data/
│   └── Mall_Customers.csv
│
├── kmeans/
│   ├── notebook.ipynb
│   ├── app.py
│   ├── requirements.txt
│   └── models/
│
├── hierarchical/
│   ├── notebook.ipynb
│   ├── app.py
│   ├── requirements.txt
│   └── models/
│
├── dbscan/
│   ├── notebook.ipynb
│   ├── app.py
│   ├── requirements.txt
│   └── models/
│
├── gmm/
│   ├── notebook.ipynb
│   ├── app.py
│   ├── requirements.txt
│   └── models/
│
└── README.md
```

---

## Workflow

1. Data Cleaning
2. Feature Selection
3. Feature Scaling
4. Clustering Model Training
5. Cluster Evaluation
6. Visualization
7. Model Serialization
8. Streamlit Deployment

---

# 1. K-Means Clustering

### Description

K-Means is a centroid-based clustering algorithm that partitions customers into K distinct groups.

### Model Selection

* Elbow Method (WCSS)
* Silhouette Analysis

### Final Configuration

```python
n_clusters = 5
```

### Results

| Metric           | Value  |
| ---------------- | ------ |
| Silhouette Score | 0.5547 |
| Clusters         | 5      |

### Identified Segments

* High Income – High Spending
* High Income – Low Spending
* Low Income – High Spending
* Low Income – Low Spending
* Average Customers

---

# 2. Hierarchical Clustering

### Description

Agglomerative Hierarchical Clustering builds a hierarchy of customer groups using Ward linkage.

### Model Selection

* Dendrogram Analysis

### Results

| Metric           | Value  |
| ---------------- | ------ |
| Silhouette Score | 0.5538 |
| Clusters         | 5      |

### Features

* Dendrogram Visualization
* Cluster Center Analysis
* Customer Segment Prediction

---

# 3. DBSCAN

### Description

DBSCAN is a density-based clustering algorithm capable of detecting outliers automatically.

### Parameters

```python
eps = 0.4
min_samples = 5
```

### Results

| Metric           | Value  |
| ---------------- | ------ |
| Silhouette Score | 0.4133 |
| Clusters         | 4      |
| Noise Points     | 15     |

### Key Feature

DBSCAN identifies customers that do not belong to any cluster and labels them as noise points.

---

# 4. Gaussian Mixture Model (GMM)

### Description

GMM is a probabilistic clustering algorithm that performs soft clustering.

Unlike K-Means, customers can belong to multiple clusters with different probabilities.

### Model Selection

* AIC Analysis
* BIC Analysis

### Final Configuration

```python
n_components = 5
```

### Results

| Metric           | Value  |
| ---------------- | ------ |
| Silhouette Score | 0.5537 |
| Components       | 5      |

### Key Feature

GMM provides cluster membership probabilities and confidence scores.

---

# Algorithm Comparison

| Algorithm    | Type           | Silhouette Score |
| ------------ | -------------- | ---------------- |
| K-Means      | Centroid-Based | 0.5547           |
| Hierarchical | Tree-Based     | 0.5538           |
| GMM          | Probabilistic  | 0.5537           |
| DBSCAN       | Density-Based  | 0.4133           |

---

# Key Findings

* K-Means achieved the highest clustering quality on this dataset.
* Hierarchical Clustering produced nearly identical results while providing a dendrogram for visualization.
* GMM successfully modeled customer memberships probabilistically and achieved comparable performance.
* DBSCAN identified outliers effectively but produced lower-quality clusters for this dataset.

---

# Streamlit Applications

Each clustering algorithm includes a dedicated Streamlit dashboard featuring:

* Customer Segment Prediction
* Cluster Visualization
* Dataset Exploration
* Model Metrics
* Interactive User Inputs

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* SciPy
* Matplotlib
* Joblib
* Streamlit

---

# Installation

```bash
git clone <repository-url>

```

Each project contains its own requirements file.

---

# Run Applications

K-Means

```bash
cd kmeans
pip install -r requirements.txt
streamlit run app.py
```

Hierarchical

```bash
cd hierarchical
pip install -r requirements.txt
streamlit run app.py
```

DBSCAN

```bash
cd dbscan
pip install -r requirements.txt
streamlit run app.py
```

GMM

```bash
cd gmm
pip install -r requirements.txt
streamlit run app.py
```

---

# Learning Outcomes

This project demonstrates:

* Unsupervised Machine Learning
* Customer Segmentation
* Cluster Evaluation Metrics
* Model Comparison
* Interactive Dashboard Development
* Deployment-Ready Machine Learning Applications

---

## Author

Deeksha Reddy

