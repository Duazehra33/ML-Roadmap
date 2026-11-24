# Machine Learning – Week 03
## Unsupervised Learning & Hyperparameter Tuning

### Overview

This repository contains the theoretical and practical learning materials for Week 3 of Machine Learning. It focuses on unsupervised learning techniques, clustering algorithms, dimensionality reduction, and hyperparameter optimization.

The repository includes a theoretical PDF explaining the core concepts and a Jupyter Notebook demonstrating clustering techniques, data visualization, and model tuning using Python and Scikit-learn.

## Repository Contents

### 1. `Week3 - ML.pdf`
**Theory and Conceptual Understanding**

This PDF covers the fundamental concepts of Unsupervised Learning and Hyperparameter Tuning, including:

- Introduction to Unsupervised Learning.
- K-Means Clustering.
- Hierarchical Clustering.
- Principal Component Analysis (PCA).
- PCA for Data Visualization and Feature Reduction.
- Hyperparameter Tuning.
- GridSearchCV.
- RandomizedSearchCV.
- Elbow Method.
- Silhouette Score.

### 2. `week3.ipynb`
**Practical Implementation Using Python and Scikit-learn**

This notebook demonstrates the practical implementation of clustering algorithms, dimensionality reduction, and hyperparameter tuning.

Key areas covered:

- Loading and exploring datasets.
- Implementing K-Means Clustering on the Mall Customers dataset.
- Identifying customer segments using clustering.
- Applying the Elbow Method to determine the optimal number of clusters.
- Evaluating clustering quality using the Silhouette Score.
- Visualizing clusters using Principal Component Analysis (PCA).
- Performing dimensionality reduction for data visualization.
- Applying GridSearchCV for hyperparameter optimization.
- Tuning classification models to improve performance.
- Analyzing and comparing model results before and after tuning.

### Concepts and Techniques

**1. K-Means Clustering**

An unsupervised learning algorithm that groups similar data points into a predefined number of clusters based on their distance from cluster centroids.

**2. Hierarchical Clustering**

A clustering technique that creates a hierarchy of clusters by progressively merging or splitting data points based on similarity.

**3. Principal Component Analysis (PCA)**

A dimensionality reduction technique that transforms high-dimensional data into fewer components while preserving as much relevant variance as possible.

**4. Hyperparameter Tuning**

The process of finding suitable model configurations to improve machine learning performance.

- GridSearchCV: Tests specified combinations of hyperparameters to identify the best configuration.
- RandomizedSearchCV: Tests randomly selected hyperparameter combinations within specified search ranges.

**5. Clustering Evaluation**

- Elbow Method: Helps identify an appropriate number of clusters.
- Silhouette Score: Measures how well data points fit within their assigned clusters compared with other clusters.

### Dataset

**Mall Customers Dataset**

Used for implementing K-Means Clustering and identifying customer segments based on customer-related features.

The dataset helps demonstrate how unsupervised learning can be applied to customer segmentation and pattern discovery.

### Model Optimization

Hyperparameter tuning is performed using GridSearchCV to explore different model configurations and identify the best-performing parameters for a classification model.

The results are analyzed to understand how parameter selection affects model performance.

## Technologies and Libraries

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## How to Run

1. Clone or download this repository.

2. Navigate to the project directory.

3. Install the required libraries:

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn jupyter
   ```

4. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Open `week3.ipynb` and execute the cells sequentially.

## Learning Objectives

By completing these materials, you will gain an understanding of:

- Unsupervised learning and its applications.
- Clustering techniques and customer segmentation.
- K-Means and Hierarchical Clustering.
- Dimensionality reduction using PCA.
- Cluster evaluation using Elbow Method and Silhouette Score.
- Hyperparameter optimization using GridSearchCV and RandomizedSearchCV.
- Model performance improvement through parameter tuning.
- Visualization and interpretation of clustering results.

## Week 3 Checkpoint Evaluation

- Implement K-Means Clustering with appropriate visualizations.
- Apply PCA for cluster visualization.
- Determine the optimal number of clusters using the Elbow Method.
- Evaluate clusters using the Silhouette Score.
- Apply GridSearchCV to at least one classification model.
- Analyze and explain the hyperparameter tuning results.

---

**Phase:** Machine Learning Fundamentals  
**Week:** 03  
**Focus:** Unsupervised Learning & Hyperparameter Tuning  
**Framework:** Scikit-learn  
**Language:** Python  
**Environment:** Jupyter Notebook