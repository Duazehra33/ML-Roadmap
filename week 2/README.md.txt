# Machine Learning – Week 02
## Core Classification Models & Data Preprocessing

### Overview

This repository contains the theoretical and practical learning materials for Week 2 of Machine Learning. It focuses on understanding core classification algorithms, data preprocessing techniques, model training, and performance evaluation.

The repository includes a theoretical PDF covering the fundamental concepts and a Jupyter Notebook demonstrating the practical implementation and comparison of classification models.

## Repository Contents

### 1. `week 2 ml.pdf`
**Theory and Conceptual Understanding**

This PDF covers the theoretical foundations of classification algorithms and data preprocessing, including:

- Decision Trees
- K-Nearest Neighbors (kNN)
- Naive Bayes
- Train/Test Split
- Cross-Validation
- Feature Scaling
- Handling Missing Data
- Encoding Categorical Features
- Feature Selection vs. Dimensionality Reduction
- Classification Model Evaluation

### 2. `week2.ipynb`
**Practical Implementation Using Python and Scikit-learn**

This notebook demonstrates the implementation and evaluation of multiple classification algorithms using different datasets.

Key areas covered:

- Loading and exploring datasets.
- Data preprocessing and cleaning.
- Handling missing values.
- Encoding categorical features.
- Applying feature scaling.
- Splitting data into training and testing sets.
- Implementing Decision Tree classification.
- Implementing K-Nearest Neighbors (kNN).
- Implementing Naive Bayes classification.
- Building preprocessing and classification pipelines.
- Comparing classification models.
- Visualizing confusion matrices and model performance.

### Datasets

The practical exercises focus on the following datasets:

| Dataset | Description | Task |
|---|---|---|
| Titanic | Passenger information and survival records | Binary Classification |
| Wine | Chemical properties of wine samples | Multi-Class Classification |
| Breast Cancer | Diagnostic features of breast cancer cases | Binary Classification |

### Machine Learning Algorithms

The following classification algorithms are explored:

**1. Decision Tree**

A tree-based classification algorithm that makes predictions through a sequence of feature-based decisions.

**2. K-Nearest Neighbors (kNN)**

A distance-based classification algorithm that predicts the class of a sample using the majority class of its nearest neighbors.

**3. Naive Bayes**

A probabilistic classification algorithm based on Bayes' theorem and the assumption of conditional independence between features.

### Data Preprocessing Techniques

- Missing Value Imputation
- Feature Scaling and Normalization
- Categorical Feature Encoding
- Train/Test Splitting
- Cross-Validation
- Feature Selection
- Dimensionality Reduction

### Model Evaluation

Classification models are compared using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Visualizations are used to understand and compare the performance of different classification algorithms.

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

5. Open `week2.ipynb` and execute the cells sequentially.

## Learning Objectives

By completing these materials, you will gain an understanding of:

- Core classification algorithms and their working principles.
- Data cleaning and preprocessing techniques.
- Feature scaling and categorical encoding.
- Training and testing machine learning models.
- Cross-validation and model generalization.
- Building machine learning pipelines.
- Evaluating and comparing classification models.
- Visualizing model performance using confusion matrices.

## Week 2 Checkpoint Evaluation

- Build a complete machine learning pipeline combining preprocessing and three classification algorithms.
- Implement Decision Tree, kNN, and Naive Bayes classifiers.
- Visualize confusion matrices for different models.
- Compare model performance using evaluation metrics.
- Prepare a performance comparison summary.

---

**Phase:** Machine Learning Fundamentals  
**Week:** 02  
**Focus:** Classification Models & Data Preprocessing  
**Framework:** Scikit-learn  
**Language:** Python  
**Environment:** Jupyter Notebook