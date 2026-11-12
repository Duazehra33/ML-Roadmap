# Customer Intelligence System

A dynamic, interactive, and deployable Streamlit-based machine learning web application designed to analyze customer behavior, explore spending patterns, understand customer segments, and predict spending categories using data analytics and machine learning.

The system allows users to upload their own Mall Customer dataset in CSV or Excel format. All dashboard metrics, visualizations, customer insights, and machine learning results are generated dynamically according to the uploaded dataset.

## Features

### 1. Interactive Dashboard
- Total customer count.
- Average customer age.
- Average annual income.
- Average spending score.
- Gender distribution visualization.
- Customer age distribution.
- Annual income vs. spending score analysis.

### 2. Customer Insights (Exploratory Data Analysis)
- Age distribution analysis.
- Annual income distribution.
- Spending score distribution.
- Average spending by gender.
- Correlation analysis using a heatmap.
- Income vs. spending patterns by age.
- Existing customer cluster distribution.
- Cluster-wise customer summary.

### 3. Customer Segmentation
- Analysis of existing customer clusters.
- Interactive cluster filtering.
- Customer distribution by cluster.
- Income vs. spending visualization.
- Age vs. spending visualization.
- Customer segment profiles.
- Cluster-wise average age, income, and spending.
- View and download segmented customer records.

### 4. Spending Category Prediction
- Decision Tree Classification.
- Automatic creation of Low, Medium, and High spending categories.
- 80/20 training and testing split.
- Numerical and categorical data preprocessing.
- Model evaluation using accuracy.
- Confusion matrix visualization.
- Classification report.
- Interactive prediction for new customers.
- Spending category probability estimates.

### 5. Dataset Management
- Upload CSV and Excel datasets.
- Dataset preview and information.
- Automatic column-name normalization.
- Duplicate record detection and removal.
- Missing value analysis.
- Missing value percentage calculation.
- Numerical missing value imputation using the median.
- Categorical missing value imputation using the mode.
- Cleaned dataset preview.
- Download cleaned customer data.

### 6. User Interface
- Interactive Streamlit interface.
- Light and dark appearance modes.
- Custom purple-themed UI.
- Responsive dashboard layout.
- Interactive charts, tables, metrics, and filters.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| Pandas | Dataset loading and manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical charts and visualizations |
| Scikit-learn | Machine learning and preprocessing |

## Machine Learning Algorithms and Techniques

### 1. K-Means Clustering
An unsupervised machine learning algorithm used to group customers into different segments based on their age, annual income, and spending score.

**Purpose:** Understand customer groups and spending behavior.

**Note:** The current application analyzes existing cluster labels from the uploaded dataset. It does not train a new K-Means model.

### 2. Decision Tree Classification
A supervised machine learning algorithm that makes predictions using a tree-like structure of decision rules.

**Purpose:** Predict whether a customer belongs to the Low, Medium, or High spending category based on age, annual income, and gender.

### 3. Data Preprocessing
- **StandardScaler:** Standardizes numerical features.
- **OneHotEncoder:** Converts categorical variables into numerical representations.
- **SimpleImputer:** Handles missing numerical and categorical values.
- **ColumnTransformer:** Applies different preprocessing techniques to different columns.
- **Pipeline:** Combines preprocessing and machine learning steps into one workflow.

## Dataset Requirements

The application accepts CSV (`.csv`) and Excel (`.xlsx`, `.xls`) files.

The uploaded dataset must contain the following columns:

| Column | Description |
|---|---|
| CustomerID | Unique customer identifier |
| Gender | Customer gender |
| Age | Customer age |
| Annual Income (k$) | Customer annual income |
| Spending Score (1-100) | Customer spending score |
| Cluster | Existing customer cluster assignment |

**Important:** All six columns are required in the current implementation. The `CustomerID` column is used for identification, not model training. The `Cluster` column is used to analyze existing customer segments.

The application supports common column-name variations through automatic column normalization.

## ML Design Notes

- Spending categories are generated automatically using one-third and two-third spending-score quantiles (tertiles).
- Spending Score is excluded from classification input features to prevent target leakage.
- The Decision Tree model uses Age, Annual Income, and Gender as input features.
- The dataset is divided into 80% training data and 20% testing data.
- Random state is set to 42 for reproducible results.
- Numerical features are processed using median imputation and StandardScaler.
- Gender is processed using most-frequent imputation and OneHotEncoder.
- A Decision Tree classifier is configured with a maximum depth of 5 and minimum samples split of 4.
- CustomerID is excluded from machine learning features.
- K-Means is treated as an unsupervised clustering technique, not a classification algorithm.
- Existing cluster assignments are analyzed without being overwritten.
- Missing numerical values are replaced with median values, while missing categorical values are filled using the mode.
- Duplicate records are removed during data cleaning.

## Run Locally

1. Install Python 3.10 or newer.
2. Open a terminal in the project folder.
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit application:

```bash
streamlit run app.py
```

5. Open the application in your browser.
6. Upload your customer dataset using the sidebar.
7. Explore the dashboard, customer insights, segmentation, predictions, and dataset management sections.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload the following files:
   - `app.py`
   - `requirements.txt`
   - `README.md`
3. Visit [Streamlit Community Cloud](https://share.streamlit.io/).
4. Sign in using your GitHub account.
5. Select **Create App**.
6. Choose your repository, branch, and `app.py` file.
7. Click **Deploy**.
8. Upload your customer dataset through the application sidebar.

## Project Objective

The main objective of the Customer Intelligence System is to transform raw customer data into meaningful business insights.

It helps businesses:

- Understand customer demographics and spending behavior.
- Identify patterns in customer income and spending.
- Analyze existing customer segments.
- Predict potential spending categories.
- Improve customer understanding through data-driven analysis.
- Support better marketing strategies and business decisions.

**Project Type:** Machine Learning | Data Analytics | Customer Intelligence | Interactive Web Application