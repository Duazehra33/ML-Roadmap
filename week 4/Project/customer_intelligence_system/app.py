import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    silhouette_score
)

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Intelligence System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Appearance control

# =========================================================
# APPEARANCE CONTROL
# =========================================================

_dark_mode = st.sidebar.toggle(
    "Dark appearance",
    value=False,
    key="dark_appearance"
)

# =========================================================
# COMPLETE LIGHT AND DARK THEMES
# =========================================================

if _dark_mode:

    COLORS = {
        "background": "#12101A",
        "surface": "#211C2B",
        "surface_soft": "#2B2438",
        "text": "#F5F1FC",
        "muted": "#C2B8D2",
        "heading": "#D6C5FF",
        "border": "#842aeb",
        "accent": "#A98BFF",
        "accent_dark": "#7954D6",
        "input": "#292336",
        "table": "#792ce6",
        "hover": "#382D4C",
        "grid": "#40364D"
    }

else:

    COLORS = {
        "background": "#F5F3FA",
        "surface": "#FFFFFF",
        "surface_soft": "#EFEBF7",
        "text": "#211A2D",
        "muted": "#5D536B",
        "heading": "#4D2A9A",
        "border": "#DCD4E9",
        "accent": "#6842C2",
        "accent_dark": "#4D2A9A",
        "input": "#FFFFFF",
        "table": "#FFFFFF",
        "hover": "#EAE3F6",
        "grid": "#E5DFED"
    }

# =========================================================
# CUSTOM UI / UX STYLING
# =========================================================

_theme_css = f"""
<style>

/* ==============================
   GLOBAL PAGE COLORS
============================== */

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {{
    background-color: {COLORS["background"]} !important;
    color: {COLORS["text"]} !important;
}}

.main .block-container {{
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}}

/* ==============================
   MAIN PAGE TEXT
============================== */

.stApp p,
.stApp li,
.stApp span,
.stApp label,
.stApp .stMarkdown,
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stWidgetLabel"] {{
    color: {COLORS["text"]};
}}

.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5 {{
    color: {COLORS["heading"]} !important;
    font-weight: 750;
    letter-spacing: -0.02em;
}}

.main-title {{
    font-size: 2.35rem;
    line-height: 1.2;
    font-weight: 850;
    color: {COLORS["heading"]} !important;
    letter-spacing: -0.035em;
    margin-bottom: 0.4rem;
}}

.sub-title {{
    color: {COLORS["muted"]} !important;
    font-size: 1.05rem;
    line-height: 1.7;
    margin-bottom: 1.5rem;
}}

/* ==============================
   SIDEBAR
============================== */

[data-testid="stSidebar"] {{
    background: linear-gradient(
        180deg,
        #241044 0%,
        #432078 55%,
        #57319A 100%
    ) !important;
    border-right: 1px solid #60439A;
}}

[data-testid="stSidebar"] * {{
    color: #FFFFFF !important;
}}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{
    color: #FFFFFF !important;
}}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] .stMarkdown {{
    color: #F5EEFF !important;
}}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
    color: #D8C9F3 !important;
}}

/* Sidebar navigation radio buttons */

[data-testid="stSidebar"] [role="radiogroup"] label {{
    background-color: rgba(255,255,255,0.06);
    border-radius: 9px;
    padding: 7px 10px;
    margin-bottom: 5px;
}}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {{
    background-color: rgba(255,255,255,0.16);
}}

/* Sidebar uploader */

[data-testid="stSidebar"] [data-testid="stFileUploader"] {{
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 12px;
    padding: 10px;
}}

/* ==============================
   CARDS AND METRICS
============================== */

[data-testid="stMetric"] {{
    background: {COLORS["surface"]} !important;
    border: 1px solid {COLORS["border"]} !important;
    border-left: 5px solid {COLORS["accent"]} !important;
    border-radius: 16px;
    padding: 1.1rem;
    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
}}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {{
    color: {COLORS["muted"]} !important;
    font-size: 0.95rem !important;
    font-weight: 650 !important;
}}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {{
    color: {COLORS["heading"]} !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
}}

/* ==============================
   INPUTS AND DROPDOWNS
============================== */

.stTextInput input,
.stNumberInput input,
.stTextArea textarea,
[data-testid="stDateInput"] input {{
    background-color: {COLORS["input"]} !important;
    color: {COLORS["text"]} !important;
    border: 1px solid {COLORS["border"]} !important;
    border-radius: 10px !important;
}}

[data-testid="stSelectbox"] div[data-baseweb="select"],
[data-testid="stMultiSelect"] div[data-baseweb="select"] {{
    background-color: {COLORS["input"]} !important;
    color: {COLORS["text"]} !important;
    border-radius: 10px !important;
}}

[data-baseweb="select"] span,
[data-baseweb="select"] div {{
    color: {COLORS["text"]} !important;
}}

[data-baseweb="popover"],
[data-baseweb="menu"] {{
    background-color: {COLORS["surface"]} !important;
}}

[data-baseweb="menu"] li {{
    color: {COLORS["text"]} !important;
}}

[data-baseweb="menu"] li:hover {{
    background-color: {COLORS["hover"]} !important;
}}

/* Input placeholders */

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {{
    color: {COLORS["muted"]} !important;
    opacity: 1;
}}

/* ==============================
   CHECKBOXES AND RADIO BUTTONS
============================== */

[data-testid="stCheckbox"] label,
[data-testid="stRadio"] label,
[data-testid="stToggle"] label {{
    color: {COLORS["text"]} !important;
}}

/* ==============================
   BUTTONS
============================== */

div.stButton > button,
div.stDownloadButton > button,
[data-testid="stFormSubmitButton"] button {{
    background: linear-gradient(
        100deg,
        {COLORS["accent_dark"]},
        {COLORS["accent"]}
    ) !important;
    color: #FFFFFF !important;
    border: 1px solid {COLORS["accent_dark"]} !important;
    border-radius: 11px !important;
    padding: 0.65rem 1.1rem;
    font-weight: 750;
    transition: all 0.2s ease;
}}

div.stButton > button:hover,
div.stDownloadButton > button:hover,
[data-testid="stFormSubmitButton"] button:hover {{
    filter: brightness(1.12);
    transform: translateY(-2px);
    color: #FFFFFF !important;
}}

/* ==============================
   DATAFRAMES AND TABLES
============================== */

[data-testid="stDataFrame"],
[data-testid="stTable"] {{
    background-color: {COLORS["surface"]} !important;
    border: 1px solid {COLORS["border"]} !important;
    border-radius: 12px;
    overflow: hidden;
}}

[data-testid="stDataFrame"] *,
[data-testid="stTable"] * {{
    color: {COLORS["text"]} !important;
}}

/* ==============================
   TABS
============================== */

[data-testid="stTabs"] button {{
    color: {COLORS["text"]} !important;
    font-weight: 700;
}}

[data-testid="stTabs"] button[aria-selected="true"] {{
    color: {COLORS["accent"]} !important;
}}

/* ==============================
   EXPANDERS
============================== */

[data-testid="stExpander"] {{
    background-color: {COLORS["surface"]} !important;
    border: 1px solid {COLORS["border"]} !important;
    border-radius: 12px;
}}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary span {{
    color: {COLORS["text"]} !important;
}}

/* ==============================
   ALERTS AND NOTICES
============================== */

[data-testid="stAlert"] {{
    background-color: {COLORS["surface"]} !important;
    border-radius: 12px;
}}

/* ==============================
   FILE UPLOADER
============================== */

[data-testid="stFileUploader"] {{
    background-color: {COLORS["surface"]} !important;
    border: 1px dashed {COLORS["border"]} !important;
    border-radius: 12px;
    padding: 12px;
}}

[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] small {{
    color: {COLORS["text"]} !important;
}}


hr {{
    border-color: {COLORS["border"]} !important;
}}

</style>
"""

st.markdown(_theme_css, unsafe_allow_html=True)

# =========================================================
# CHART COLORS
# =========================================================

_chart_colors = {
    "axes.facecolor": COLORS["surface"],
    "figure.facecolor": COLORS["surface"],
    "axes.edgecolor": COLORS["border"],
    "axes.labelcolor": COLORS["text"],
    "xtick.color": COLORS["muted"],
    "ytick.color": COLORS["muted"],
    "text.color": COLORS["text"],
    "grid.color": COLORS["grid"]
}

sns.set_theme(
    style="darkgrid" if _dark_mode else "whitegrid",
    rc=_chart_colors
)

plt.rcParams.update(_chart_colors)



# =========================================================
# CONSTANTS
# =========================================================

CUSTOMER_ID = "CustomerID"
GENDER = "Gender"
AGE = "Age"
INCOME = "Annual Income (k$)"
SPENDING = "Spending Score (1-100)"
CLUSTER = "Cluster"

REQUIRED_COLUMNS = [
    CUSTOMER_ID,
    GENDER,
    AGE,
    INCOME,
    SPENDING,
    CLUSTER
]

NUMERIC_COLUMNS = [
    AGE,
    INCOME,
    SPENDING
]

CLUSTER_FEATURES = [
    AGE,
    INCOME,
    SPENDING
]

# =========================================================
# DATASET COLUMN NORMALIZATION
# =========================================================

def normalize_columns(df):

    df = df.copy()

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace("_", " ", regex=False)
        .str.replace("-", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
    )

    aliases = {
        "customer id": CUSTOMER_ID,
        "customerid": CUSTOMER_ID,
        "id": CUSTOMER_ID,

        "gender": GENDER,
        "age": AGE,

        "annual income": INCOME,
        "annual income (k$)": INCOME,
        "annual income (k)": INCOME,
        "income": INCOME,

        "spending score": SPENDING,
        "spending score (1 100)": SPENDING,
        "spendingscore": SPENDING,

        "cluster": CLUSTER,
        "clusters": CLUSTER,
        "cluster label": CLUSTER,
        "cluster labels": CLUSTER
    }

    rename_dict = {
        col: aliases[col]
        for col in df.columns
        if col in aliases
    }

    df = df.rename(columns=rename_dict)

    return df

# =========================================================
# LOAD CSV OR EXCEL DATASET
# =========================================================

def load_dataset(uploaded_file):

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    elif file_name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)

    else:
        raise ValueError(
            "Please upload a CSV or Excel file."
        )

    df = normalize_columns(df)

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
            + ". Your dataset must contain CustomerID, Gender, "
              "Age, Annual Income (k$), Spending Score (1-100) "
              "and Cluster."
        )

    # Convert numerical customer attributes
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Customer ID is an identifier, not a model feature
    df[CUSTOMER_ID] = (
        df[CUSTOMER_ID]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

    # Gender normalization
    df[GENDER] = (
        df[GENDER]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

    # Preserve existing cluster labels
    df[CLUSTER] = (
        pd.to_numeric(df[CLUSTER], errors="coerce")
        .astype("Int64")
        .astype("string")
    )

    return df

# =========================================================
# DATA CLEANING
# =========================================================

def clean_dataset(df):

    data = df.copy()

    original_rows = len(data)

    data = data.drop_duplicates().copy()

    duplicate_count = original_rows - len(data)

    # Clean numerical columns
    for col in NUMERIC_COLUMNS:

        data[col] = data[col].replace(
            [np.inf, -np.inf],
            np.nan
        )

        median_value = data[col].median()

        if pd.isna(median_value):
            raise ValueError(
                f"The column '{col}' does not contain "
                "valid numeric values."
            )

        data[col] = data[col].fillna(
            median_value
        )

    # Clean Gender
    if data[GENDER].notna().any():

        gender_mode = data[GENDER].mode().iloc[0]

        data[GENDER] = data[GENDER].fillna(
            gender_mode
        )

    else:
        data[GENDER] = data[GENDER].fillna("Unknown")

    data[GENDER] = data[GENDER].astype(str)

    # Preserve CustomerID
    data[CUSTOMER_ID] = (
        data[CUSTOMER_ID]
        .fillna("Unknown")
        .astype(str)
    )

    # Preserve existing cluster assignments
    data[CLUSTER] = (
        data[CLUSTER]
        .fillna("Unassigned")
        .astype(str)
    )

    if data.empty:
        raise ValueError(
            "The dataset contains no usable records."
        )

    return data, duplicate_count

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def metric_card(label, value, color=None):
    """Render a native Streamlit KPI component."""
    st.metric(
        label=label,
        value=value
    )


def show_chart(fig):

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


def create_preprocessor():

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, [AGE, INCOME]),
        ("categorical", categorical_pipeline, [GENDER])
    ])

    return preprocessor


def create_spending_categories(scores):

    q1, q2 = scores.quantile(
        [1/3, 2/3]
    ).values

    if q1 >= q2:

        labels = np.where(
            scores <= q1,
            "Low",
            "High"
        )

        return pd.Series(
            labels,
            index=scores.index
        )

    return pd.cut(
        scores,
        bins=[-np.inf, q1, q2, np.inf],
        labels=["Low", "Medium", "High"],
        include_lowest=True
    ).astype(str)
# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.markdown(
    """
    <h2 style="color:white;">Customer Intelligence</h2>
    <p style="color:#E7D9FF;">
        Understand customers. Discover insights.
    </p>
    """,
    unsafe_allow_html=True
)

st.sidebar.divider()

page = st.sidebar.radio(
    "MAIN NAVIGATION",
    [
        "Dashboard",
        "Customer Insights",
        "Customer Segmentation",
        "Spending Prediction",
        "Dataset Management"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload Customer Dataset",
    type=["csv", "xlsx", "xls"],
    help="Upload a Kaggle Mall Customer Spending dataset."
)

st.sidebar.caption(
    "Supported formats: CSV, XLSX and XLS"
)

# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    '<div class="main-title">Customer Intelligence System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Analyze customer behavior, understand spending patterns '
    'and make data-driven business decisions.'
    '</div>',
    unsafe_allow_html=True
)

# =========================================================
# EMPTY STATE: NO DATASET UPLOADED
# =========================================================

if uploaded_file is None:

    st.markdown(
        """
        <div class="welcome-box">
            <h2>Welcome to Customer Intelligence System</h2>
            <p>
                Upload your customer spending dataset to unlock
                interactive dashboards, customer segmentation
                and machine learning predictions.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "No dataset uploaded yet. Please upload your "
        "Kaggle customer spending CSV or Excel file from the sidebar."
    )

    st.subheader("Explore the Platform")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Dashboard")
        st.write(
            "View customer statistics, average income, "
            "spending scores and business summaries."
        )

    with col2:
        st.markdown("### Customer Segmentation")
        st.write(
            "Group customers into meaningful segments "
            "using the K-Means clustering algorithm."
        )

    with col3:
        st.markdown("### Spending Prediction")
        st.write(
            "Predict customer spending categories using "
            "a Decision Tree machine learning model."
        )

    st.warning(
        "All statistics, graphs, charts and model results "
        "will appear only after a valid dataset is uploaded."
    )

    st.stop()


# =========================================================
# LOAD AND VALIDATE UPLOADED DATASET
# =========================================================

try:

    raw_data = load_dataset(uploaded_file)

    clean_data, duplicate_count = clean_dataset(raw_data)

except Exception as error:

    st.error(f"Dataset Error: {error}")

    st.stop()


# =========================================================
# DATASET VALIDATION
# =========================================================

if len(clean_data) < 10:

    st.error(
        "Please upload a dataset containing at least "
        "10 valid customer records."
    )

    st.stop()

if clean_data[GENDER].nunique() < 1:

    st.error("The Gender column does not contain usable values.")

    st.stop()

st.sidebar.success("Dataset successfully loaded!")

st.sidebar.caption(
    f"Records: {len(clean_data):,}"
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.markdown(
        """
        <div class="welcome-box">
            <h2>Business Overview</h2>
            <p>
                Get a quick overview of your uploaded customer
                dataset and explore important business metrics.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Key Performance Indicators")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card(
            "Total Customers",
            f"{len(clean_data):,}",
            "#8054D9"
        )

    with col2:
        metric_card(
            "Average Age",
            f"{clean_data[AGE].mean():.1f}",
            "#E879A9"
        )

    with col3:
        metric_card(
            "Average Income",
            f"{clean_data[INCOME].mean():.1f}k",
            "#18A999"
        )

    with col4:
        metric_card(
            "Average Spending",
            f"{clean_data[SPENDING].mean():.1f}",
            "#E89B45"
        )

    st.write("")
    st.divider()

    st.subheader("Customer Overview")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("#### Gender Distribution")

        fig, ax = plt.subplots(figsize=(7, 4))

        gender_counts = clean_data[GENDER].value_counts()

        ax.pie(
            gender_counts.values,
            labels=gender_counts.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=[
                "#8054D9",
                "#E879A9",
                "#18A999",
                "#E89B45"
            ],
            wedgeprops={"edgecolor": "white"}
        )

        ax.set_title("Customers by Gender")

        show_chart(fig)

    with col2:

        st.markdown("#### Age Distribution")

        fig, ax = plt.subplots(figsize=(7, 4))

        sns.histplot(
            data=clean_data,
            x=AGE,
            kde=True,
            color="#8054D9",
            ax=ax
        )

        ax.set_title("Customer Age Distribution")
        ax.set_xlabel("Age")
        ax.set_ylabel("Number of Customers")

        show_chart(fig)

    st.divider()

    st.subheader("Income and Spending Relationship")

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.scatterplot(
        data=clean_data,
        x=INCOME,
        y=SPENDING,
        hue=GENDER,
        palette="viridis",
        s=90,
        ax=ax
    )

    ax.set_title("Annual Income vs Spending Score")

    show_chart(fig)


# =========================================================
# CUSTOMER INSIGHTS
# =========================================================

elif page == "Customer Insights":

    st.header("Customer Insights")

    st.write(
        "Explore customer demographics, income distribution, "
        "spending behavior and relationships between variables."
    )

    st.subheader("Dataset Overview")

    st.write(
        f"Total records: **{len(clean_data):,}** | "
        f"Total features: **{len(clean_data.columns)}**"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("#### Age Distribution")

        fig, ax = plt.subplots(figsize=(7, 4))

        sns.histplot(
            clean_data[AGE],
            kde=True,
            color="#8054D9",
            ax=ax
        )

        ax.set_title("Age Distribution")

        show_chart(fig)

    with col2:

        st.markdown("#### Annual Income Distribution")

        fig, ax = plt.subplots(figsize=(7, 4))

        sns.histplot(
            clean_data[INCOME],
            kde=True,
            color="#18A999",
            ax=ax
        )

        ax.set_title("Annual Income Distribution")

        show_chart(fig)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("#### Spending Score Distribution")

        fig, ax = plt.subplots(figsize=(7, 4))

        sns.histplot(
            clean_data[SPENDING],
            kde=True,
            color="#E879A9",
            ax=ax
        )

        ax.set_title("Spending Score Distribution")

        show_chart(fig)

    with col2:

        st.markdown("#### Average Spending by Gender")

        gender_spending = (
            clean_data.groupby(GENDER)[SPENDING]
            .mean()
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(7, 4))

        sns.barplot(
            data=gender_spending,
            x=GENDER,
            y=SPENDING,
            hue=GENDER,
            palette="viridis",
            legend=False,
            ax=ax
        )

        ax.set_title("Average Spending by Gender")
        ax.set_ylabel("Average Spending Score")

        show_chart(fig)

    st.divider()

    st.subheader("Correlation Analysis")

    numeric_data = clean_data[NUMERIC_COLUMNS]

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.heatmap(
        numeric_data.corr(),
        annot=True,
        cmap="Purples",
        fmt=".2f",
        linewidths=0.5,
        ax=ax
    )

    ax.set_title("Correlation Between Customer Features")

    show_chart(fig)

    st.subheader("Income vs Spending by Age")

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.scatterplot(
        data=clean_data,
        x=INCOME,
        y=SPENDING,
        hue=AGE,
        palette="magma",
        size=AGE,
        sizes=(30, 150),
        ax=ax
    )

    ax.set_title("Customer Income and Spending Patterns")

    show_chart(fig)
   
    # -----------------------------------------
    # EXISTING CLUSTER DISTRIBUTION
    # -----------------------------------------

    st.divider()

    st.subheader("Existing Customer Cluster Analysis")

    cluster_distribution = (
        clean_data[CLUSTER]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.barplot(
        x=cluster_distribution.index,
        y=cluster_distribution.values,
        color="#8054D9",
        ax=ax
    )

    ax.set_title("Customer Count by Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Customer Count")

    show_chart(fig)

    st.subheader("Cluster Summary")

    cluster_summary = (
        clean_data
        .groupby(CLUSTER)
        .agg(
            Total_Customers=(CUSTOMER_ID, "count"),
            Average_Age=(AGE, "mean"),
            Average_Income=(INCOME, "mean"),
            Average_Spending=(SPENDING, "mean")
        )
        .round(2)
    )

    st.dataframe(
        cluster_summary,
        use_container_width=True
    )

# =========================================================
# CUSTOMER SEGMENTATION - EXISTING CLUSTERS
# =========================================================

elif page == "Customer Segmentation":

    st.header("Customer Segmentation")

    st.write(
        "Explore the existing customer segments assigned "
        "in your uploaded dataset."
    )

    st.info(
        "The Cluster column contains the existing K-Means "
        "assignments. This page analyzes those assignments "
        "without overwriting them."
    )

    # -----------------------------------------
    # CLUSTER VALIDATION
    # -----------------------------------------

    segmented_data = clean_data.copy()

    available_clusters = sorted(
        segmented_data[CLUSTER].unique(),
        key=lambda x: (
            x == "Unassigned",
            int(x) if x.lstrip("-").isdigit() else x
        )
    )

    if len(available_clusters) < 1:

        st.error("No customer clusters are available.")

        st.stop()

    # -----------------------------------------
    # CLUSTER FILTER
    # -----------------------------------------

    selected_clusters = st.multiselect(
        "Select Customer Clusters",
        options=available_clusters,
        default=available_clusters
    )

    if not selected_clusters:

        st.warning("Please select at least one cluster.")

        st.stop()

    filtered_data = segmented_data[
        segmented_data[CLUSTER].isin(selected_clusters)
    ].copy()

    # -----------------------------------------
    # CLUSTER METRICS
    # -----------------------------------------

    total_customers = len(filtered_data)

    total_clusters = filtered_data[CLUSTER].nunique()

    average_income = filtered_data[INCOME].mean()

    average_spending = filtered_data[SPENDING].mean()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card(
            "Total Customers",
            f"{total_customers:,}"
        )

    with col2:
        metric_card(
            "Customer Segments",
            total_clusters
        )

    with col3:
        metric_card(
            "Average Income",
            f"{average_income:.1f}k"
        )

    with col4:
        metric_card(
            "Average Spending",
            f"{average_spending:.1f}"
        )

    st.divider()

    # -----------------------------------------
    # CUSTOMER DISTRIBUTION
    # -----------------------------------------

    st.subheader("Customer Distribution by Cluster")

    cluster_counts = (
        filtered_data[CLUSTER]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        x=cluster_counts.index,
        y=cluster_counts.values,
        color="#8054D9",
        ax=ax
    )

    ax.set_title("Number of Customers in Each Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Customers")

    show_chart(fig)

    st.divider()

    # -----------------------------------------
    # CLUSTER VISUALIZATION
    # -----------------------------------------

    st.subheader("Customer Segments Visualization")

    col1, col2 = st.columns(2)

    with col1:

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.scatterplot(
            data=filtered_data,
            x=INCOME,
            y=SPENDING,
            hue=CLUSTER,
            palette="viridis",
            s=100,
            ax=ax
        )

        ax.set_title("Annual Income vs Spending by Cluster")

        show_chart(fig)

    with col2:

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.scatterplot(
            data=filtered_data,
            x=AGE,
            y=SPENDING,
            hue=CLUSTER,
            palette="viridis",
            s=100,
            ax=ax
        )

        ax.set_title("Age vs Spending by Cluster")

        show_chart(fig)

    st.divider()

    # -----------------------------------------
    # CLUSTER PROFILES
    # -----------------------------------------

    st.subheader("Customer Segment Profiles")

    cluster_profiles = (
        filtered_data
        .groupby(CLUSTER)
        .agg(
            Customer_Count=(CUSTOMER_ID, "count"),
            Average_Age=(AGE, "mean"),
            Average_Income=(INCOME, "mean"),
            Average_Spending=(SPENDING, "mean")
        )
        .round(2)
    )

    st.dataframe(
        cluster_profiles,
        use_container_width=True
    )

    st.divider()

    # -----------------------------------------
    # CUSTOMER RECORDS
    # -----------------------------------------

    st.subheader("Segmented Customer Records")

    st.dataframe(
        filtered_data[
            REQUIRED_COLUMNS
        ],
        use_container_width=True
    )

    # -----------------------------------------
    # DOWNLOAD SEGMENTED DATA
    # -----------------------------------------

    csv_clusters = (
        filtered_data[
            REQUIRED_COLUMNS
        ]
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        "Download Segmented Dataset",
        data=csv_clusters,
        file_name="customer_segments.csv",
        mime="text/csv"
    )


# =========================================================
# SPENDING PREDICTION - DECISION TREE
# =========================================================

elif page == "Spending Prediction":

    st.header("Spending Category Prediction")

    st.write(
        "Train a Decision Tree classifier to predict whether "
        "a customer belongs to the Low, Medium or High spending category."
    )

    st.info(
        "The spending categories are generated from the uploaded "
        "dataset's spending-score tertiles. Spending Score is not "
        "used as an input feature because it defines the target label."
    )

    # -----------------------------------------
    # CREATE TARGET LABELS
    # -----------------------------------------

    model_data = clean_data.copy()

    model_data["Spending Category"] = (
        create_spending_categories(
            model_data[SPENDING]
        )
    )

    # Spending score is deliberately excluded.
    feature_columns = [
        AGE,
        INCOME,
        GENDER
    ]

    X = model_data[feature_columns]

    y = model_data["Spending Category"]

    class_counts = y.value_counts()

    st.subheader("Spending Category Distribution")

    col1, col2 = st.columns(2)

    with col1:

        fig, ax = plt.subplots(figsize=(7, 4))

        sns.countplot(
            data=model_data,
            x="Spending Category",
            hue="Spending Category",
            order=["Low", "Medium", "High"],
            palette=[
                "#E879A9",
                "#8054D9",
                "#18A999"
            ],
            legend=False,
            ax=ax
        )

        ax.set_title("Customer Spending Categories")

        show_chart(fig)

    with col2:

        st.markdown("#### Category Counts")

        st.dataframe(
            class_counts.rename("Customers"),
            use_container_width=True
        )

    st.divider()

    # -----------------------------------------
    # TRAIN TEST SPLIT
    # -----------------------------------------

    if y.nunique() < 2:

        st.error(
            "The uploaded dataset does not contain enough "
            "variation to train a classification model."
        )

        st.stop()

    can_stratify = (
        class_counts.min() >= 2
        and len(class_counts) * 2 <= len(model_data)
    )

    stratify_values = y if can_stratify else None

    try:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=stratify_values
        )

    except ValueError as error:

        st.error(
            f"Unable to split this dataset for classification: {error}"
        )

        st.stop()

    # -----------------------------------------
    # MODEL PIPELINE
    # -----------------------------------------

    preprocessor = create_preprocessor()

    classifier = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=4,
        random_state=42
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    # -----------------------------------------
    # MODEL PERFORMANCE
    # -----------------------------------------

    st.subheader("Model Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        metric_card(
            "Accuracy",
            f"{accuracy * 100:.2f}%",
            "#8054D9"
        )

    with col2:
        metric_card(
            "Training Records",
            len(X_train),
            "#18A999"
        )

    with col3:
        metric_card(
            "Testing Records",
            len(X_test),
            "#E879A9"
        )

    st.divider()

    # -----------------------------------------
    # CONFUSION MATRIX
    # -----------------------------------------

    st.subheader("Confusion Matrix")

    labels_order = [
        label for label in ["Low", "Medium", "High"]
        if label in y.unique()
    ]

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=labels_order
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Purples",
        xticklabels=labels_order,
        yticklabels=labels_order,
        linewidths=0.5,
        ax=ax
    )

    ax.set_xlabel("Predicted Category")
    ax.set_ylabel("Actual Category")
    ax.set_title("Decision Tree Confusion Matrix")

    show_chart(fig)

    # -----------------------------------------
    # CLASSIFICATION REPORT
    # -----------------------------------------

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

    st.divider()

    # -----------------------------------------
    # CUSTOMER PREDICTION FORM
    # -----------------------------------------

    st.subheader(
        "Predict a New Customer's Spending Category"
    )

    st.write(
        "Enter customer details to predict their spending category."
    )

    with st.form("customer_prediction_form"):

        col1, col2, col3 = st.columns(3)

        with col1:

            new_age = st.number_input(
                "Customer Age",
                min_value=1,
                max_value=120,
                value=30
            )

        with col2:

            new_income = st.number_input(
                "Annual Income (k$)",
                min_value=0.0,
                value=float(clean_data[INCOME].median()),
                step=1.0
            )

        with col3:

            gender_options = sorted(
                clean_data[GENDER].dropna().unique().tolist()
            )

            new_gender = st.selectbox(
                "Gender",
                options=gender_options
            )

        predict_button = st.form_submit_button(
            "Predict Spending Category"
        )

    if predict_button:

        new_customer = pd.DataFrame({
            AGE: [new_age],
            INCOME: [new_income],
            GENDER: [new_gender]
        })

        predicted_category = model.predict(
            new_customer
        )[0]

        probability = model.predict_proba(
            new_customer
        )[0]

        st.success(
            f"Predicted Spending Category: {predicted_category}"
        )

        probability_df = pd.DataFrame({
            "Category": model.classes_,
            "Probability (%)": probability * 100
        })

        st.dataframe(
            probability_df.round(2),
            use_container_width=True
        )
        # =========================================================
# DATASET MANAGEMENT
# =========================================================

elif page == "Dataset Management":

    st.header("Dataset Management")

    st.write(
        "View your uploaded dataset, inspect data quality, "
        "and download the cleaned customer records."
    )
    st.dataframe(
    raw_data[REQUIRED_COLUMNS].head(20),
    use_container_width=True
)

    # -----------------------------------------
    # DATASET METRICS
    # -----------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card(
            "Total Records",
            len(raw_data),
            "#8054D9"
        )

    with col2:
        metric_card(
            "Total Columns",
            len(raw_data.columns),
            "#18A999"
        )

    with col3:
        metric_card(
            "Duplicate Rows",
            duplicate_count,
            "#E879A9"
        )

    with col4:
        metric_card(
            "Missing Values",
            int(raw_data.isnull().sum().sum()),
            "#E89B45"
        )

    st.divider()

    # -----------------------------------------
    # RAW DATA PREVIEW
    # -----------------------------------------

    st.subheader("Original Dataset Preview")

    st.dataframe(
        raw_data.head(20),
        use_container_width=True
    )

    st.divider()

    # -----------------------------------------
    # DATA INFORMATION
    # -----------------------------------------

    st.subheader("Dataset Information")

    info_df = pd.DataFrame({
        "Column": raw_data.columns,
        "Data Type": raw_data.dtypes.astype(str).values,
        "Missing Values": raw_data.isnull().sum().values,
        "Unique Values": raw_data.nunique().values
    })

    st.dataframe(
        info_df,
        use_container_width=True
    )

    st.divider()

# -----------------------------------------
# MISSING VALUE ANALYSIS
# -----------------------------------------

st.subheader("Missing Values Analysis")

# Detect missing values and blank strings
missing_data = raw_data.copy()

missing_data = missing_data.replace(
    r"^\s*$",
    np.nan,
    regex=True
)

# Calculate missing values
missing_df = pd.DataFrame({
    "Column": missing_data.columns,
    "Missing Values": missing_data.isnull().sum().values
})

missing_df["Missing Percentage"] = (
    missing_df["Missing Values"] /
    len(missing_data) * 100
)

# Overall statistics
total_missing = int(
    missing_df["Missing Values"].sum()
)

total_cells = (
    len(missing_data) * len(missing_data.columns)
)

missing_percentage = (
    total_missing / total_cells * 100
    if total_cells > 0 else 0
)

# KPI metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Missing Values",
        f"{total_missing:,}"
    )

with col2:
    st.metric(
        "Missing Data Percentage",
        f"{missing_percentage:.2f}%"
    )

with col3:
    complete_columns = (
        missing_df["Missing Values"] == 0
    ).sum()

    st.metric(
        "Complete Columns",
        f"{complete_columns}/{len(missing_df)}"
    )

st.divider()

# Filter columns containing missing values
columns_with_missing = missing_df[
    missing_df["Missing Values"] > 0
].copy()

# Display graph only when missing values exist
if total_missing == 0:

    st.success(
        "Excellent! Your uploaded dataset contains "
        "no missing values. All columns are complete."
    )

    st.markdown(
        """
        ### Dataset Quality

        Your dataset has passed the missing-value check.
        There is no missing data to visualize.
        """
    )

else:

    st.warning(
        f"{total_missing:,} missing values detected "
        f"across {len(columns_with_missing)} columns."
    )

    st.dataframe(
        columns_with_missing.style.format({
            "Missing Percentage": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

    # Create graph only if missing values exist
    fig, ax = plt.subplots(figsize=(10, 5))

    plot_data = columns_with_missing.sort_values(
        "Missing Values",
        ascending=True
    )

    sns.barplot(
        data=plot_data,
        x="Missing Values",
        y="Column",
        color="#8054D9",
        ax=ax
    )

    ax.set_title(
        "Missing Values by Dataset Column",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel("Number of Missing Values")
    ax.set_ylabel("Dataset Columns")

    ax.grid(axis="x", alpha=0.25)

    show_chart(fig)


    # -----------------------------------------
    # CLEANED DATA PREVIEW
    # -----------------------------------------

    st.subheader("Cleaned Dataset")

    st.caption(
        "Duplicate records are removed and missing numeric "
        "and categorical values are filled."
    )

    st.dataframe(
        clean_data,
        use_container_width=True
    )

    # -----------------------------------------
    # DOWNLOAD CLEANED DATASET
    # -----------------------------------------

    cleaned_csv = clean_data.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Cleaned Dataset",
        data=cleaned_csv,
        file_name="cleaned_customer_dataset.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.success(
        "Your uploaded dataset has been validated and processed successfully."
    )


# =========================================================
# FOOTER
# =========================================================

st.sidebar.divider()

st.sidebar.caption(
    "Customer Intelligence System | Machine Learning Dashboard"
)