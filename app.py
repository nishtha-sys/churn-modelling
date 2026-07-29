import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Churn Model Comparison",
    page_icon="📊",
    layout="wide"
)

# ---------------------------
# Header
# ---------------------------
st.title("📊 Customer Churn — ML Model Comparison")

st.markdown(
    """
    This app trains and compares **4 machine learning models**
    (Logistic Regression, Decision Tree, Random Forest, XGBoost) to predict
    whether a customer will **churn** (leave a company/bank), and shows
    which one performs best on your data.

    Upload your own CSV, or click below to try it instantly with the
    included bank customer churn dataset.
    """
)

# ---------------------------
# Data Source
# ---------------------------
SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "Churn_Modelling.csv")

col_a, col_b = st.columns([1, 2])

with col_a:
    use_sample = st.button("⚡ Try with Sample Dataset", use_container_width=True)

with col_b:
    uploaded_file = st.file_uploader("Or upload your own CSV Dataset", type=["csv"])

df = None

if use_sample:
    if os.path.exists(SAMPLE_PATH):
        df = pd.read_csv(SAMPLE_PATH)
        st.session_state["df"] = df
    else:
        st.error("Sample dataset not found in the repo.")
elif uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state["df"] = df
elif "df" in st.session_state:
    df = st.session_state["df"]

if df is not None:

    tab_data, tab_results, tab_matrix = st.tabs(
        ["🔍 Data Overview", "🏆 Model Performance", "🧩 Confusion Matrix"]
    )

    # ---------------------------
    # Tab 1: Data Overview
    # ---------------------------
    with tab_data:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing Values", int(df.isnull().sum().sum()))

        with st.expander("Column Data Types"):
            st.write(df.dtypes)

        default_index = list(df.columns).index("Exited") if "Exited" in df.columns else len(df.columns) - 1
        target = st.selectbox(
            "Select Target Column (what you want to predict)",
            df.columns,
            index=default_index,
            help="Pick a categorical column with a small number of classes "
                 "(e.g. Exited, Geography) — not a continuous number like salary."
        )

        n_unique = df[target].nunique()
        n_rows = len(df)

        # Guardrail: this app does classification, not regression.
        # A near-continuous column (like EstimatedSalary) will break stratified
        # splitting and isn't something a classifier should predict anyway.
        if n_unique > 20 or n_unique > 0.5 * n_rows:
            st.error(
                f"⚠️ **'{target}'** has {n_unique} unique values — too many for a "
                "classification target (it looks continuous, like a salary or ID). "
                "This app predicts categories (e.g. Exited: 0/1, Geography: 3 countries). "
                "Please pick a column with a small, fixed set of categories instead."
            )
            st.stop()

        # Guardrail: every class needs at least 2 samples for a stratified split.
        value_counts = df[target].value_counts()
        if (value_counts < 2).any():
            st.error(
                f"⚠️ **'{target}'** has at least one category with only 1 sample, "
                "which isn't enough to split into train/test sets. Please pick a "
                "different target column."
            )
            st.stop()

    # ---------------------------
    # Preprocessing (shared across tabs)
    # ---------------------------
    drop_cols = [target]
    for col in ["RowNumber", "CustomerId", "Surname"]:
        if col in df.columns:
            drop_cols.append(col)

    X = df.drop(columns=drop_cols)
    y = df[target]

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna("Unknown")

    if pd.api.types.is_numeric_dtype(y):
        y = y.fillna(y.mode()[0])
    else:
        y = y.fillna("Unknown")

    X = pd.get_dummies(X, drop_first=True)
    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype(int)

    if not pd.api.types.is_numeric_dtype(y):
        le = LabelEncoder()
        y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    n_classes = len(np.unique(y))
    # "logloss" is valid for binary targets only; multi-class targets
    # (e.g. Geography with 3 countries) need "mlogloss", otherwise XGBoost
    # raises a ValueError at fit time.
    xgb_eval_metric = "logloss" if n_classes == 2 else "mlogloss"

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(eval_metric=xgb_eval_metric, random_state=42)
    }

    results = []
    trained_models = {}

    with st.spinner("Training models..."):
        for name, model in models.items():
            model.fit(X_train, y_train)
            trained_models[name] = model
            pred = model.predict(X_test)

            results.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred, average="weighted", zero_division=0),
                "Recall": recall_score(y_test, pred, average="weighted", zero_division=0),
                "F1 Score": f1_score(y_test, pred, average="weighted", zero_division=0)
            })

    results_df = pd.DataFrame(results)
    best = results_df.sort_values(by="Accuracy", ascending=False).iloc[0]

    # ---------------------------
    # Tab 2: Model Performance
    # ---------------------------
    with tab_results:
        st.subheader("🏆 Best Performing Model")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Best Model", best["Model"])
        m2.metric("Accuracy", f"{best['Accuracy']:.2%}")
        m3.metric("Precision", f"{best['Precision']:.2%}")
        m4.metric("F1 Score", f"{best['F1 Score']:.2%}")

        st.divider()

        st.subheader("Full Comparison Table")
        st.dataframe(
            results_df.style.highlight_max(axis=0, color="lightgreen"),
            use_container_width=True
        )

        st.subheader("Accuracy by Model")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(
            data=results_df, x="Model", y="Accuracy",
            hue="Model", legend=False, palette="viridis", ax=ax
        )
        plt.xticks(rotation=15)
        st.pyplot(fig)

    # ---------------------------
    # Tab 3: Confusion Matrix
    # ---------------------------
    with tab_matrix:
        st.subheader("Confusion Matrix Explorer")
        selected_model = st.selectbox("Select a model", list(trained_models.keys()))

        model = trained_models[selected_model]
        pred = model.predict(X_test)
        cm = confusion_matrix(y_test, pred)

        fig2, ax2 = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2)
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Actual")
        ax2.set_title(selected_model)
        st.pyplot(fig2)

        st.caption(
            "Rows = actual outcome, Columns = predicted outcome. "
            "The diagonal shows correct predictions; off-diagonal cells are errors."
        )

else:
    st.info("👆 Click **Try with Sample Dataset** or upload a CSV to get started.")