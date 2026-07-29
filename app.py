import streamlit as st
import pandas as pd
import numpy as np

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

st.set_page_config(page_title="ML Model Comparison", layout="wide")

st.title("Machine Learning Model Comparison")

uploaded_file = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    # ---------------------------
    # Read Dataset
    # ---------------------------
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Information")
    st.write(df.dtypes)

    # ---------------------------
    # Select Target Column
    # ---------------------------
    target = st.selectbox(
        "Select Target Column",
        df.columns
    )

    # ---------------------------
    # Remove unwanted ID columns
    # ---------------------------
    drop_cols = [target]

    for col in ["RowNumber", "CustomerId", "Surname"]:
        if col in df.columns:
            drop_cols.append(col)

    X = df.drop(columns=drop_cols)
    y = df[target]

    # ---------------------------
    # Handle Missing Values
    # ---------------------------
    for col in X.columns:

        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].median())

        else:
            X[col] = X[col].fillna("Unknown")

    if pd.api.types.is_numeric_dtype(y):
        y = y.fillna(y.mode()[0])

    else:
        y = y.fillna("Unknown")

    # ---------------------------
    # Encode Features
    # ---------------------------
    X = pd.get_dummies(X, drop_first=True)

    # Convert bool columns to int
    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype(int)

    # ---------------------------
    # Encode Target
    # ---------------------------
    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)

    # ---------------------------
    # Train/Test Split
    # ---------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ---------------------------
    # Models
    # ---------------------------
    models = {
        "Logistic Regression":
            LogisticRegression(max_iter=1000),

        "Decision Tree":
            DecisionTreeClassifier(random_state=42),

        "Random Forest":
            RandomForestClassifier(random_state=42),

        "XGBoost":
            XGBClassifier(
                eval_metric="logloss",
                random_state=42
            )
    }

    results = []

    trained_models = {}

    st.subheader("Training Models...")

    # ---------------------------
    # Train Models
    # ---------------------------
    for name, model in models.items():

        model.fit(X_train, y_train)

        trained_models[name] = model

        pred = model.predict(X_test)

        results.append({

            "Model": name,

            "Accuracy":
                accuracy_score(y_test, pred),

            "Precision":
                precision_score(
                    y_test,
                    pred,
                    average="weighted",
                    zero_division=0
                ),

            "Recall":
                recall_score(
                    y_test,
                    pred,
                    average="weighted",
                    zero_division=0
                ),

            "F1 Score":
                f1_score(
                    y_test,
                    pred,
                    average="weighted",
                    zero_division=0
                )
        })

    # ---------------------------
    # Results
    # ---------------------------
    results_df = pd.DataFrame(results)

    st.subheader("Performance Comparison")

    st.dataframe(
        results_df.style.highlight_max(
            axis=0,
            color="lightgreen"
        )
    )

    # ---------------------------
    # Accuracy Chart
    # ---------------------------
    st.subheader("Accuracy Comparison")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.barplot(
        data=results_df,
        x="Model",
        y="Accuracy",
        hue="Model",
        legend=False,
        palette="viridis",
        ax=ax
    )

    plt.xticks(rotation=15)

    st.pyplot(fig)

    # ---------------------------
    # Best Model
    # ---------------------------
    best = results_df.sort_values(
        by="Accuracy",
        ascending=False
    ).iloc[0]

    st.success(
        f"🏆 Best Model: {best['Model']} "
        f"(Accuracy = {best['Accuracy']:.4f})"
    )

    # ---------------------------
    # Confusion Matrix
    # ---------------------------
    st.subheader("Confusion Matrix")

    selected_model = st.selectbox(
        "Select Model",
        list(trained_models.keys())
    )

    model = trained_models[selected_model]

    pred = model.predict(X_test)

    cm = confusion_matrix(y_test, pred)

    fig2, ax2 = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax2
    )

    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    ax2.set_title(selected_model)

    st.pyplot(fig2)