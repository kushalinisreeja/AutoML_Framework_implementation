"""
Streamlit Web Dashboard for AutoML Framework.
Drag-and-drop CSV dataset training, automated benchmarking, interactive charts,
leaderboards, explainability, and single-click production model download.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import os
import tempfile
from sklearn.datasets import load_iris, load_breast_cancer, fetch_california_housing, load_wine

from automl import AutoML
from automl.preprocessing.cleaner import TaskDetector

st.set_page_config(
    page_title="AutoML Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 AutoML Framework Interactive Studio")
st.markdown("Automated end-to-end Machine Learning: Data Preprocessing, Feature Engineering, Multi-Model Tuning, Ensembling, & Model Export.")

# ----------------- Sidebar: Dataset & Configuration -----------------
st.sidebar.header("📁 1. Dataset Selection")

data_source = st.sidebar.radio(
    "Choose Data Source:",
    ["Sample Dataset", "Upload CSV"]
)

df = None
if data_source == "Sample Dataset":
    sample_choice = st.sidebar.selectbox(
        "Select Sample Dataset:",
        ["Breast Cancer (Classification)", "Iris Flowers (Multiclass)", "California Housing (Regression)", "Wine Recognition (Multiclass)"]
    )
    if "Breast Cancer" in sample_choice:
        dataset = load_breast_cancer(as_frame=True)
        df = dataset.frame
        target_default = "target"
    elif "Iris" in sample_choice:
        dataset = load_iris(as_frame=True)
        df = dataset.frame
        target_default = "target"
    elif "California Housing" in sample_choice:
        dataset = fetch_california_housing(as_frame=True)
        df = dataset.frame.sample(n=1200, random_state=42)
        target_default = "MedHouseVal"
    else:
        dataset = load_wine(as_frame=True)
        df = dataset.frame
        target_default = "target"
else:
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            target_default = df.columns[-1]
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {e}")

if df is not None:
    st.sidebar.header("⚙️ 2. AutoML Parameters")
    target_col = st.sidebar.selectbox("Select Target Column", options=df.columns, index=list(df.columns).index(target_default) if target_default in df.columns else len(df.columns)-1)
    
    # Auto task detection preview
    task_info = TaskDetector.detect_task(df[target_col])
    detected_task = task_info["task"]
    
    task_option = st.sidebar.selectbox(
        "Task Type",
        options=["auto", "classification", "regression"],
        index=0,
        help=f"Auto-detected as: {detected_task.upper()}"
    )

    time_budget = st.sidebar.slider("Time Budget (seconds)", min_value=15, max_value=600, value=60, step=15)
    cv_folds = st.sidebar.slider("Cross-Validation Folds", min_value=3, max_value=10, value=5)
    trials_per_model = st.sidebar.slider("Hyperparameter Trials per Model", min_value=2, max_value=20, value=5)

    enable_ensemble = st.sidebar.checkbox("Enable Ensembling (Voting & Stacking)", value=True)
    ensemble_type = "both" if enable_ensemble else "none"

    run_btn = st.sidebar.button("🚀 Run AutoML", type="primary", use_container_width=True)

    # ----------------- Main View -----------------
    tab_data, tab_results, tab_predict = st.tabs(["📊 Dataset Overview", "🏆 AutoML Results & Leaderboard", "🔮 Live Prediction"])

    with tab_data:
        st.subheader("Dataset Preview")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Rows", df.shape[0])
        col_m2.metric("Columns", df.shape[1])
        col_m3.metric("Detected Task", detected_task.capitalize())
        col_m4.metric("Missing Values", int(df.isna().sum().sum()))

        st.dataframe(df.head(10), use_container_width=True)

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("#### Feature Data Types")
            dtypes_df = pd.DataFrame({"Data Type": df.dtypes.astype(str), "Null Count": df.isnull().sum()})
            st.dataframe(dtypes_df, use_container_width=True)

        with col_right:
            st.markdown("#### Target Distribution")
            if detected_task == "classification":
                fig_target = px.histogram(df, x=target_col, color=target_col, title="Class Frequencies")
            else:
                fig_target = px.histogram(df, x=target_col, nbins=30, marginal="box", title="Target Distribution")
            st.plotly_chart(fig_target, use_container_width=True)

    # State storage for fitted AutoML
    if "automl_model" not in st.session_state:
        st.session_state["automl_model"] = None
        st.session_state["target_col"] = None

    if run_btn:
        with tab_results:
            st.subheader("Training Progress")
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.info("Preprocessing dataset, handling missing values, and selecting features...")
            progress_bar.progress(20)

            X = df.drop(columns=[target_col])
            y = df[target_col]

            automl = AutoML(
                task=task_option,
                time_budget_secs=time_budget,
                n_splits=cv_folds,
                n_iter_per_model=trials_per_model,
                ensemble=enable_ensemble,
                ensemble_type=ensemble_type,
                verbose=0
            )

            status_text.info("Benchmarking algorithms and tuning hyperparameters...")
            progress_bar.progress(50)

            try:
                automl.fit(X, y)
                progress_bar.progress(100)
                status_text.success(f"Training completed successfully in {automl.training_time_secs_}s!")

                st.session_state["automl_model"] = automl
                st.session_state["target_col"] = target_col
                st.session_state["df_features"] = X
            except Exception as e:
                st.error(f"Error during AutoML execution: {str(e)}")

    # Display results if model exists in session
    if st.session_state["automl_model"] is not None:
        automl = st.session_state["automl_model"]
        with tab_results:
            st.markdown("---")
            st.subheader("Performance Summary")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Champion Model", automl.best_model_name_)
            m2.metric("Primary Metric", automl.primary_metric.upper())
            best_cv_score = automl.leaderboard().iloc[0]["mean_cv_score"] if not automl.leaderboard().empty else 0.0
            m3.metric("Best CV Score", f"{best_cv_score:.4f}")
            m4.metric("Total Time", f"{automl.training_time_secs_}s")

            # Leaderboard table
            st.markdown("### 🏆 Algorithm Leaderboard")
            lb_df = automl.leaderboard()
            st.dataframe(
                lb_df[["model", "mean_cv_score", "std_cv_score", "fit_time_secs", "is_ensemble"]].style.highlight_max(subset=["mean_cv_score"], color="#90EE90"),
                use_container_width=True
            )

            # Plots: Leaderboard Comparison and Feature Importance
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.markdown("#### Model Performance Comparison")
                fig_lb = px.bar(
                    lb_df,
                    x="model",
                    y="mean_cv_score",
                    error_y="std_cv_score",
                    color="is_ensemble",
                    color_discrete_map={True: "#FF6F61", False: "#4682B4"},
                    labels={"mean_cv_score": f"CV Score ({automl.primary_metric})", "model": "Algorithm", "is_ensemble": "Ensemble Model"},
                    title="Cross-Validation Score by Model"
                )
                fig_lb.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_lb, use_container_width=True)

            with col_chart2:
                st.markdown("#### Feature Importance Ranking")
                fi_df = automl.get_feature_importance(top_n=12)
                if not fi_df.empty:
                    fig_fi = px.bar(
                        fi_df.sort_values(by="importance", ascending=True),
                        x="importance",
                        y="feature",
                        orientation="h",
                        title="Top Most Influential Features",
                        labels={"importance": "Relative Importance", "feature": "Feature"}
                    )
                    st.plotly_chart(fig_fi, use_container_width=True)
                else:
                    st.info("Feature importance not available for this model type.")

            # Model Export Download Button
            st.markdown("---")
            st.subheader("💾 Export Production Model")
            with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
                tmp_path = tmp.name
            automl.save(tmp_path)
            with open(tmp_path, "rb") as f:
                model_bytes = f.read()
            try:
                os.remove(tmp_path)
            except Exception:
                pass

            st.download_button(
                label=f"⬇️ Download {automl.best_model_name_} Pipeline (.joblib)",
                data=model_bytes,
                file_name=f"automl_{automl.best_model_name_}.joblib",
                mime="application/octet-stream"
            )

        with tab_predict:
            st.subheader("🔮 Interactive Real-time Prediction Playground")
            st.write("Modify feature values below to obtain immediate predictions from the champion model:")

            features_df = st.session_state["df_features"]
            input_data = {}

            # Create input grid
            cols_per_row = 3
            cols = st.columns(cols_per_row)
            for i, col in enumerate(features_df.columns):
                curr_col = cols[i % cols_per_row]
                with curr_col:
                    if pd.api.types.is_numeric_dtype(features_df[col]):
                        val_mean = float(features_df[col].median()) if not np.isnan(features_df[col].median()) else 0.0
                        val_min = float(features_df[col].min()) if not np.isnan(features_df[col].min()) else 0.0
                        val_max = float(features_df[col].max()) if not np.isnan(features_df[col].max()) else 100.0
                        input_data[col] = st.number_input(f"{col}", value=val_mean)
                    else:
                        options = list(features_df[col].dropna().unique())
                        if not options:
                            options = ["Unknown"]
                        input_data[col] = st.selectbox(f"{col}", options=options)

            if st.button("Predict Single Sample", type="secondary"):
                test_sample = pd.DataFrame([input_data])
                pred = automl.predict(test_sample)[0]
                
                st.success(f"### Predicted Result: **{pred}**")
                if automl.task == "classification" and hasattr(automl.pipeline_.model, "predict_proba"):
                    try:
                        probs = automl.predict_proba(test_sample)[0]
                        st.markdown("#### Class Probabilities:")
                        classes = getattr(automl.pipeline_.model, "classes_", range(len(probs)))
                        prob_df = pd.DataFrame({"Class": classes, "Probability": probs})
                        st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}), use_container_width=True)
                    except Exception:
                        pass
else:
    st.info("👈 Please select a sample dataset or upload a CSV file from the sidebar to begin.")
