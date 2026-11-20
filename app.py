import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="💳",
    layout="wide"
)

# ---------------------------------------------------------
# LOAD MODEL FILES
# ---------------------------------------------------------

MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURES_PATH = "models/features.pkl"
RESULTS_PATH = "models/model_results.csv"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
features = joblib.load(FEATURES_PATH)

results = pd.read_csv(RESULTS_PATH)

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .metric-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #444;
        text-align: center;
    }

    .metric-title {
        font-size: 16px;
        font-weight: bold;
    }

    .metric-value {
        font-size: 28px;
        font-weight: bold;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">💳 Credit Card Fraud Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning Based Transaction Classification System</div>',
    unsafe_allow_html=True
)

st.divider()

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🏠 Dashboard",
        "🔍 Fraud Prediction",
        "🤖 Model Performance",
        "📊 Data Visualization"
    ]
)

# =========================================================
# TAB 1 - DASHBOARD
# =========================================================

with tab1:

    st.header("📊 Project Dashboard")

    st.write(
        "This system uses Machine Learning algorithms to classify "
        "credit card transactions as legitimate or fraudulent."
    )

    st.divider()

    # Dataset statistics

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Transactions",
            "284,807"
        )

    with col2:
        st.metric(
            "Legitimate Transactions",
            "284,315"
        )

    with col3:
        st.metric(
            "Fraud Transactions",
            "492"
        )

    with col4:
        st.metric(
            "Best Model",
            "Random Forest"
        )

    st.divider()

    st.subheader("🎯 Best Model Performance")

    best_row = results.loc[
        results["Model"].str.lower().str.contains("random forest")
    ].iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Accuracy",
            f"{best_row['Accuracy'] * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Precision",
            f"{best_row['Precision'] * 100:.2f}%"
        )

    with col3:
        st.metric(
            "Recall",
            f"{best_row['Recall'] * 100:.2f}%"
        )

    with col4:
        st.metric(
            "F1 Score",
            f"{best_row['F1 Score'] * 100:.2f}%"
        )

    st.divider()

    st.subheader("📌 Project Workflow")

    st.write("""
    **1. Dataset Collection**  
    ↓  
    **2. Data Preprocessing**  
    ↓  
    **3. Feature Scaling**  
    ↓  
    **4. Machine Learning Model Training**  
    ↓  
    **5. Model Evaluation**  
    ↓  
    **6. Fraud Prediction**
    """)

    st.info(
        "The dataset contains highly imbalanced classes. "
        "Therefore, Precision, Recall and F1 Score are important "
        "along with Accuracy."
    )


# =========================================================
# TAB 2 - FRAUD PREDICTION
# =========================================================

with tab2:

    st.header("🔍 Check a Transaction")

    st.write(
        "Enter transaction details or use a real example "
        "from the dataset."
    )

    st.divider()

    # -----------------------------------------------------
    # LOAD DEMO EXAMPLES
    # -----------------------------------------------------

    demo_path = "data/demo_examples.csv"

    selected_example = "Manual Input"

    if os.path.exists(demo_path):

        demo_data = pd.read_csv(demo_path)

        selected_example = st.selectbox(
            "🎯 Select Transaction Type",
            [
                "Manual Input",
                "Legitimate Transaction",
                "Fraud Transaction"
            ]
        )

    else:

        st.warning(
            "demo_examples.csv not found. "
            "Please run create_demo_examples.py first."
        )

        demo_data = None

    # -----------------------------------------------------
    # SELECT DEMO ROW
    # -----------------------------------------------------

    demo_row = None

    if (
        demo_data is not None
        and selected_example != "Manual Input"
    ):

        demo_row = demo_data[
            demo_data["Example"] == selected_example
        ].iloc[0]

        if selected_example == "Fraud Transaction":

            st.error(
                "🚨 Real fraud example selected from the dataset."
            )

        else:

            st.success(
                "✅ Real legitimate example selected from the dataset."
            )

    else:

        st.info(
            "💡 Choose a demo transaction above or enter "
            "your own values manually."
        )

    st.divider()

    # -----------------------------------------------------
    # INPUT FEATURES
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    input_values = {}

    for i, feature in enumerate(features):

        # Get default value
        if demo_row is not None:
            default_value = float(demo_row[feature])
        else:

            if feature == "Amount":
                default_value = 100.0
            else:
                default_value = 0.0

        # Put inputs into two columns
        if i % 2 == 0:

            with col1:

                input_values[feature] = st.number_input(
                    feature,
                    value=default_value,
                    format="%.6f",
                    key=f"{feature}_{selected_example}_left"
                )

        else:

            with col2:

                input_values[feature] = st.number_input(
                    feature,
                    value=default_value,
                    format="%.6f",
                    key=f"{feature}_{selected_example}_right"
                )

    st.divider()

    # -----------------------------------------------------
    # PREDICTION BUTTON
    # -----------------------------------------------------

    check_button = st.button(
        "🔎 Check Transaction",
        type="primary",
        width="stretch"
    )

    if check_button:

        # Create dataframe
        input_data = pd.DataFrame(
            [input_values],
            columns=features
        )

        try:

            # Scale data
            input_scaled = scaler.transform(input_data)

            # Make prediction
            prediction = model.predict(input_scaled)[0]

            # Fraud probability
            if hasattr(model, "predict_proba"):

                probability = model.predict_proba(
                    input_scaled
                )[0][1]

            else:

                probability = None

            st.divider()

            # -------------------------------------------------
            # RESULT
            # -------------------------------------------------

            if prediction == 1:

                st.error(
                    "🚨 FRAUDULENT TRANSACTION DETECTED"
                )

                st.warning(
                    "The Random Forest Machine Learning model "
                    "classified this transaction as potentially fraudulent."
                )

            else:

                st.success(
                    "✅ LEGITIMATE TRANSACTION"
                )

                st.info(
                    "The Random Forest Machine Learning model "
                    "classified this transaction as legitimate."
                )

            # -------------------------------------------------
            # PROBABILITY
            # -------------------------------------------------

            if probability is not None:

                st.metric(
                    "Fraud Probability",
                    f"{probability * 100:.2f}%"
                )

                st.progress(
                    float(probability)
                )

            # -------------------------------------------------
            # DEMO VERIFICATION
            # -------------------------------------------------

            if demo_row is not None:

                actual_class = int(
                    demo_row["Class"]
                )

                st.divider()

                st.subheader(
                    "📋 Dataset Verification"
                )

                if actual_class == 1:

                    st.write(
                        "**Actual Dataset Class:** 🚨 Fraud"
                    )

                else:

                    st.write(
                        "**Actual Dataset Class:** ✅ Legitimate"
                    )

                if prediction == actual_class:

                    st.success(
                        "✅ Model prediction matches the actual dataset class."
                    )

                else:

                    st.warning(
                        "⚠️ Model prediction differs from the actual dataset class."
                    )

        except Exception as e:

            st.error(
                f"Prediction error: {e}"
            )

# =========================================================
# TAB 3 - MODEL PERFORMANCE
# =========================================================

with tab3:

    st.header("🤖 Machine Learning Model Performance")

    st.write(
        "Four Machine Learning algorithms were trained and "
        "evaluated on the credit card transaction dataset."
    )

    st.divider()

    # -----------------------------------------------------
    # MODEL COMPARISON TABLE
    # -----------------------------------------------------

    st.subheader("📋 Model Comparison")

    display_results = results.copy()

    # Convert values to percentages
    for column in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]:

        display_results[column] = (
            display_results[column] * 100
        ).round(2).astype(str) + "%"

    st.dataframe(
        display_results,
        width="stretch",
        hide_index=True
    )

    st.divider()

    # -----------------------------------------------------
    # BEST MODEL
    # -----------------------------------------------------

    st.subheader("🏆 Best Model")

    best_row = results.loc[
        results["Model"].str.lower().str.contains("random forest")
    ].iloc[0]

    st.success(
        "Random Forest was selected as the best supervised "
        "learning model."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Accuracy",
            f"{best_row['Accuracy'] * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Precision",
            f"{best_row['Precision'] * 100:.2f}%"
        )

    with col3:
        st.metric(
            "Recall",
            f"{best_row['Recall'] * 100:.2f}%"
        )

    with col4:
        st.metric(
            "F1 Score",
            f"{best_row['F1 Score'] * 100:.2f}%"
        )

    st.divider()

    # -----------------------------------------------------
    # EXPLANATION
    # -----------------------------------------------------

    st.subheader("💡 Why Random Forest?")

    st.write("""
    Random Forest is an ensemble Machine Learning algorithm
    that combines multiple decision trees to make a final
    prediction.

    In our experiment, Random Forest achieved the highest
    F1 Score among the supervised learning models.

    It achieved very high precision while also detecting a
    significant portion of fraudulent transactions.

    Therefore, Random Forest was selected as the main model
    for the fraud prediction application.
    """)

    st.divider()

    # -----------------------------------------------------
    # IMPORTANT METRIC EXPLANATION
    # -----------------------------------------------------

    st.subheader("📚 Evaluation Metrics")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Accuracy")

        st.write(
            "Percentage of total transactions classified correctly."
        )

        st.markdown("### Precision")

        st.write(
            "Out of transactions predicted as fraud, "
            "how many were actually fraud."
        )

    with col2:

        st.markdown("### Recall")

        st.write(
            "Out of all actual fraud transactions, "
            "how many were successfully detected."
        )

        st.markdown("### F1 Score")

        st.write(
            "A balance between Precision and Recall."
        )

    st.warning(
        "⚠️ The dataset is highly imbalanced. Therefore, "
        "Accuracy alone is not enough to evaluate a fraud "
        "detection model. Precision, Recall and F1 Score "
        "are also important."
    )

# =========================================================
# TAB 4 - DATA VISUALIZATION
# =========================================================

with tab4:

    st.header("📊 Data Visualization")

    st.write(
        "Visual analysis of the dataset and Machine Learning "
        "model performance."
    )

    st.divider()

    graph_folder = "graphs"

    # -----------------------------------------------------
    # CLASS DISTRIBUTION
    # -----------------------------------------------------

    st.subheader("📌 Dataset Class Distribution")

    if os.path.exists(
        os.path.join(graph_folder, "class_distribution.png")
    ):

        st.image(
            os.path.join(
                graph_folder,
                "class_distribution.png"
            )
        )

        st.info(
            "The dataset is highly imbalanced. "
            "There are 284,315 legitimate transactions "
            "and only 492 fraudulent transactions."
        )

    st.divider()

    # -----------------------------------------------------
    # MODEL COMPARISON GRAPHS
    # -----------------------------------------------------

    st.subheader("🤖 Model Performance Comparison")

    col1, col2 = st.columns(2)

    # Accuracy
    with col1:

        if os.path.exists(
            os.path.join(
                graph_folder,
                "accuracy_comparison.png"
            )
        ):

            st.image(
                os.path.join(
                    graph_folder,
                    "accuracy_comparison.png"
                ),
                caption="Accuracy Comparison"
            )

    # Precision
    with col2:

        if os.path.exists(
            os.path.join(
                graph_folder,
                "precision_comparison.png"
            )
        ):

            st.image(
                os.path.join(
                    graph_folder,
                    "precision_comparison.png"
                ),
                caption="Precision Comparison"
            )

    # Recall and F1
    col3, col4 = st.columns(2)

    with col3:

        if os.path.exists(
            os.path.join(
                graph_folder,
                "recall_comparison.png"
            )
        ):

            st.image(
                os.path.join(
                    graph_folder,
                    "recall_comparison.png"
                ),
                caption="Recall Comparison"
            )

    with col4:

        if os.path.exists(
            os.path.join(
                graph_folder,
                "f1_score_comparison.png"
            )
        ):

            st.image(
                os.path.join(
                    graph_folder,
                    "f1_score_comparison.png"
                ),
                caption="F1 Score Comparison"
            )

    st.divider()

    # -----------------------------------------------------
    # CONFUSION MATRIX
    # -----------------------------------------------------

    st.subheader("🎯 Confusion Matrix")

    if os.path.exists(
        os.path.join(
            graph_folder,
            "confusion_matrix.png"
        )
    ):

        st.image(
            os.path.join(
                graph_folder,
                "confusion_matrix.png"
            )
        )

    st.info(
        "The confusion matrix shows how many legitimate "
        "and fraudulent transactions were correctly or "
        "incorrectly classified by the Random Forest model."
    )

    st.divider()

    # -----------------------------------------------------
    # SIMPLE INTERPRETATION
    # -----------------------------------------------------

    st.subheader("📝 Visualization Summary")

    st.write("""
    **Class Distribution:**  
    The dataset contains a very small number of fraud
    transactions compared with legitimate transactions.

    **Accuracy:**  
    Random Forest achieved the highest accuracy among
    the evaluated models.

    **Precision:**  
    Random Forest achieved very high precision, meaning
    most transactions predicted as fraud were actually fraud.

    **Recall:**  
    Random Forest detected a significant portion of the
    actual fraudulent transactions.

    **F1 Score:**  
    Random Forest achieved the highest F1 Score among
    the supervised learning models.

    **Confusion Matrix:**  
    It provides a detailed view of correct and incorrect
    predictions.
    """)
# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "Credit Card Fraud Detection using Machine Learning | "
    "Python • Pandas • Scikit-learn • Streamlit"
)
