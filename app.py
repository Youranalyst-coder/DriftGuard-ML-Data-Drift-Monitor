import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from drift_detector import detect_drift

st.set_page_config(
    page_title="DriftGuard – ML Drift Monitor",
    layout="wide"
)

st.title("🔍 DriftGuard – ML Data Drift Monitor")
st.write(
    "Compare your training (reference) data with new (current) data "
    "to detect distribution shifts that may require model retraining."
)

# Sidebar inputs
st.sidebar.header("Upload Data")

ref_file = st.sidebar.file_uploader("Reference data (CSV)", type=["csv"])
cur_file = st.sidebar.file_uploader("Current data (CSV)", type=["csv"])

psi_threshold = st.sidebar.slider("PSI Drift Threshold", 0.0, 1.0, 0.2, 0.05)
ks_threshold = st.sidebar.slider("KS Drift Threshold", 0.0, 1.0, 0.1, 0.05)

if st.sidebar.button("Run Drift Check"):

    if ref_file is None or cur_file is None:
        st.error("Please upload both reference and current CSV files.")
    else:
        reference_df = pd.read_csv(ref_file)
        current_df = pd.read_csv(cur_file)

        st.subheader("📄 Data Preview")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Reference Data**")
            st.dataframe(reference_df.head())
        with col2:
            st.markdown("**Current Data**")
            st.dataframe(current_df.head())

        # Run detection
        overall_drift, result_df = detect_drift(
            reference_df,
            current_df,
            psi_threshold=psi_threshold,
            ks_threshold=ks_threshold
        )

        st.subheader("📊 Drift Summary")

        if overall_drift:
            st.error("⚠️ Drift Detected in one or more features.")
        else:
            st.success("✅ No significant drift detected based on given thresholds.")

        if not result_df.empty:
            st.dataframe(result_df)

            # Visualize top N drifted features
            st.subheader("📈 Feature-wise Drift Visualization")

            top_n = st.slider("Number of top features to plot by PSI", 1, min(10, len(result_df)), 5)
            top_features = result_df.head(top_n)["feature"].tolist()

            for feat in top_features:
                st.markdown(f"#### Feature: `{feat}`")

                fig, ax = plt.subplots()
                ax.hist(
                    reference_df[feat].dropna(),
                    bins=30,
                    alpha=0.5,
                    label="Reference"
                )
                ax.hist(
                    current_df[feat].dropna(),
                    bins=30,
                    alpha=0.5,
                    label="Current"
                )
                ax.set_xlabel(feat)
                ax.set_ylabel("Count")
                ax.legend()
                st.pyplot(fig)
        else:
            st.info("No numeric features found or unable to compute drift.")
else:
    st.info("Upload CSV files and click **Run Drift Check** to start.")
