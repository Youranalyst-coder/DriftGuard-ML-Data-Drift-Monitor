🔍 DriftGuard – ML Data Drift Monitor

🚀 Overview

DriftGuard helps you identify when your ML model starts receiving data that is different from the data it was trained on. It compares reference (training) data with current (incoming) data and highlights drift using robust statistical measures and rich visualizations.

It is designed to be simple, modular, and production-friendly.

🌟 Key Features
🔹 Upload-Based Drift Detection

• Upload reference_data.csv and current_data.csv
• Automatic schema matching and type detection
• Instant drift computation

🔹 Multi-Metric Drift Analysis

Industry-standard metrics:
• Population Stability Index (PSI)
• Kolmogorov–Smirnov Statistic (KS Test)
• KL Divergence (Kullback–Leibler Divergence)

🔹 Feature-Wise Drift Report

• Drift score per feature
• PSI, KS, and KL values
• Drift Detected / No Drift label
• Human-readable interpretations

🔹 Interactive Visualizations

• Histogram overlays
• Top-N drifted features
• Dynamic plots with side-by-side comparison
• Informative drift banners

🔹 Clean, Modern UI

• Dark theme
• Threshold sliders
• Dataset preview
• Optimized layout for usability

💡 Why DriftGuard?

Over time, data feeding an ML model may shift. This can break predictive performance and create hidden failures.

DriftGuard helps you:
• Detect early warning signals
• Know which features are drifting
• Quantify how severe the drift is
• Decide whether retraining is needed

This makes it ideal for ML pipelines, MLOps teams, and deployed models.

📦 Installation

Requires Python 3.8 or above.

pip install -r requirements.txt

▶️ Usage

Run the Streamlit dashboard:

streamlit run app.py


Upload:
• reference_data.csv
• current_data.csv

View:
• Drift metrics
• Feature drift table
• Visual plots

🧱 Project Structure
DriftGuard/
 ├── app.py                      Streamlit UI
 ├── utils/
 │    ├── drift_metrics.py       PSI, KS, KL calculations
 │    ├── visualizations.py      Plotting functions
 │    ├── helpers.py             Data cleaning and utilities
 ├── assets/                     Images and static files
 ├── requirements.txt
 └── README.md

🛠️ Tech Stack

• Streamlit
• Python
• NumPy
• Pandas
• SciPy
• Matplotlib / Plotly

🔮 Future Enhancements

• Categorical drift detection
• Scheduled monitoring & alerts
• MLflow / Weights & Biases integration
• Optional API mode for automation
• Docker + Cloud deployment

📄 License

This project is released under the MIT License.











