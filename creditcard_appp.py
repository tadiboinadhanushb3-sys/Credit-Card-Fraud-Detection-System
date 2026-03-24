import streamlit as st
import pandas as pd
import numpy as np
import base64
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from imblearn.over_sampling import SMOTE
import shap
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from twilio.rest import Client
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# ---------------- NOTIFICATION SYSTEM ----------------
def send_email_alert(alert_time, amount, probability):
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")
    receiver_email = os.getenv("RECEIVER_EMAIL")
    
    if not sender_email or not sender_password or not receiver_email:
        return
        
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = "🚨 URGENT: Fraudulent Transaction Detected"
        
        body = f"""Alert: Fraudulent transaction detected!

Transaction Details:
- Time: {alert_time}
- Amount: ${amount:.2f}
- Fraud Probability: {probability*100:.2f}%

Please verify immediately."""
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
    except Exception as e:
        print(f"Failed to send email: {e}")

def send_sms_alert(alert_time, amount):
    twilio_sid = os.getenv("TWILIO_ACCOUNT_SID")
    twilio_token = os.getenv("TWILIO_AUTH_TOKEN")
    twilio_from = os.getenv("TWILIO_FROM_PHONE")
    twilio_to = os.getenv("TWILIO_TO_PHONE")
    
    if not twilio_sid or not twilio_token or not twilio_from or not twilio_to:
        return
        
    try:
        client = Client(twilio_sid, twilio_token)
        client.messages.create(
            body=f"🚨 Alert: Fraud transaction detected (${amount:.2f} at {alert_time}). Please verify immediately.",
            from_=twilio_from,
            to=twilio_to
        )
    except Exception as e:
        print(f"Failed to send SMS: {e}")

# ---------------- Page Config ----------------
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# ---------------- LOGO FUNCTION ----------------
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img:
            return base64.b64encode(img.read()).decode()
    return None

logo_base64 = get_base64_image("rce.png")
logo_html = f'<img src="data:image/png;base64,{logo_base64}" width="700" style="margin-bottom: 60px;">' if logo_base64 else ''

# Add Custom FinTech Animated Background and Professional Banner
st.markdown(f"""
<style>

/* ---------------- Background ---------------- */
.stApp {{
    background: radial-gradient(circle at 50% 50%, #0a1128 0%, #000000 100%);
}}

.stApp::before {{
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background-image:
        linear-gradient(rgba(0,210,255,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,210,255,0.04) 1px, transparent 1px);
    background-size: 40px 40px;
    z-index: -2;
    pointer-events: none;
}}

/* ---------------- Animation ---------------- */
@keyframes flow {{
    0% {{ transform: translateY(-100%); opacity: 0; }}
    50% {{ opacity: 0.8; }}
    100% {{ transform: translateY(100vh); opacity: 0; }}
}}

.data-flow {{
    position: fixed;
    top: 0;
    width: 2px;
    height: 120px;
    background: linear-gradient(to bottom, transparent, #00d2ff, transparent);
    opacity: 0;
    z-index: -1;
    animation: flow 6s infinite linear;
    pointer-events: none;
}}

.data-flow:nth-child(1) {{ left: 15%; animation-delay: 0s; }}
.data-flow:nth-child(2) {{ left: 35%; animation-delay: 2.5s; }}
.data-flow:nth-child(3) {{ left: 55%; animation-delay: 5s; }}
.data-flow:nth-child(4) {{ left: 75%; animation-delay: 1.5s; }}
.data-flow:nth-child(5) {{ left: 95%; animation-delay: 4s; }}

/* ---------------- UI Enhancements ---------------- */
.banner-container {{
    background: linear-gradient(135deg, rgba(10,25,47,0.95), rgba(2,12,27,0.98));
    border-left: 6px solid #00d2ff;
    border-right: 6px solid #00d2ff;
    border-radius: 16px;
    padding: 45px 20px;
    margin-bottom: 40px;
    text-align: center;
    box-shadow: 0 0 30px rgba(0,210,255,0.15);
}}

.college-name {{ font-size: 80px; font-weight: 900; letter-spacing: 2px; color: #e6f1ff; }}
.autonomous-text {{ font-size: 32px; color: #94a3b8; margin-bottom: 20px; }}
.project-title {{ font-size: 90px; font-weight: 900; color: #ffd700; margin: 10px 0; line-height: 1.2; text-shadow: 0px 4px 15px rgba(255,215,0,0.3); }}
.subtitle {{ font-size: 46px; color: #64ffda; margin-bottom: 25px; }}
.team-section {{ display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; margin-top: 15px; }}
.team-member {{ background: rgba(100,255,218,0.08); padding: 12px 28px; border-radius: 40px; font-size: 28px; color: #ccd6f6; border: 1px solid rgba(100,255,218,0.15); }}

.section-card {{
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(100, 255, 218, 0.2);
    border-radius: 12px;
    padding: 25px 30px;
    margin: 20px 0;
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
}}

.section-title {{
    font-size: 50px;
    font-weight: 800;
    color: #00d2ff;
    margin-bottom: 20px;
    border-bottom: 2px solid rgba(0, 210, 255, 0.3);
    padding-bottom: 10px;
}}

/* Ensure all default Streamlit text handles dark mode */
.stApp p, .stApp label, .stMarkdown {{
    color: #e2e8f0 !important;
    font-size: 24px !important;
}}
.stApp h1 {{ font-size: 60px !important; color: #e2e8f0 !important; }}
.stApp h2 {{ font-size: 50px !important; color: #e2e8f0 !important; }}
.stApp h3 {{ font-size: 40px !important; color: #e2e8f0 !important; }}
div[data-testid="stDataFrame"] * {{
    font-size: 20px !important;
}}

/* Enlarging Metrics */
[data-testid="stMetricValue"] {{
    font-size: 64px !important;
    font-weight: bold !important;
    color: #ffd700 !important;
}}
[data-testid="stMetricLabel"] {{
    font-size: 30px !important;
    color: #64ffda !important;
}}

/* Prediction styling */
.huge-prob {{
    font-size: 70px;
    font-weight: bold;
    color: #E74C3C;
    text-align: center;
    margin: 20px 0;
    text-shadow: 0 0 10px rgba(231, 76, 60, 0.4);
}}
.safe-prob {{
    font-size: 50px;
    font-weight: bold;
    color: #28B463;
    text-align: center;
    margin: 20px 0;
    text-shadow: 0 0 10px rgba(40, 180, 99, 0.4);
}}

/* Enlarging upload text globally */
.stFileUploader label {{
    font-size: 20px !important;
}}

</style>

<div class="data-flow"></div>
<div class="data-flow"></div>
<div class="data-flow"></div>
<div class="data-flow"></div>
<div class="data-flow"></div>

<div class="banner-container">
{logo_html}
<div class="college-name">RAMACHANDRA COLLEGE OF ENGINEERING</div>
<div class="autonomous-text">AUTONOMOUS</div>
<div class="project-title">CREDIT CARD FRAUD DETECTION SYSTEM</div>
<div class="subtitle">ARTIFICIAL INTELLIGENCE AND DATA SCIENCE</div>
<div class="team-section">
<div class="team-member">24ME1A5467 – L. Chandana Sasi</div>
<div class="team-member">24ME1A54B4 – T. Akshaya</div>
<div class="team-member">24ME1A54C7 – V. Madhuri</div>
</div>
</div>
""", unsafe_allow_html=True)

# ---------------- Load Dataset ----------------
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# ---------------- Upload Section ----------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📂 Upload Dataset</div>', unsafe_allow_html=True)
st.markdown("<p style='font-size: 20px; color: #a0aec0;'>Please upload your transaction CSV file below. The system automatically detects the features and the target variable.</p>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["csv"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    data = load_data(uploaded_file)
    
    # Auto-detect target column
    possible_targets = ["Class", "Fraud", "Target", "Label", "is_fraud"]
    target_col = None
    for pt in possible_targets:
        if pt in data.columns:
            target_col = pt
            break
            
    if target_col is None:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.warning("⚠️ Target column not auto-detected.")
        target_col = st.selectbox("Please select the target column manually:", data.columns)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- Dashboard Additions ----------------
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 System Dashboard</div>', unsafe_allow_html=True)
    
    total_txn = len(data)
    fraud_txn = len(data[data[target_col] == 1])
    safe_txn = len(data[data[target_col] == 0])
    fraud_pct = (fraud_txn / total_txn) * 100

    col_dash1, col_dash2, col_dash3, col_dash4 = st.columns(4)
    col_dash1.metric("Total Transactions", f"{total_txn:,}")
    col_dash2.metric("Safe Transactions", f"{safe_txn:,}")
    col_dash3.metric("Fraud Transactions", f"{fraud_txn:,}")
    col_dash4.metric("Fraud Percentage", f"{fraud_pct:.3f}%")
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- Preview ----------------
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Dataset Overview</div>', unsafe_allow_html=True)
    st.write(f"<p style='font-size: 22px;'>Dataset Shape: <b>{data.shape[0]:,}</b> Rows & <b>{data.shape[1]}</b> Columns</p>", unsafe_allow_html=True)
    
    # Styled dataframe
    def highlight_fraud(row):
        return ['background-color: rgba(231, 76, 60, 0.3)' if row[target_col] == 1 else '' for _ in row]

    st.write("<p style='font-size: 20px;'>Sample Data (Fraudulent rows highlighted in red):</p>", unsafe_allow_html=True)
    st.dataframe(data.head(50).style.apply(highlight_fraud, axis=1), use_container_width=True, height=400)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- Visualizations ----------------
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📉 Visual Data Analysis</div>', unsafe_allow_html=True)
    
    col_vis1, col_vis2 = st.columns(2)
    with col_vis1:
        st.write(f"### 📊 Distribution of {target_col}")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x=target_col, data=data, palette=["#28B463", "#E74C3C"], ax=ax)
        ax.set_title(f"{target_col} Distribution", fontsize=14, fontweight='bold')
        st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- Model Training ----------------
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚙️ Model Training & Evaluation</div>', unsafe_allow_html=True)
    st.info("Applying SMOTE to handle class imbalance and training comparative models...", icon="ℹ️")

    # Drop Time if it exists
    drop_cols = [target_col]
    if "Time" in data.columns:
        drop_cols.append("Time")
        
    X = data.drop(drop_cols, axis=1)
    y = data[target_col]

    # Stratified Sampling to ensure we have fraud cases (avoid complete class 0 sample)
    sample_size = min(10000, len(data))
    fraud_indices = data[data[target_col] == 1].index
    safe_indices = data[data[target_col] == 0].index
    
    if len(fraud_indices) > 0:
        fraud_sample = data.loc[fraud_indices].sample(min(500, len(fraud_indices)), random_state=42)
        safe_sample = data.loc[safe_indices].sample(min(sample_size - len(fraud_sample), len(safe_indices)), random_state=42)
        sampled_data = pd.concat([fraud_sample, safe_sample]).sample(frac=1, random_state=42)
        X_sample = sampled_data.drop(drop_cols, axis=1)
        y_sample = sampled_data[target_col]
    else:
        # Fallback if no frauds in uploaded dataset
        X_sample = X.sample(sample_size, random_state=42)
        y_sample = y.loc[X_sample.index]

    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=0.2, random_state=42, stratify=y_sample if len(y_sample.unique()) > 1 else None
    )

    # Apply SMOTE only if there are multiple classes and minority has enough samples
    try:
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    except (ValueError, RuntimeError):
        st.warning("Not enough instances in minority class for SMOTE. Training without SMOTE.")
        X_train_res, y_train_res = X_train, y_train

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42, class_weight="balanced"),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        if len(y_train_res.unique()) > 1:
            model.fit(X_train_res, y_train_res)
        else:
            # Fallback for single class
            model.fit(X_train, y_train)
            
        trained_models[name] = model
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        # Calculate metrics safely
        if len(y_test.unique()) > 1:
            auc = roc_auc_score(y_test, y_prob)
        else:
            auc = 0.0
            
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": auc
        })

    results_df = pd.DataFrame(results)
    
    st.write("### 🏆 Model Comparison Results")
    st.dataframe(results_df.style.highlight_max(subset=["Accuracy", "F1 Score", "ROC-AUC"], color='rgba(40,180,99,0.5)'), use_container_width=True)

    # Best model selection by highest F1 Score (since detecting fraud is the goal)
    best_model_row = results_df.sort_values(by="F1 Score", ascending=False).iloc[0]
    best_model_name = best_model_row["Model"]
    
    st.success(f"🌟 **Best Model Selected Automatically:** {best_model_name} (F1 Score: {best_model_row['F1 Score']:.4f})")
    
    best_model = trained_models[best_model_name]

    # Additional visualizations for best model
    st.write(f"### 🟩 Confusion Matrix & 📈 ROC Curve ({best_model_name})")
    col_vis3, col_vis4 = st.columns(2)
    
    with col_vis3:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(confusion_matrix(y_test, best_model.predict(X_test)), annot=True, fmt='d', cmap="Blues", ax=ax, 
                    xticklabels=['Safe', 'Fraud'], yticklabels=['Safe', 'Fraud'])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with col_vis4:
        fig, ax = plt.subplots(figsize=(5, 4))
        for name, model in trained_models.items():
            if len(y_test.unique()) > 1:
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                ax.plot(fpr, tpr, label=f"{name}")
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)

    # Feature Importance for tree models
    if best_model_name in ["Random Forest", "XGBoost"]:
        st.write(f"### 🌲 Feature Importance ({best_model_name})")
        feat_importances = pd.Series(best_model.feature_importances_, index=X.columns)
        fig, ax = plt.subplots(figsize=(10, 5))
        feat_importances.nlargest(10).plot(kind='barh', ax=ax, color='lightcoral')
        ax.set_title("Top 10 Important Features")
        ax.set_xlabel("Relative Importance")
        st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- Explainable AI ----------------
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧠 Explainable AI (SHAP)</div>', unsafe_allow_html=True)
    if st.checkbox("🔍 Analyze predictions with SHAP (Takes a moment)"):
        with st.spinner("Calculating SHAP values..."):
            try:
                st.write(f"Generating SHAP explanation using **{best_model_name}** on a sample of test data.")
                X_test_sample = X_test.sample(min(100, len(X_test)), random_state=42)
                
                if best_model_name in ["Random Forest", "XGBoost"]:
                    explainer = shap.TreeExplainer(best_model)
                    shap_values = explainer.shap_values(X_test_sample)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    if isinstance(shap_values, list):
                        shap.summary_plot(shap_values[1], X_test_sample, show=False)
                    else:
                        shap.summary_plot(shap_values, X_test_sample, show=False)
                    st.pyplot(fig)
                else:
                    # Logistic Regression
                    explainer = shap.LinearExplainer(best_model, X_train_res)
                    shap_values = explainer.shap_values(X_test_sample)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    if isinstance(shap_values, list):
                        shap.summary_plot(shap_values[1], X_test_sample, show=False)
                    else:
                        shap.summary_plot(shap_values, X_test_sample, show=False)
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"⚠️ SHAP visualization failed for {best_model_name}: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- Prediction ----------------
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔍 Real-time Active Prediction System</div>', unsafe_allow_html=True)

    st.write("<p style='font-size: 20px;'>Dynamically inject values corresponding to the detected dataset schema to test the model's prediction.</p>", unsafe_allow_html=True)
    threshold = st.slider("Fraud Probability Threshold (Sensitivity)", min_value=0.0, max_value=1.0, value=0.50, step=0.01)

    st.markdown("### 🧩 Input Transaction Data")
    input_data = {}
    
    # Dynamically generate input fields in 4 columns
    input_cols = X.columns
    cols_per_row = 4
    
    for i in range(0, len(input_cols), cols_per_row):
        row_cols = st.columns(cols_per_row)
        for j, feature in enumerate(input_cols[i:i+cols_per_row]):
            with row_cols[j]:
                input_data[feature] = st.number_input(feature, value=0.0, format="%.4f")

    input_df = pd.DataFrame([input_data])
    
    st.write("") # spacer

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        if st.button("📧 Send Test Alert"):
            send_email_alert("TEST_TIME", 100.0, 0.95)
            st.success("Test email triggered")

    with col_btn2:
        if st.button("🚨 Evaluate Transaction", type="primary", use_container_width=True):
            # Predict
            prob = best_model.predict_proba(input_df)[0][1]
            prediction = 1 if prob >= threshold else 0

            st.markdown("---")
            if prediction == 1:
                st.markdown(f'<div class="huge-prob">Probability of Fraud: {prob*100:.2f}%</div>', unsafe_allow_html=True)
                
                alert_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                amount_val = input_df["Amount"].iloc[0] if "Amount" in input_df.columns else 0.0
                
                st.markdown(f"""
                <div style="background: rgba(231, 76, 60, 0.15); padding: 20px; border-radius: 12px; border: 2px solid #E74C3C; text-align: center;">
                    <h2 style="color: #E74C3C; margin: 0; font-size: 36px;">🚨 FRUAD DETECTED</h2>
                    <p style="color: #ffcccc; font-size: 20px; margin-top: 10px;">Bank and Admin Have Been Notified.</p>
                    <p style="color: white; font-size: 18px;">🕒 Time: {alert_time} | Threshold Exceeded: {threshold*100:.0f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Log to CSV
                log_file = "fraud_log.csv"
                new_log = pd.DataFrame([{
                    "Time": alert_time,
                    "Amount": amount_val,
                    "Prediction": "Fraud",
                    "Probability": f"{prob*100:.2f}%"
                }])
                
                if os.path.exists(log_file):
                    new_log.to_csv(log_file, mode='a', header=False, index=False)
                else:
                    new_log.to_csv(log_file, mode='w', header=True, index=False)
                
                # Send Notification
                send_email_alert(alert_time, amount_val, prob)
                send_sms_alert(alert_time, amount_val)

            else:
                st.markdown(f'<div class="safe-prob">Probability of Fraud: {prob*100:.2f}%</div>', unsafe_allow_html=True)
                st.markdown("""
                <div style="background: rgba(40, 180, 99, 0.15); padding: 20px; border-radius: 12px; border: 2px solid #28B463; text-align: center;">
                    <h2 style="color: #28B463; margin: 0; font-size: 36px;">✅ SAFE TRANSACTION</h2>
                    <p style="color: #ccffcc; font-size: 20px; margin-top: 10px;">Transaction looks normal and is verified.</p>
                </div>
                """, unsafe_allow_html=True)

            # Download result logic
            pred_result = input_df.copy()
            pred_result["Fraud_Probability"] = f"{prob*100:.2f}%"
            pred_result["Prediction"] = "Fraud Detected" if prediction == 1 else "Safe Transaction"
            
            csv = pred_result.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Prediction Receipt",
                data=csv,
                file_name="prediction_receipt.csv",
                mime="text/csv",
            )
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- Fraud Logs System ----------------
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 Secure Fraud Logs System</div>', unsafe_allow_html=True)
    if os.path.exists("fraud_log.csv"):
        fraud_logs_df = pd.read_csv("fraud_log.csv")
        st.write("<p style='font-size: 18px;'>Latest Registered Fraud Incidents:</p>", unsafe_allow_html=True)
        st.dataframe(fraud_logs_df.tail(10), use_container_width=True)
        
        csv_log = fraud_logs_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Fraud Incident Logs",
            data=csv_log,
            file_name="fraud_log.csv",
            mime="text/csv",
        )
    else:
        st.info("No recorded fraud incidents yet.")
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("👆 Please securely upload your transaction dataset (CSV format) to activate the intelligent monitoring system.")
    if not os.path.exists("rce.png"):
        pass
