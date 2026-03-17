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
st.set_page_config(page_title="Fraud Detection", layout="wide")

# Add Neat Background
st.markdown("""
<style>
/* App background setup */
.stApp {
    background-color: #f4f6fa; 
    background-image: radial-gradient(#c7d4e8 1.5px, transparent 1.5px);
    background-size: 24px 24px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOGO FUNCTION ----------------
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img:
            return base64.b64encode(img.read()).decode()
    return None

logo_base64 = get_base64_image("rce.png")

# ---------------- HEADER ----------------
col1, col2 = st.columns([3, 1])

with col1:
    if logo_base64:
        st.image(f"data:image/png;base64,{logo_base64}", width=150)

    st.markdown("""
    <h1 style='color:#2E86C1; margin-bottom:0;'>
        Ramachandra College of Engineering
    </h1>
    <h3 style='margin-top:5px;'>
        💳 Credit Card Fraud Detection System
    </h3>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='text-align:right; margin-top:35px; white-space: nowrap;'>
        <div style='display: inline-block; padding: 8px 16px; background: linear-gradient(135deg, #1A5276 0%, #2980B9 100%); color: white; border-radius: 25px; font-weight: 600; font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif; box-shadow: 0px 4px 6px rgba(0,0,0,0.1); font-size: 15px; letter-spacing: 0.5px;'>
            🧠 Artificial Intelligence & Data Science
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------- Load Dataset ----------------
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# ---------------- Upload ----------------
st.markdown("## 📂 Upload Dataset")
uploaded_file = st.file_uploader("Upload creditcard.csv", type=["csv"])

if uploaded_file is not None:
    data = load_data(uploaded_file)

    # ---------------- Dashboard Additions ----------------
    st.markdown("## 📈 Dashboard Overview")
    total_txn = len(data)
    fraud_txn = len(data[data["Class"] == 1])
    safe_txn = len(data[data["Class"] == 0])
    fraud_pct = (fraud_txn / total_txn) * 100

    col_dash1, col_dash2, col_dash3, col_dash4 = st.columns(4)
    col_dash1.metric("Total Transactions", f"{total_txn:,}")
    col_dash2.metric("Safe Transactions", f"{safe_txn:,}")
    col_dash3.metric("Fraud Transactions (1s)", f"{fraud_txn:,}")
    col_dash4.metric("Fraud Percentage", f"{fraud_pct:.3f}%")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ---------------- Preview ----------------
    st.markdown("## 📊 Dataset Preview")
    st.dataframe(data.head())

    st.write("### 🚨 Fraudulent Transactions Highlight")
    st.dataframe(data[data["Class"] == 1].head(10))

    st.write("### 📐 Dataset Shape")
    st.write(f"Rows: **{data.shape[0]}**, Columns: **{data.shape[1]}**")

    # ---------------- Visualizations ----------------
    st.markdown("## 📉 Visualizations")
    col_vis1, col_vis2 = st.columns(2)

    with col_vis1:
        st.write("### 📊 Class Distribution (0 = Safe, 1 = Fraud)")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x="Class", hue="Class", data=data, palette=["#28B463", "#E74C3C"], ax=ax, legend=False)
        ax.set_title("Transaction Class Distribution")
        st.pyplot(fig)

    # ---------------- Model Training ----------------
    st.markdown("## ⚙️ Model Training & Evaluation")
    st.info("Applying SMOTE to handle class imbalance before training models...", icon="ℹ️")

    X = data.drop(["Class", "Time"], axis=1)
    y = data["Class"]

    # Stratified Sampling to ensure we have fraud cases (avoid complete class 0 sample)
    sample_size = min(10000, len(data))
    fraud_indices = data[data["Class"] == 1].index
    safe_indices = data[data["Class"] == 0].index
    
    if len(fraud_indices) > 0:
        fraud_sample = data.loc[fraud_indices].sample(min(500, len(fraud_indices)), random_state=42)
        safe_sample = data.loc[safe_indices].sample(min(sample_size - len(fraud_sample), len(safe_indices)), random_state=42)
        sampled_data = pd.concat([fraud_sample, safe_sample]).sample(frac=1, random_state=42)
        X_sample = sampled_data.drop(["Class", "Time"], axis=1)
        y_sample = sampled_data["Class"]
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
    
    st.write("### 🏆 Model Comparison")
    # Highlight the maximum values in the dataframe safely
    st.dataframe(results_df.style.highlight_max(subset=["Accuracy", "F1 Score", "ROC-AUC"], color='lightgreen'))

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

    # ---------------- Explainable AI ----------------
    st.markdown("## 🧠 Explainable AI (SHAP)")
    if st.checkbox("🔍 Analyze predictions with SHAP (Takes a moment)"):
        with st.spinner("Calculating SHAP values..."):
            try:
                st.write(f"Generating SHAP explanation using **{best_model_name}** on a sample of test data.")
                X_test_sample = X_test.sample(min(100, len(X_test)), random_state=42)
                
                if best_model_name in ["Random Forest", "XGBoost"]:
                    explainer = shap.TreeExplainer(best_model)
                    shap_values = explainer.shap_values(X_test_sample)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    # Handle shapes varying by model
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

    # ---------------- Prediction ----------------
    st.markdown("## 🔍 Predict a Transaction")

    st.write("Modify the threshold and input values to simulate a transaction.")
    threshold = st.slider("Fraud Probability Threshold", min_value=0.0, max_value=1.0, value=0.50, step=0.01)

    input_data = {}
    col_input = st.columns(5)
    
    input_cols = X.columns[:10]
    for i, feature in enumerate(input_cols):
        with col_input[i % 5]:
            input_data[feature] = st.number_input(feature, value=0.0)

    input_df = pd.DataFrame([input_data])

    if st.button("📧 Send Test Email"):
        send_email_alert("TEST_TIME", 100.0, 0.95)
        st.success("Test email triggered")

    if st.button("Predict Fraud"):
        for col in X.columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[X.columns]
        
        prob = best_model.predict_proba(input_df)[0][1]
        prediction = 1 if prob >= threshold else 0

        st.markdown(f"### Probability of Fraud: **{prob*100:.2f}%**")

        if prediction == 1:
            alert_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            amount_val = input_df["Amount"].iloc[0] if "Amount" in input_df.columns else 0.0
            
            st.markdown(f"""
            <div style="background-color: #ffcccc; padding: 15px; border-radius: 8px; border-left: 8px solid red; margin-bottom: 10px;">
                <h3 style="color: red; margin: 0;">🚨 Fraud Detected – Bank Notified</h3>
                <p style="color: #660000; font-weight: bold; margin: 5px 0 0 0;">🕒 Alert Time: {alert_time} | Exceeded Threshold: {threshold*100:.0f}%</p>
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
            st.success(f"✅ **Safe Transaction!** (Below {threshold*100:.0f}% threshold)")

        # Download result logic
        pred_result = input_df.copy()
        pred_result["Fraud_Probability"] = f"{prob*100:.2f}%"
        pred_result["Prediction"] = "Fraud Detected" if prediction == 1 else "Safe Transaction"
        
        csv = pred_result.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Prediction Result as CSV",
            data=csv,
            file_name="prediction_result.csv",
            mime="text/csv",
        )

        # Fraud Logs System UI
        st.markdown("---")
        st.markdown("### 📋 Fraud Logs System")
        if os.path.exists("fraud_log.csv"):
            fraud_logs_df = pd.read_csv("fraud_log.csv")
            st.dataframe(fraud_logs_df.tail(5)) # show recent 5 logs
            
            csv_log = fraud_logs_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download fraud_log.csv",
                data=csv_log,
                file_name="fraud_log.csv",
                mime="text/csv",
            )
        else:
            st.info("No fraud logs generated yet.")

else:
    st.info("👆 Please upload the creditcard.csv file to proceed.")
    if not os.path.exists("rce.png"):
        pass

# ---------------- FOOTER ----------------
st.markdown("""
<style>
.footer-box {
    background: linear-gradient(135deg, #f5f7fa, #e4ecf7);
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    text-align: center;
    margin-top: 40px;
}

.footer-title {
    font-size: 24px;
    font-weight: bold;
    color: #2E86C1;
    margin-bottom: 12px;
}

.footer-ids {
    font-size: 20px;
    color: #333;
    letter-spacing: 1px;
}
</style>

<div class="footer-box">
    <div class="footer-title">Team IDs</div>
    <div class="footer-ids">
        24ME1A54B4 | 24ME1A54C7 | 24ME1A5467 | 24ME1A54B3
    </div>
</div>
""", unsafe_allow_html=True)
