import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from io import BytesIO
from fpdf import FPDF

# --- PAGE CONFIG ---
st.set_page_config(page_title="Credit Scoring App", layout="wide")
st.title("📊 Credit Decisioning App")

# --- FUNCTIONS ---

def load_csv(file):
    try:
        return pd.read_csv(file)
    except Exception as e:
        st.error(f"❌ Error loading CSV: {e}")
        return None

def convert_drive_url_to_download_link(url):
    if "drive.google.com" in url:
        file_id = url.split("/d/")[1].split("/")[0]
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return url

def unzip_file(zip_file):
    with zipfile.ZipFile(zip_file, 'r') as zf:
        zf.extractall("content/")
        st.success("✅ Zip extracted!")
        return zf.namelist()[0]

def generate_pdf_report(user_id, score):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Credit Scoring Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"User ID: {user_id}", ln=True)
    pdf.cell(200, 10, txt=f"Credit Score: {score}", ln=True)
    return pdf.output(dest='S').encode('latin-1')

def visualize_data(df, name="Data"):
    st.subheader(f"📈 Distribution in {name}")
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols[:3]:  # Limit to 3 plots
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.success("✅ Model Trained!")
    st.json(report)
    return model

# --- USER FILE INPUT ---

st.sidebar.header("📁 Upload Files")

personal_file = st.sidebar.file_uploader("Upload Personal Finance CSV", type=["csv"])
energy_file = st.sidebar.file_uploader("Upload Energy Use CSV", type=["csv", "zip"])
paysim_link = st.sidebar.text_input("🔗 Enter PaySim Google Drive link")

# --- LOAD DATA ---

personal_df = load_csv(personal_file) if personal_file else None

energy_df = None
if energy_file:
    if energy_file.name.endswith(".zip"):
        energy = pd.read_csv("content/AEP_hourly.csv")
        energy_df = load_csv(f"content/{filename}")
    else:
        energy_df = load_csv(energy_file)

paysim_df = None
if paysim_link:
    try:
        download_url = convert_drive_url_to_download_link(paysim_link)
        paysim_df = pd.read_csv(download_url)
        if 'step' in paysim_df.columns:
            paysim_df['step'] = pd.to_timedelta(paysim_df['step'], unit='h')
            paysim_df['Date'] = pd.to_datetime("2020-01-01") + paysim_df['step']
        else:
            st.warning("⚠️ 'step' column missing in PaySim. Skipping time logic.")
    except Exception as e:
        st.error(f"❌ Error loading PaySim: {e}")

# --- PREVIEW DATA ---
if personal_df is not None:
    st.subheader("📂 Personal Finance Data")
    st.dataframe(personal_df.head())
    visualize_data(personal_df, "Personal Finance")

if energy_df is not None:
    st.subheader("⚡ Energy Consumption Data")
    st.dataframe(energy_df.head())
    visualize_data(energy_df, "Energy Use")

if paysim_df is not None:
    st.subheader("💸 PaySim Data")
    st.dataframe(paysim_df.head())
    visualize_data(paysim_df, "PaySim")

# --- CUSTOM CREDIT SCORING ---
st.sidebar.subheader("🎯 Custom Scoring Weights")
income_weight = st.sidebar.slider("Income weight", 0.0, 1.0, 0.3)
debt_weight = st.sidebar.slider("Debt weight", 0.0, 1.0, 0.3)
energy_weight = st.sidebar.slider("Energy usage weight", 0.0, 1.0, 0.4)

if st.button("💡 Calculate Score and Train Model") and personal_df is not None:
    try:
        personal_df = personal_df.dropna()
        personal_df['score'] = (
            income_weight * personal_df['Income'] -
            debt_weight * personal_df['Debt'] -
            energy_weight * personal_df['MonthlyExpenses']
        )

        st.success("✅ Custom scores calculated.")
        st.dataframe(personal_df[['ID', 'score']].head())

        # Train model
        X = personal_df[['Income', 'Debt', 'MonthlyExpenses']]
        y = (personal_df['score'] > personal_df['score'].median()).astype(int)
        model = train_model(X, y)

        # PDF Report
        selected_id = st.selectbox("🧍 Select User ID to Generate Report", personal_df['ID'])
        if st.button("🧾 Download PDF Report"):
            selected_score = personal_df.loc[personal_df['ID'] == selected_id, 'score'].values[0]
            pdf_bytes = generate_pdf_report(selected_id, selected_score)
            st.download_button("📄 Download Report", data=pdf_bytes, file_name="report.pdf")

    except Exception as e:
        st.error(f"❌ Error during scoring or training: {e}")

# --- M-PESA PLACEHOLDER ---
st.info("📡 M-Pesa API integration coming soon (requires Safaricom developer access).")

