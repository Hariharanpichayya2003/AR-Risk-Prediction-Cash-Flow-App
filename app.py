import os
import streamlit as st
import pandas as pd
import joblib

# =========================
# 🔧 MEMORY FIX
# =========================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# =========================
# 📦 LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return joblib.load("ar_model.pkl")

model = load_model()

# 🔥 IMPORTANT: FIX FEATURE ORDER ISSUE
FEATURES = list(model.feature_names_in_)

# =========================
# 🎯 PAGE CONFIG
# =========================
st.set_page_config(page_title="AR Risk Dashboard", layout="wide")
st.title("📊 AR Risk Prediction Dashboard")

industry_map = {"SaaS": 0, "Retail": 1, "Manufacturing": 2, "Healthcare": 3}
region_map = {"India": 0, "US": 1, "UK": 2, "Germany": 3}

# =========================
# 🔹 SINGLE PREDICTION
# =========================
st.subheader("🔍 Single Customer Prediction")

col1, col2 = st.columns(2)

with col1:
    customer_id = st.number_input("Customer ID", value=1050)
    invoice_amount = st.number_input("Invoice Amount", value=50000)
    invoice_date = st.date_input("Invoice Date")
    due_date = st.date_input("Due Date")

with col2:
    industry = st.selectbox("Industry", list(industry_map.keys()))
    region = st.selectbox("Region", list(region_map.keys()))
    past_avg_delay = st.number_input("Past Avg Delay", value=3)
    total_outstanding = st.number_input("Total Outstanding", value=80000)
    num_invoices = st.number_input("Number of Invoices", value=12)

if st.button("Predict Single Customer"):

    df = pd.DataFrame([{
        "customer_id": customer_id,
        "invoice_amount": invoice_amount,
        "invoice_date": invoice_date,
        "due_date": due_date,
        "industry": industry_map[industry],
        "region": region_map[region],
        "past_avg_delay": past_avg_delay,
        "total_outstanding": total_outstanding,
        "num_invoices": num_invoices
    }])

    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["due_date"] = pd.to_datetime(df["due_date"])

    df["invoice_month"] = df["invoice_date"].dt.month
    df["due_month"] = df["due_date"].dt.month
    df["days_to_due"] = (df["due_date"] - df["invoice_date"]).dt.days

    df = df.drop(columns=["invoice_date", "due_date"])

    # 🔥 FIX FEATURE ORDER
    df = df.reindex(columns=FEATURES, fill_value=0)

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    if df["days_to_due"].iloc[0] == 0 and past_avg_delay == 0:
        risk = "🟢 Low Risk"
    elif probability > 0.5:
        risk = "🔴 High Risk"
    elif probability > 0.3:
        risk = "🟠 Medium Risk"
    else:
        risk = "🟢 Low Risk"

    st.write("Prediction:", prediction)
    st.write("Probability:", round(probability, 3))
    st.write("Risk:", risk)

# =========================
# 🔹 BULK PREDICTION
# =========================
st.subheader("📂 Bulk Prediction")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:

    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower()

    df = df.rename(columns={
        "invoice date": "invoice_date",
        "due date": "due_date"
    })

    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["due_date"] = pd.to_datetime(df["due_date"])

    df["invoice_month"] = df["invoice_date"].dt.month
    df["due_month"] = df["due_date"].dt.month
    df["days_to_due"] = (df["due_date"] - df["invoice_date"]).dt.days

    df["industry"] = df["industry"].map(industry_map)
    df["region"] = df["region"].map(region_map)

    df = df.drop(columns=["invoice_date", "due_date"])

    # 🔥 FIX FEATURE ORDER (IMPORTANT)
    features = df.reindex(columns=FEATURES, fill_value=0)

    df["prediction"] = model.predict(features)
    df["probability"] = model.predict_proba(features)[:, 1]

    df["risk_level"] = df["probability"].apply(
        lambda x: "High Risk" if x > 0.5 else "Medium Risk" if x > 0.16 else "Low Risk"
    )

    st.dataframe(df)

    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        "output.csv",
        "text/csv"
    )

# =========================
# ➕ ADD CUSTOMER MODULE
# =========================
st.subheader("➕ Add Customer")

if "records" not in st.session_state:
    st.session_state.records = []

c1, c2 = st.columns(2)

with c1:
    cid = st.number_input("Customer ID (Add)")
    amt = st.number_input("Invoice Amount (Add)")
    inv_date = st.date_input("Invoice Date (Add)")
    due = st.date_input("Due Date (Add)")

with c2:
    ind = st.selectbox("Industry (Add)", list(industry_map.keys()))
    reg = st.selectbox("Region (Add)", list(region_map.keys()))
    delay = st.number_input("Past Avg Delay")
    outstanding = st.number_input("Total Outstanding")
    num = st.number_input("Number of Invoices")

if st.button("Add Customer"):

    temp = pd.DataFrame([{
        "customer_id": cid,
        "invoice_amount": amt,
        "invoice_date": inv_date,
        "due_date": due,
        "industry": industry_map[ind],
        "region": region_map[reg],
        "past_avg_delay": delay,
        "total_outstanding": outstanding,
        "num_invoices": num
    }])

    temp["invoice_date"] = pd.to_datetime(temp["invoice_date"])
    temp["due_date"] = pd.to_datetime(temp["due_date"])

    temp["invoice_month"] = temp["invoice_date"].dt.month
    temp["due_month"] = temp["due_date"].dt.month
    temp["days_to_due"] = (temp["due_date"] - temp["invoice_date"]).dt.days

    temp = temp.drop(columns=["invoice_date", "due_date"])

    # 🔥 FIX FEATURE ORDER (MOST IMPORTANT PART)
    temp = temp.reindex(columns=FEATURES, fill_value=0)

    pred = model.predict(temp)[0]
    prob = model.predict_proba(temp)[0][1]

    risk = "High Risk" if prob > 0.5 else "Medium Risk" if prob > 0.17 else "Low Risk"

    temp["prediction"] = pred
    temp["probability"] = prob
    temp["risk_level"] = risk

    st.session_state.records.append(temp)

    st.success("Customer Added")

# =========================
# 📊 STORED DATA
# =========================
if len(st.session_state.records) > 0:

    final = pd.concat(st.session_state.records, ignore_index=True)
    st.dataframe(final)

    st.download_button(
        "Download Added Customers",
        final.to_csv(index=False).encode("utf-8"),
        "customers.csv",
        "text/csv"
    )