import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AML Transaction Monitoring",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Load artifacts saved from the notebook
#   model.save('AML.keras')
#   joblib.dump(preprocess, 'preprocessor')
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = load_model("AML.keras")
    preprocessor = joblib.load("preprocessor")
    return model, preprocessor


try:
    model, preprocessor = load_artifacts()
except Exception as e:
    st.error(
        "Could not load `AML.keras` and `preprocessor`. "
        "Place them in the same folder as this app.\n\n"
        f"Details: {e}"
    )
    st.stop()

# ----------------------------------------------------------------------------
# Category options (taken from the dataset in the notebook)
# ----------------------------------------------------------------------------
CURRENCIES = [
    "UK pounds", "Indian rupee", "Albanian lek", "Swiss franc", "Pakistani rupee",
    "Naira", "Yen", "Euro", "Dirham", "Mexican Peso", "Turkish lira",
    "US dollar", "Moroccan dirham",
]
LOCATIONS = [
    "UK", "Albania", "Nigeria", "Japan", "Spain", "Switzerland", "UAE", "Italy",
    "France", "Pakistan", "Mexico", "Austria", "Germany", "Turkey", "India",
    "Morocco", "USA", "Netherlands",
]
PAYMENT_TYPES = [
    "Cash Deposit", "Cross-border", "Cheque", "ACH", "Credit card",
    "Debit card", "Cash Withdrawal",
]

FEATURE_COLUMNS = [
    "Amount", "Payment_currency", "Received_currency",
    "Sender_bank_location", "Receiver_bank_location", "Payment_type",
]


def predict(df: pd.DataFrame) -> np.ndarray:
    """Preprocess with the saved ColumnTransformer and score with the ANN."""
    X = preprocessor.transform(df[FEATURE_COLUMNS])
    if hasattr(X, "toarray"):  # sparse -> dense
        X = X.toarray()
    return model.predict(X, verbose=0).ravel()


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("Settings")
threshold = st.sidebar.slider(
    "Suspicious threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.05,
    help="Transactions with a probability at or above this value are flagged.",
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "ANN model trained on the SAML-D synthetic AML dataset. "
    "Intended as a decision-support tool, not a final verdict."
)

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("Intelligent Anti-Money Laundering Detection")
st.write("Classify transactions as **legitimate** or **suspicious** using an Artificial Neural Network.")

# ----------------------------------------------------------------------------
# Single transaction
# ----------------------------------------------------------------------------
with st.form("single_form"):
    c1, c2 = st.columns(2)
    with c1:
        amount = st.number_input(
            "Transaction Amount", min_value=0.0, step=100.0, format="%.2f"
        )
        payment_type = st.selectbox("Payment Type", PAYMENT_TYPES)
        payment_currency = st.selectbox("Payment Currency", CURRENCIES, index=0)
        received_currency = st.selectbox("Received Currency", CURRENCIES, index=0)
    with c2:
        sender_loc = st.selectbox("Sender Bank Location", LOCATIONS, index=0)
        receiver_loc = st.selectbox("Receiver Bank Location", LOCATIONS, index=0)

    submitted = st.form_submit_button("🔍 Analyze Transaction", use_container_width=True)

if submitted:
    row = pd.DataFrame(
        [{
            "Amount": amount,
            "Payment_currency": payment_currency,
            "Received_currency": received_currency,
            "Sender_bank_location": sender_loc,
            "Receiver_bank_location": receiver_loc,
            "Payment_type": payment_type,
        }]
    )
    prob = float(predict(row)[0])

    st.markdown("### Result")
    m1, m2 = st.columns(2)
    m1.metric("Suspicion probability", f"{prob:.2%}")
    m2.metric("Threshold", f"{threshold:.0%}")
    st.progress(min(max(prob, 0.0), 1.0))

    if prob >= threshold:
        st.error("🚨 **SUSPICIOUS** transaction: recommend manual review / escalation.")
    else:
        st.success("✅ **LEGITIMATE** transaction: no action required.")

    with st.expander("Input summary"):
        st.dataframe(row, use_container_width=True)