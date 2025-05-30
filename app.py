import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from lyapunov_utils import estimate_lyapunov, estimate_slope

st.set_page_config(page_title="CHAOS-Fusion", layout="centered")
st.title("🔬 CHAOS-Fusion: Lyapunov-Based Time Series Analyzer")

uploaded_file = st.file_uploader("Upload your time series CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    col = st.selectbox("Select a column to analyze", df.columns)
    ts = df[col].dropna().values

    st.subheader("⚙️ Configure Parameters")
    emb_dim = st.slider("Embedding Dimension (m)", 2, 10, 5)
    tau = st.slider("Time Delay (τ)", 1, 5, 1)
    max_t = st.slider("Max Time Step", 10, 100, 50)

    if st.button("🚀 Start Analysis"):
        if len(ts) < emb_dim * tau + 10:
            st.error(f"Time series is too short for embedding dimension = {emb_dim} and delay = {tau}.")
        else:
            with st.spinner("Computing Lyapunov exponent..."):
                try:
                    divergence = estimate_lyapunov(ts, emb_dim, tau, max_t)
                    slope = estimate_slope(divergence)
                    st.success(f"Estimated Largest Lyapunov Exponent: **{slope:.4f}**")

                    st.subheader("📈 Divergence Curve")
                    fig, ax = plt.subplots()
                    ax.plot(divergence, label="log(Divergence)")
                    ax.set_xlabel("Time Step")
                    ax.set_ylabel("log(Distance)")
                    ax.set_title("Lyapunov Divergence Curve")
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
else:
    st.info("Please upload a CSV file with at least one numeric column.")
