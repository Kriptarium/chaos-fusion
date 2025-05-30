import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from lyapunov_utils import estimate_lyapunov, estimate_slope

st.set_page_config(page_title="CHAOS-Fusion", layout="centered")
st.title("🔬 CHAOS-Fusion: Data Fusion & Lyapunov Analysis")

uploaded_file = st.file_uploader("Upload a CSV file with at least two numeric time series", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    if len(numeric_cols) < 2:
        st.error("CSV file must contain at least two numeric columns.")
    else:
        col1 = st.selectbox("Select first column (e.g., IMU)", numeric_cols, index=0)
        col2 = st.selectbox("Select second column (e.g., GPS)", numeric_cols, index=1)

        s1 = pd.to_numeric(df[col1].dropna(), errors='coerce').dropna().values
        s2 = pd.to_numeric(df[col2].dropna(), errors='coerce').dropna().values
        min_len = min(len(s1), len(s2))
        s1, s2 = s1[:min_len], s2[:min_len]
        fused = 0.5 * s1 + 0.5 * s2

        st.subheader("⚙️ Configure Analysis Parameters")
        emb_dim = st.slider("Embedding Dimension (m)", 2, 10, 5)
        tau = st.slider("Time Delay (τ)", 1, 5, 1)
        max_t = st.slider("Max Time Step", 10, 100, 50)

        if len(s1) < emb_dim * tau + 10:
            st.error("Time series too short for given parameters.")
        elif st.button("🚀 Run Analysis"):
            with st.spinner("Analyzing..."):
                try:
                    div_s1 = estimate_lyapunov(s1, emb_dim, tau, max_t)
                    div_s2 = estimate_lyapunov(s2, emb_dim, tau, max_t)
                    div_fused = estimate_lyapunov(fused, emb_dim, tau, max_t)

                    slope_s1 = estimate_slope(div_s1)
                    slope_s2 = estimate_slope(div_s2)
                    slope_fused = estimate_slope(div_fused)

                    st.success(f"Lyapunov Exponents:")
                    st.write(f"• {col1}: **{slope_s1:.4f}**")
                    st.write(f"• {col2}: **{slope_s2:.4f}**")
                    st.write(f"• Fused: **{slope_fused:.4f}**")

                    st.subheader("📈 Divergence Comparison")
                    fig, ax = plt.subplots()
                    ax.plot(div_s1, label=f"{col1}")
                    ax.plot(div_s2, label=f"{col2}")
                    ax.plot(div_fused, label="Fused", linewidth=2, linestyle='--')
                    ax.set_xlabel("Time Step")
                    ax.set_ylabel("log(Distance)")
                    ax.set_title("Lyapunov Divergence Curves")
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
else:
    st.info("Please upload a CSV file with at least two numeric columns.")
