import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# ---- Lyapunov Functions ----
def estimate_lyapunov(ts, emb_dim=5, tau=1, max_t=50):
    n = len(ts) - (emb_dim - 1) * tau
    if n <= max_t:
        return np.zeros(max_t)
    emb_data = np.array([ts[i:i + n] for i in range(0, emb_dim * tau, tau)]).T
    dists = np.linalg.norm(emb_data[:, None, :] - emb_data[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    nearest_idx = np.argmin(dists, axis=1)
    divergence = []
    for t in range(1, max_t):
        dist_sum = 0
        count = 0
        for i in range(n - t):
            j = nearest_idx[i]
            if i + t < n and j + t < n:
                dist = np.linalg.norm(emb_data[i + t] - emb_data[j + t])
                if dist > 0:
                    dist_sum += np.log(dist)
                    count += 1
        divergence.append(dist_sum / count if count > 0 else 0)
    return np.array(divergence)

def estimate_slope(divergence, steps=20):
    if len(divergence) < steps:
        return 0.0
    X = np.arange(steps).reshape(-1, 1)
    y = divergence[:steps]
    model = LinearRegression().fit(X, y)
    return model.coef_[0]

# ---- Streamlit App ----
st.set_page_config(page_title="CHAOS-Fusion", layout="wide")
st.title("🔬 CHAOS-Fusion: General Lyapunov & Fusion Analyzer")

uploaded_file = st.file_uploader("📂 Upload a CSV file with at least two numeric time series", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("The uploaded file must contain at least two numeric columns.")
    else:
        col1 = st.selectbox("Select First Time Series Column", numeric_cols, index=0)
        col2 = st.selectbox("Select Second Time Series Column", numeric_cols, index=1)

        series1_full = pd.to_numeric(df[col1], errors='coerce').dropna().values
        series2_full = pd.to_numeric(df[col2], errors='coerce').dropna().values
        min_len = min(len(series1_full), len(series2_full))
        series1_full = series1_full[:min_len]
        series2_full = series2_full[:min_len]

        st.subheader("⚙️ Configure Analysis")
        weight1 = st.slider(f"{col1} Weight (%)", 0, 100, 70, step=10)
        weight2 = 100 - weight1
        embed_dim = st.slider("Embedding Dimension (m)", 2, 10, 5)
        tau = st.slider("Time Delay (τ)", 1, 5, 1)
        max_t = st.slider("Max Time Step", 10, 100, 50)
        lengths = list(range(500, min(5000, min_len), 500))

        if st.button("🚀 Run Analysis"):
            results = []
            for L in lengths:
                s1 = series1_full[:L]
                s2 = series2_full[:L]
                fused = (weight1 / 100) * s1 + (weight2 / 100) * s2

                div1 = estimate_lyapunov(s1, embed_dim, tau, max_t)
                div2 = estimate_lyapunov(s2, embed_dim, tau, max_t)
                div_fused = estimate_lyapunov(fused, embed_dim, tau, max_t)

                slope1 = estimate_slope(div1)
                slope2 = estimate_slope(div2)
                slope_fused = estimate_slope(div_fused)

                results.append({
                    "Length": L,
                    f"Lyapunov ({col1})": slope1,
                    f"Lyapunov ({col2})": slope2,
                    "Lyapunov (Fused)": slope_fused,
                    f"{col1} Weight": weight1 / 100,
                    f"{col2} Weight": weight2 / 100
                })

            res_df = pd.DataFrame(results)

            st.success("✅ Analysis Complete")
            st.subheader("📊 Lyapunov Exponents Comparison Table")
            st.dataframe(res_df)

            # GRAPH 1
            st.subheader("📈 Lyapunov vs. Length (Fused Signal)")
            fig1 = plt.figure(figsize=(10, 5))
            sns.lineplot(data=res_df, x="Length", y="Lyapunov (Fused)", marker="o", label="Fused")
            plt.grid(True)
            st.pyplot(fig1)

            # GRAPH 2
            st.subheader("📉 All Signals Comparison")
            fig2 = plt.figure(figsize=(10, 5))
            sns.lineplot(data=res_df, x="Length", y=f"Lyapunov ({col1})", label=col1, marker="o")
            sns.lineplot(data=res_df, x="Length", y=f"Lyapunov ({col2})", label=col2, marker="o")
            sns.lineplot(data=res_df, x="Length", y="Lyapunov (Fused)", label="Fused", linestyle="--", marker="o")
            plt.grid(True)
            plt.legend()
            st.pyplot(fig2)

            # INTERPRETATION SECTION
            st.subheader("🧠 Automated Interpretation")

            mean_1 = res_df[f"Lyapunov ({col1})"].mean()
            mean_2 = res_df[f"Lyapunov ({col2})"].mean()
            mean_fused = res_df["Lyapunov (Fused)"].mean()

            most_chaotic = max([(col1, mean_1), (col2, mean_2), ('Fused', mean_fused)], key=lambda x: x[1])[0]
            least_chaotic = min([(col1, mean_1), (col2, mean_2), ('Fused', mean_fused)], key=lambda x: x[1])[0]

            st.markdown(f"""
            - 📌 **Most chaotic signal** on average: **{most_chaotic}**
            - 📌 **Least chaotic signal**: **{least_chaotic}**
            - 📉 As length increases, fused signal tends to show {'increased' if res_df['Lyapunov (Fused)'].iloc[-1] > res_df['Lyapunov (Fused)'].iloc[0] else 'decreased or stable'} Lyapunov exponent.
            - ⚖️ Fusion with {weight1}% {col1} and {weight2}% {col2} appears to {'dampen' if mean_fused < max(mean_1, mean_2) else 'amplify'} chaos compared to individual signals.
            """)
