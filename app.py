import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from lyapunov_utils import estimate_lyapunov, estimate_slope

st.set_page_config(page_title="CHAOS-Fusion", layout="centered")
st.title("🔬 CHAOS-Fusion: Lyapunov Tabanlı Veri Analizi")

uploaded_file = st.file_uploader("Zaman serisi verinizi yükleyin (.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    col = st.selectbox("Analiz edilecek sütunu seçin", df.columns)
    ts = df[col].dropna().values

    st.subheader("🔧 Parametre Seçimi")
    emb_dim = st.slider("Embed Dimension (m)", 2, 10, 5)
    tau = st.slider("Time Delay (τ)", 1, 5, 1)
    max_t = st.slider("Maksimum Zaman Adımı", 10, 100, 50)

    if st.button("🚀 Analizi Başlat"):
        with st.spinner("Lyapunov üstel hesaplanıyor..."):
            divergence = estimate_lyapunov(ts, emb_dim, tau, max_t)
            slope = estimate_slope(divergence)

        st.success(f"Lyapunov Üsteli: **{slope:.4f}**")

        st.subheader("📈 Diverjans Eğrisi")
        fig, ax = plt.subplots()
        ax.plot(divergence, label="log(Divergence)")
        ax.set_xlabel("Zaman Adımı")
        ax.set_ylabel("log(Dist)")
        ax.set_title("Lyapunov Diverjans Eğrisi")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
else:
    st.info("Lütfen bir .csv dosyası yükleyin. İlk sütun zaman serisi içermelidir.")
