# CHAOS-Fusion 🔬

**CHAOS-Fusion** is a web-based interactive application built with [Streamlit](https://streamlit.io), designed for analyzing time series data using **Lyapunov exponents** to detect chaotic behavior.

Users can upload their own `.csv` files and instantly evaluate the dynamic stability of the data by configuring embedding and delay parameters.

---

## 🚀 Features

- Upload your own `.csv` time series data
- Select column for analysis
- Configure:
  - Embedding dimension
  - Time delay
  - Maximum time step
- Compute largest Lyapunov exponent using the Rosenstein method
- Plot divergence curves
- Intuitive and responsive web interface

---

## 📊 Sample Input Format

```csv
time_series
0.1
0.15
0.2
0.18
0.22
0.3
