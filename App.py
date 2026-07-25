"""
Aplikasi Web Interaktif — Analisis Regresi Linier Berganda
Dataset: Books Scraping (Harga vs Rating & Estimasi Stok)

Jalankan dengan:
    streamlit run app.py
"""

import pandas as pd
import numpy as np
import streamlit as st
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

sns.set(style="whitegrid")

st.set_page_config(page_title="Analisis Regresi - Books Dataset", layout="wide")

# ----------------------------------------------------------------------------
# SIDEBAR - Upload / Load Data
# ----------------------------------------------------------------------------
st.sidebar.title("⚙️ Pengaturan Data")
uploaded = st.sidebar.file_uploader("Upload dataset_original_books.csv", type=["csv"])

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

if uploaded is not None:
    df = load_data(uploaded)
else:
    try:
        df = load_data("dataset_original_books.csv")
        st.sidebar.info("Menggunakan file default: dataset_original_books.csv")
    except FileNotFoundError:
        st.sidebar.warning("Silakan upload file dataset_original_books.csv terlebih dahulu.")
        st.stop()

required_cols = {"Harga_Poundsterling_Y", "Rating_Bintang_X1", "Estimasi_Stok_X2"}
if not required_cols.issubset(df.columns):
    st.error(f"Dataset harus memiliki kolom: {required_cols}")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.metric("Jumlah Baris", len(df))
st.sidebar.metric("Jumlah Kolom", df.shape[1])

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.title("📚 Analisis Regresi Linier Berganda — Dataset Books Scraping")
st.caption("Y: Harga_Poundsterling_Y  |  X1: Rating_Bintang_X1  |  X2: Estimasi_Stok_X2")

tab_eda, tab_corr, tab_model, tab_assump, tab_viz, tab_predict, tab_conclusion = st.tabs(
    ["1️⃣ EDA", "2️⃣ Korelasi", "3️⃣ Model OLS", "4️⃣ Uji Asumsi",
     "5️⃣ Visualisasi", "6️⃣ Prediksi", "7️⃣ Kesimpulan"]
)

# ----------------------------------------------------------------------------
# 1. EDA
# ----------------------------------------------------------------------------
with tab_eda:
    st.subheader("Struktur & Statistik Deskriptif")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.write("**Data Preview**")
        st.dataframe(df.head(10), use_container_width=True)
    with c2:
        st.write("**Missing Value**")
        st.dataframe(df.isnull().sum().rename("jumlah"))
        st.write("**Data Duplikat**", int(df.duplicated().sum()))

    st.write("**Statistik Deskriptif**")
    st.dataframe(df.describe(), use_container_width=True)

# ----------------------------------------------------------------------------
# 2. KORELASI
# ----------------------------------------------------------------------------
with tab_corr:
    st.subheader("Analisis Korelasi Pearson")
    korelasi, p_value_korelasi = stats.pearsonr(df["Rating_Bintang_X1"], df["Harga_Poundsterling_Y"])
    c1, c2 = st.columns(2)
    c1.metric("Koefisien Korelasi (Rating vs Harga)", f"{korelasi:.4f}")
    c2.metric("P-value", f"{p_value_korelasi:.4f}")

    st.write("**Matriks Korelasi**")
    corr_matrix = df[["Harga_Poundsterling_Y", "Rating_Bintang_X1", "Estimasi_Stok_X2"]].corr()
    st.dataframe(corr_matrix.style.background_gradient(cmap="coolwarm"), use_container_width=True)

# ----------------------------------------------------------------------------
# 3. MODEL OLS
# ----------------------------------------------------------------------------
X = df[["Rating_Bintang_X1", "Estimasi_Stok_X2"]]
y = df["Harga_Poundsterling_Y"]
X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit()
residuals = model.resid

with tab_model:
    st.subheader("Regresi Linier Berganda (OLS)")
    st.text(model.summary().as_text())

    st.write("**Ringkasan Koefisien**")
    coef_df = pd.DataFrame({
        "Koefisien": model.params,
        "Std Error": model.bse,
        "t-value": model.tvalues,
        "p-value": model.pvalues,
    })
    st.dataframe(coef_df.round(4), use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("R-squared", f"{model.rsquared:.4f}")
    c2.metric("Prob (F-statistic)", f"{model.f_pvalue:.4f}")

# ----------------------------------------------------------------------------
# 4. UJI ASUMSI KLASIK
# ----------------------------------------------------------------------------
with tab_assump:
    st.subheader("4.1 Uji Normalitas Residual (Shapiro-Wilk)")
    stat_shapiro, p_val_shapiro = stats.shapiro(residuals)
    c1, c2 = st.columns(2)
    c1.metric("Statistik Shapiro-Wilk", f"{stat_shapiro:.4f}")
    c2.metric("P-value", f"{p_val_shapiro:.4f}")
    if p_val_shapiro > 0.05:
        st.success("Residual berdistribusi normal (asumsi normalitas terpenuhi).")
    else:
        st.warning("Residual TIDAK berdistribusi normal (asumsi normalitas belum terpenuhi).")

    st.subheader("4.2 Uji Multikolinearitas (VIF)")
    X_vif = df[["Rating_Bintang_X1", "Estimasi_Stok_X2"]]
    vif_data = pd.DataFrame()
    vif_data["Variabel"] = X_vif.columns
    vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    st.dataframe(vif_data, use_container_width=True)
    if (vif_data["VIF"] < 10).all():
        st.success("Tidak terjadi multikolinearitas antar variabel independen (VIF < 10).")
    else:
        st.warning("Terindikasi multikolinearitas pada salah satu variabel (VIF ≥ 10).")

# ----------------------------------------------------------------------------
# 5. VISUALISASI
# ----------------------------------------------------------------------------
with tab_viz:
    st.subheader("5.1 Scatter Plot: Rating vs Harga")
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.regplot(x="Rating_Bintang_X1", y="Harga_Poundsterling_Y", data=df,
                scatter_kws={"alpha": 0.5, "color": "royalblue"},
                line_kws={"color": "red"}, ax=ax1)
    ax1.set_title("Scatter Plot: Rating vs Harga")
    ax1.set_xlabel("Rating Bintang (X1)")
    ax1.set_ylabel("Harga (Y)")
    st.pyplot(fig1)

    st.subheader("5.2 Heatmap Matriks Korelasi")
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=0, vmax=1, ax=ax2)
    ax2.set_title("Heatmap Matriks Korelasi")
    st.pyplot(fig2)

    st.subheader("5.3 Residual Plot")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.scatter(model.fittedvalues, residuals, alpha=0.5, color="purple")
    ax3.axhline(y=0, color="red", linestyle="--")
    ax3.set_title("Residual Plot (Uji Asumsi)")
    ax3.set_xlabel("Nilai Prediksi")
    ax3.set_ylabel("Residual")
    st.pyplot(fig3)

    st.subheader("5.4 Distribusi Estimasi Stok")
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    sns.histplot(df["Estimasi_Stok_X2"], kde=True, color="teal", ax=ax4)
    st.pyplot(fig4)

# ----------------------------------------------------------------------------
# 6. PREDIKSI INTERAKTIF
# ----------------------------------------------------------------------------
with tab_predict:
    st.subheader("Coba Prediksi Harga Buku")
    st.write("Geser slider di bawah untuk melihat estimasi harga berdasarkan model OLS.")
    c1, c2 = st.columns(2)
    rating_input = c1.slider("Rating Bintang (X1)", 1, 5, 3)
    stok_input = c2.slider(
        "Estimasi Stok (X2)",
        int(df["Estimasi_Stok_X2"].min()),
        int(df["Estimasi_Stok_X2"].max()),
        int(df["Estimasi_Stok_X2"].median()),
    )
    pred_input = pd.DataFrame({"const": [1], "Rating_Bintang_X1": [rating_input], "Estimasi_Stok_X2": [stok_input]})
    pred_value = model.predict(pred_input)[0]
    st.metric("Estimasi Harga (£)", f"{pred_value:.2f}")
    st.caption("Catatan: hasil model memiliki R-squared rendah, sehingga prediksi ini bersifat indikatif, bukan akurat.")

# ----------------------------------------------------------------------------
# 7. KESIMPULAN
# ----------------------------------------------------------------------------
with tab_conclusion:
    st.subheader("Ringkasan Hasil")
    ringkasan = pd.DataFrame({
        "Pengujian": ["Korelasi Pearson", "R-Squared", "Prob(F-statistic)",
                      "Rating -> Harga (p-value)", "Estimasi Stok -> Harga (p-value)",
                      "Shapiro-Wilk (p-value)", "VIF Rating", "VIF Estimasi Stok"],
        "Hasil": [
            round(korelasi, 4),
            round(model.rsquared, 4),
            round(model.f_pvalue, 4),
            round(model.pvalues["Rating_Bintang_X1"], 4),
            round(model.pvalues["Estimasi_Stok_X2"], 4),
            round(p_val_shapiro, 4),
            round(vif_data.loc[vif_data["Variabel"] == "Rating_Bintang_X1", "VIF"].values[0], 3),
            round(vif_data.loc[vif_data["Variabel"] == "Estimasi_Stok_X2", "VIF"].values[0], 3),
        ]
    })
    st.dataframe(ringkasan, use_container_width=True)

    st.subheader("Kesimpulan Singkat")
    st.markdown("""
- Korelasi antara Rating Bintang dan Harga Buku sangat lemah.
- Model regresi berganda hanya menjelaskan sebagian kecil variasi harga buku (R-squared rendah),
  dan secara keseluruhan tidak signifikan (Prob F-statistic > 0.05).
- Variabel Estimasi Stok berpengaruh signifikan terhadap harga, sedangkan Rating Bintang tidak signifikan.
- Residual tidak berdistribusi normal (Shapiro-Wilk p-value < 0.05), sehingga asumsi normalitas
  belum terpenuhi — ini menjadi keterbatasan model.
- Tidak terjadi multikolinearitas antar variabel independen (VIF < 10).
""")