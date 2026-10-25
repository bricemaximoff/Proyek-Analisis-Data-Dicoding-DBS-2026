import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="E-Commerce Dashboard", page_icon="📊", layout="wide")
sns.set(style="dark")

COLOR_MAIN = "#72BCD4"
COLOR_GREY = "#D3D3D3"
COLOR_RED = "#E74C3C"

# =====================================================================
# 1. HELPER UMUM
# =====================================================================

def has_cols(df, cols):
    return all(c in df.columns for c in cols)

def in_period(df, start, end):
    """Filter berdasarkan tanggal pembelian (end inklusif)."""
    ts = df["order_purchase_timestamp"]
    return df[(ts >= pd.Timestamp(start)) & (ts < pd.Timestamp(end) + pd.Timedelta(days=1))]

def dedup_orders(df):
    """Dataset berbentuk 1 baris per item; untuk statistik per pesanan ambil 1 baris per order_id."""
    return df.drop_duplicates(subset="order_id")

def pct(x):
    return f"{x:.2f}%"

# =====================================================================
# 2. LOAD DATA & CLEANING
# =====================================================================

@st.cache_data
def load_data():
    for path in ["main_data.csv", "dashboard/main_data.csv"]:
        try:
            return pd.read_csv(path, low_memory=False)
        except FileNotFoundError:
            continue
    return pd.DataFrame()

all_df = load_data()
if all_df.empty:
    st.error("File main_data.csv tidak ditemukan!")
    st.stop()

required_cols = ["order_id", "order_purchase_timestamp", "customer_city", "customer_state",
                 "payment_type", "order_status", "review_score", "price"]
missing_cols = [c for c in required_cols if c not in all_df.columns]
if missing_cols:
    st.error(f"Kolom wajib tidak ada di main_data.csv: {', '.join(missing_cols)}. "
             "Jalankan build_main_data.py untuk membangun ulang file data.")
    st.stop()

for col in ["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]:
    if col in all_df.columns:
        all_df[col] = pd.to_datetime(all_df[col], errors="coerce")

for col in ["price", "freight_value", "review_score"]:
    if col in all_df.columns:
        all_df[col] = pd.to_numeric(all_df[col], errors="coerce")

all_df["customer_city"] = all_df["customer_city"].astype(str).str.lower()
all_df["customer_state"] = all_df["customer_state"].astype(str).str.upper()
all_df["payment_type"] = all_df["payment_type"].astype(str).str.lower()
all_df["order_date"] = all_df["order_purchase_timestamp"].dt.date

# is_late hanya valid jika pesanan sudah sampai
if has_cols(all_df, ["order_delivered_customer_date", "order_estimated_delivery_date"]):
    all_df["is_delivered"] = all_df["order_delivered_customer_date"].notna()
    all_df["is_late"] = all_df["order_delivered_customer_date"] > all_df["order_estimated_delivery_date"]
    all_df["delivery_days"] = (all_df["order_delivered_customer_date"] - all_df["order_purchase_timestamp"]).dt.total_seconds() / 86400
    all_df["estimated_days"] = (all_df["order_estimated_delivery_date"] - all_df["order_purchase_timestamp"]).dt.total_seconds() / 86400
else:
    all_df["is_delivered"] = False
    all_df["is_late"] = False

CAT_COL = "product_category_name_english" if "product_category_name_english" in all_df.columns else (
    "product_category_name" if "product_category_name" in all_df.columns else None)

# =====================================================================
# 3. SIDEBAR FILTER
# =====================================================================

valid_dates = all_df["order_date"].dropna()
min_date, max_date = valid_dates.min(), valid_dates.max()

with st.sidebar:
    st.image("https://github.com/dicodingacademy/assets/raw/main/logo.png")
    st.title("Filter Data")
    date_range = st.date_input("Rentang Waktu", min_value=min_date, max_value=max_date,
                               value=[min_date, max_date])
    if isinstance(date_range, (tuple, list)):
        start_date = date_range[0]
        end_date = date_range[1] if len(date_range) > 1 else start_date
    else:
        start_date = end_date = date_range
    st.caption("Filter ini berlaku untuk metrik utama, Top 10 State, dan RFM. "
               "Analisis Pertanyaan 1-3 memakai periode tetap sesuai pertanyaan.")

main_df = all_df[(all_df["order_date"] >= start_date) & (all_df["order_date"] <= end_date)]

# =====================================================================
# 4. HEADER & METRIK
# =====================================================================

st.header("E-Commerce Dashboard :sparkles:")
st.subheader("Performance & Analytics Overview")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Total Orders", f"{main_df['order_id'].nunique():,}")
with c2:
    st.metric("Total Revenue", f"R$ {main_df['price'].sum():,.2f}")
with c3:
    avg_review = dedup_orders(main_df)["review_score"].mean()
    st.metric("Avg Review Score", f"{0 if pd.isna(avg_review) else avg_review:.2f} / 5.0")

st.markdown("---")
st.subheader("📌 Analisis Pertanyaan Bisnis")

tab1, tab2, tab3 = st.tabs([
    "1️⃣ Keterlambatan SP (Q4 2017)",
    "2️⃣ Boleto Rio de Janeiro (Nov 2017)",
    "3️⃣ Elektronik Belo Horizonte (H1 2018)",
])

# =====================================================================
# TAB 1 - KETERLAMBATAN SAO PAULO
# =====================================================================
with tab1:
    st.markdown("**Pertanyaan:** Faktor apa yang menyebabkan kenaikan keterlambatan pengiriman "
                "pada pelanggan di São Paulo selama kuartal keempat 2017?")

    sp_all = dedup_orders(all_df[(all_df["customer_state"] == "SP") & all_df["is_delivered"]])
    sp_all = in_period(sp_all, "2017-07-01", "2017-12-31").copy()  # Q3 sebagai pembanding
    sp_all["month"] = sp_all["order_purchase_timestamp"].dt.strftime("%Y-%m")
    sp_q4 = sp_all[sp_all["month"] >= "2017-10"].copy()

    if sp_q4.empty:
        st.warning("Data São Paulo Q4 2017 tidak tersedia.")
    else:
        monthly = sp_all.groupby("month").agg(total_orders=("is_late", "size"),
                                              late_pct=("is_late", "mean")).reset_index()
        monthly["late_pct"] *= 100
        monthly["periode"] = np.where(monthly["month"] >= "2017-10", "Q4 2017", "Q3 2017 (pembanding)")

        # ---- Faktor 1
        st.markdown("##### Faktor 1: Volume pesanan vs persentase keterlambatan (Q3 vs Q4 2017)")
        fig, ax1 = plt.subplots(figsize=(10, 5))
        bar_colors = [COLOR_MAIN if m >= "2017-10" else COLOR_GREY for m in monthly["month"]]
        ax1.bar(monthly["month"], monthly["total_orders"], color=bar_colors)
        ax1.set_ylabel("Jumlah Pesanan (batang)")
        ax1.set_xlabel("Bulan")
        ax2 = ax1.twinx()
        ax2.plot(monthly["month"], monthly["late_pct"], color=COLOR_RED, marker="o", linewidth=2.5)
        ax2.set_ylabel("Keterlambatan % (garis)", color=COLOR_RED)
        ax2.grid(False)
        for i, v in enumerate(monthly["late_pct"]):
            ax2.annotate(f"{v:.2f}%", (i, v), textcoords="offset points", xytext=(0, 8),
                         ha="center", color=COLOR_RED)
        ax1.set_title("Volume Pesanan vs % Keterlambatan - São Paulo (Jul-Des 2017)", fontsize=14, pad=12)
        st.pyplot(fig)

        q3_late = sp_all[sp_all["month"] < "2017-10"]["is_late"].mean() * 100
        q4_late = sp_q4["is_late"].mean() * 100
        peak = monthly[monthly["month"] >= "2017-10"].sort_values("late_pct", ascending=False).iloc[0]
        st.caption(f"Rata-rata keterlambatan Q3 2017: {pct(q3_late)} → Q4 2017: {pct(q4_late)}. "
                   f"Puncak pada {peak['month']} ({pct(peak['late_pct'])}) dengan {int(peak['total_orders']):,} pesanan.")

        # ---- Faktor 2: durasi
        st.markdown("##### Faktor 2: Durasi pengiriman aktual vs estimasi")
        dur = sp_q4.groupby("month")[["delivery_days", "estimated_days"]].mean().reset_index()
        dur_long = dur.melt(id_vars="month", var_name="Jenis", value_name="Hari")
        dur_long["Jenis"] = dur_long["Jenis"].map({"delivery_days": "Aktual", "estimated_days": "Estimasi"})
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x="month", y="Hari", hue="Jenis", data=dur_long, palette=[COLOR_MAIN, COLOR_GREY], ax=ax)
        ax.set_title("Rata-rata Durasi Pengiriman: Aktual vs Estimasi (Q4 2017)", fontsize=14, pad=12)
        ax.set_xlabel("Bulan")
        ax.set_ylabel("Hari")
        st.pyplot(fig)

        # ---- Faktor 3: asal seller
        if "seller_state" in sp_q4.columns:
            st.markdown("##### Faktor 3: Asal seller (dalam SP vs luar SP)")
            tmp = sp_q4.dropna(subset=["seller_state"]).copy()
            tmp["Asal Seller"] = np.where(tmp["seller_state"] == "SP", "Seller dalam SP", "Seller luar SP")
            route = tmp.groupby(["month", "Asal Seller"]).agg(late=("is_late", "mean"), n=("is_late", "size")).reset_index()
            route["late"] *= 100
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x="month", y="late", hue="Asal Seller", data=route,
                        palette=[COLOR_MAIN, COLOR_GREY], ax=ax)
            ax.set_title("% Keterlambatan Berdasarkan Asal Seller (Q4 2017)", fontsize=14, pad=12)
            ax.set_xlabel("Bulan")
            ax.set_ylabel("Keterlambatan (%)")
            st.pyplot(fig)
            share_out = (tmp["Asal Seller"] == "Seller luar SP").mean() * 100
            st.caption(f"{pct(share_out)} pesanan SP di Q4 2017 dikirim dari seller di luar SP.")

        # ---- Faktor 4: freight
        if "freight_value" in sp_q4.columns:
            st.markdown("##### Faktor 4: Biaya pengiriman (freight) pesanan telat vs tepat waktu")
            fr = sp_q4.assign(Status=np.where(sp_q4["is_late"], "Terlambat", "Tepat waktu"))
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x="Status", y="freight_value", hue="Status", data=fr, errorbar=None, legend=False,
                        order=["Tepat waktu", "Terlambat"], palette=[COLOR_MAIN, COLOR_RED], ax=ax)
            ax.set_ylabel("Rata-rata Freight (R$)")
            ax.set_xlabel(None)
            ax.set_title("Freight: Tepat Waktu vs Terlambat", fontsize=14, pad=12)
            st.pyplot(fig)

        # ---- Faktor 5: kategori
        if CAT_COL:
            st.markdown("##### Faktor 5: Kategori produk dengan keterlambatan tertinggi")
            cat_df = sp_q4.groupby(CAT_COL).agg(n=("is_late", "size"), late_pct=("is_late", "mean")).reset_index()
            cat_df = cat_df[cat_df["n"] >= 30].sort_values("late_pct", ascending=False).head(8)
            cat_df["late_pct"] *= 100
            if not cat_df.empty:
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(x="late_pct", y=CAT_COL, data=cat_df, color=COLOR_MAIN, ax=ax)
                ax.set_xlabel("Keterlambatan (%)")
                ax.set_ylabel(None)
                ax.set_title("Kategori dengan % Keterlambatan Tertinggi (min. 30 pesanan)", fontsize=14, pad=12)
                st.pyplot(fig)

        vol_q3 = sp_all[sp_all["month"] < "2017-10"].groupby("month").size().mean()
        vol_peak = peak["total_orders"]
        st.success(
            f"**Insight:** Keterlambatan pengiriman di São Paulo naik dari {pct(q3_late)} (rata-rata Q3) menjadi "
            f"{pct(peak['late_pct'])} pada {peak['month']}. Bulan puncak tersebut memiliki {int(vol_peak):,} pesanan, "
            f"sekitar {vol_peak / vol_q3:.1f}x rata-rata volume bulanan Q3, sehingga lonjakan volume pesanan "
            "(periode promosi akhir tahun) adalah faktor yang paling konsisten dengan kenaikan keterlambatan. "
            "Faktor lain pada grafik di atas bersifat pendukung dan berupa korelasi, bukan bukti sebab-akibat."
        )

# =====================================================================
# TAB 2 - BOLETO RIO DE JANEIRO
# =====================================================================
with tab2:
    st.markdown("**Pertanyaan:** Apa penyebab utama lonjakan pembatalan pesanan boleto "
                "di Rio de Janeiro pada November 2017?")

    rj = dedup_orders(all_df[all_df["customer_city"] == "rio de janeiro"])
    rj = in_period(rj, "2017-10-01", "2017-12-31").copy()
    rj["month"] = rj["order_purchase_timestamp"].dt.strftime("%Y-%m")
    rj["is_canceled"] = rj["order_status"].astype(str).str.lower().isin(["canceled", "unavailable"])
    rj_nov = rj[rj["month"] == "2017-11"]
    rj_boleto_nov = rj_nov[rj_nov["payment_type"] == "boleto"]

    if rj_boleto_nov.empty:
        st.warning("Tidak ada pesanan boleto Rio de Janeiro pada November 2017.")
    else:
        st.info(f"⚠️ Ukuran sampel kecil: hanya **{len(rj_boleto_nov)}** pesanan boleto RJ pada November 2017. "
                "Satu pesanan saja menggeser persentase sekitar 1 poin, jadi baca hasil dengan hati-hati.")

        # ---- Faktor 1
        st.markdown("##### Faktor 1: Distribusi status pesanan boleto (Nov 2017)")
        status_df = rj_boleto_nov["order_status"].value_counts().reset_index()
        status_df.columns = ["order_status", "count"]
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = [COLOR_MAIN if s == "delivered" else COLOR_GREY for s in status_df["order_status"]]
        sns.barplot(x="count", y="order_status", data=status_df, hue="order_status",
                    palette=colors, legend=False, ax=ax)
        ax.set_title("Distribusi Status Pesanan Boleto - Rio de Janeiro (Nov 2017)", fontsize=14, pad=12)
        ax.set_xlabel("Jumlah Pesanan")
        ax.set_ylabel("Status")
        st.pyplot(fig)
        pct_delivered = status_df.loc[status_df["order_status"] == "delivered", "count"].sum() / status_df["count"].sum() * 100
        st.caption(f"{pct(pct_delivered)} pesanan boleto berstatus delivered; sisanya {pct(100 - pct_delivered)} "
                   "belum selesai atau dibatalkan.")

        # ---- Faktor 2: metode bayar
        st.markdown("##### Faktor 2: Pembatalan boleto vs metode pembayaran lain (Nov 2017)")
        pay = rj_nov.groupby("payment_type").agg(n=("is_canceled", "size"), cancel=("is_canceled", "mean")).reset_index()
        pay["cancel"] *= 100
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = [COLOR_RED if p == "boleto" else COLOR_GREY for p in pay["payment_type"]]
        sns.barplot(x="payment_type", y="cancel", data=pay, hue="payment_type",
                    palette=colors, legend=False, ax=ax)
        for i, (v, n) in enumerate(zip(pay["cancel"], pay["n"])):
            ax.annotate(f"{v:.1f}%\n(n={n})", (i, v), ha="center", va="bottom")
        ax.set_xlabel("Metode Pembayaran")
        ax.set_ylabel("Pembatalan (%)")
        ax.set_title("% Pembatalan per Metode Pembayaran - RJ (Nov 2017)", fontsize=14, pad=12)
        st.pyplot(fig)

        # ---- Faktor 3: tren bulanan
        st.markdown("##### Faktor 3: Tren pembatalan boleto Okt - Des 2017")
        bol = rj[rj["payment_type"] == "boleto"].groupby("month").agg(
            n=("is_canceled", "size"), cancel=("is_canceled", "mean")).reset_index()
        bol["cancel"] *= 100
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x="month", y="cancel", data=bol, color=COLOR_MAIN, ax=ax)
        for i, (v, n) in enumerate(zip(bol["cancel"], bol["n"])):
            ax.annotate(f"{v:.2f}%\n(n={n})", (i, v), ha="center", va="bottom")
        ax.set_xlabel("Bulan")
        ax.set_ylabel("Pembatalan (%)")
        ax.set_title("Pembatalan Pesanan Boleto per Bulan - Rio de Janeiro", fontsize=14, pad=12)
        st.pyplot(fig)

        # ---- Faktor 4: harian
        st.markdown("##### Faktor 4: Pesanan boleto harian (Nov 2017)")
        daily = rj_boleto_nov.assign(day=rj_boleto_nov["order_purchase_timestamp"].dt.day) \
            .groupby("day").agg(total=("is_canceled", "size"), canceled=("is_canceled", "sum")).reset_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(daily["day"], daily["total"], color=COLOR_GREY, marker="o", label="Total pesanan")
        ax.plot(daily["day"], daily["canceled"], color=COLOR_RED, marker="o", label="Dibatalkan")
        ax.set_xlabel("Tanggal (Nov 2017)")
        ax.set_ylabel("Jumlah Pesanan")
        ax.legend()
        ax.set_title("Pesanan Boleto Harian - RJ (Nov 2017)", fontsize=14, pad=12)
        st.pyplot(fig)

        # ---- Faktor 5: nilai pesanan
        if "payment_value" in rj_boleto_nov.columns:
            st.markdown("##### Faktor 5: Nilai pembayaran boleto vs metode lain (Nov 2017)")
            val = rj_nov.groupby("payment_type")["payment_value"].mean().reset_index()
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = [COLOR_RED if p == "boleto" else COLOR_GREY for p in val["payment_type"]]
            sns.barplot(x="payment_type", y="payment_value", data=val, hue="payment_type",
                        palette=colors, legend=False, ax=ax)
            ax.set_xlabel("Metode Pembayaran")
            ax.set_ylabel("Rata-rata Nilai Pembayaran (R$)")
            ax.set_title("Nilai Pembayaran Rata-rata per Metode (RJ, Nov 2017)", fontsize=14, pad=12)
            st.pyplot(fig)

        boleto_cancel = rj_boleto_nov["is_canceled"].mean() * 100
        other_cancel = rj_nov[rj_nov["payment_type"] != "boleto"]["is_canceled"].mean() * 100
        st.success(
            f"**Insight:** Pada November 2017, {pct(pct_delivered)} pesanan boleto di Rio de Janeiro berhasil "
            f"terkirim. Tingkat pembatalan boleto hanya {pct(boleto_cancel)} (n={len(rj_boleto_nov)}), "
            f"sedangkan metode lain {pct(other_cancel)}. Data tidak menunjukkan lonjakan pembatalan 8% yang "
            "khas pada boleto, sehingga premis pertanyaan tidak didukung data. Dataset juga tidak memuat alasan "
            "pembatalan, sehingga penyebab seperti boleto kedaluwarsa hanya bisa disebut sebagai dugaan."
        )

# =====================================================================
# TAB 3 - ELEKTRONIK BELO HORIZONTE
# =====================================================================
with tab3:
    st.markdown("**Pertanyaan:** Apa faktor utama penurunan average review score kategori elektronik "
                "di Belo Horizonte pada semester pertama 2018?")

    bh = pd.DataFrame()
    if CAT_COL:
        bh = all_df[(all_df["customer_city"] == "belo horizonte")
                    & all_df[CAT_COL].astype(str).str.contains("eletr|electro|computer|audio|phone", case=False, na=False)
                    & all_df["review_score"].notna()]
        bh = in_period(dedup_orders(bh), "2018-01-01", "2018-06-30").copy()

    if bh.empty:
        st.warning("Data elektronik Belo Horizonte Jan-Jun 2018 tidak tersedia.")
    else:
        bh["month"] = bh["order_purchase_timestamp"].dt.strftime("%Y-%m")
        bh_d = bh[bh["is_delivered"]].copy()
        bh_d["Status Kirim"] = np.where(bh_d["is_late"], "Terlambat", "Tepat waktu")

        st.info(f"⚠️ Ukuran sampel kecil: **{len(bh)}** pesanan ({bh.groupby('month').size().min()}–"
                f"{bh.groupby('month').size().max()} per bulan). Kategori cakupan: {', '.join(sorted(bh[CAT_COL].dropna().unique()))}.")

        # ---- Faktor 1: tren
        st.markdown("##### Faktor 1: Tren rata-rata review score per bulan")
        trend = bh.groupby("month").agg(avg=("review_score", "mean"), n=("review_score", "size")).reset_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(trend["month"], trend["avg"], color=COLOR_MAIN, marker="o", linewidth=2.5)
        for i, (v, n) in enumerate(zip(trend["avg"], trend["n"])):
            ax.annotate(f"{v:.2f}\n(n={n})", (i, v), textcoords="offset points", xytext=(0, 8), ha="center")
        ax.set_ylim(1, 5.5)
        ax.set_xlabel("Bulan")
        ax.set_ylabel("Rata-rata Skor")
        ax.set_title("Tren Review Score Elektronik - Belo Horizonte (H1 2018)", fontsize=14, pad=12)
        st.pyplot(fig)

        # ---- Faktor 2: distribusi
        st.markdown("##### Faktor 2: Distribusi skor ulasan")
        dist = bh["review_score"].round().astype(int).value_counts().sort_index().reset_index()
        dist.columns = ["review_score", "count"]
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = [COLOR_RED if s == 1 else COLOR_GREY for s in dist["review_score"]]
        sns.barplot(x="review_score", y="count", data=dist, hue="review_score",
                    palette=colors, legend=False, ax=ax)
        ax.set_xlabel("Skor Ulasan")
        ax.set_ylabel("Jumlah Ulasan")
        ax.set_title("Distribusi Review Score Elektronik - Belo Horizonte (H1 2018)", fontsize=14, pad=12)
        st.pyplot(fig)

        # ---- Faktor 3: telat vs tepat waktu
        st.markdown("##### Faktor 3: Review score pesanan terlambat vs tepat waktu")
        late = bh_d.groupby("Status Kirim")["review_score"].agg(["mean", "size"]).reset_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x="Status Kirim", y="mean", data=late, hue="Status Kirim",
                    palette={"Tepat waktu": COLOR_MAIN, "Terlambat": COLOR_RED}, legend=False, ax=ax)
        for i, (v, n) in enumerate(zip(late["mean"], late["size"])):
            ax.annotate(f"{v:.2f}\n(n={n})", (i, v), ha="center", va="bottom")
        ax.set_ylim(1, 5.5)
        ax.set_xlabel(None)
        ax.set_ylabel("Rata-rata Skor")
        ax.set_title("Review Score: Tepat Waktu vs Terlambat", fontsize=14, pad=12)
        st.pyplot(fig)

        late_month = bh_d.groupby("month")["is_late"].mean().mul(100).reset_index()
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(x="month", y="is_late", data=late_month, color=COLOR_GREY, ax=ax)
        ax.set_ylabel("Keterlambatan (%)")
        ax.set_xlabel("Bulan")
        ax.set_title("Persentase Keterlambatan Pesanan Elektronik per Bulan", fontsize=14, pad=12)
        st.pyplot(fig)

        # ---- Faktor 4: durasi
        st.markdown("##### Faktor 4: Durasi pengiriman vs skor ulasan")
        d = bh_d.dropna(subset=["delivery_days"]).copy()
        d["Durasi"] = pd.cut(d["delivery_days"], bins=[0, 7, 14, 21, 365],
                             labels=["≤7 hari", "8-14 hari", "15-21 hari", ">21 hari"])
        dd = d.groupby("Durasi", observed=True)["review_score"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x="Durasi", y="review_score", data=dd, color=COLOR_MAIN, ax=ax)
        ax.set_ylim(1, 5)
        ax.set_ylabel("Rata-rata Skor")
        ax.set_title("Durasi Pengiriman vs Review Score", fontsize=14, pad=12)
        st.pyplot(fig)

        # ---- Faktor 5: freight
        if "freight_value" in bh.columns and bh["freight_value"].nunique() > 3:
            st.markdown("##### Faktor 5: Biaya pengiriman vs skor ulasan")
            f = bh.dropna(subset=["freight_value"]).copy()
            f["Freight"] = pd.qcut(f["freight_value"], q=3, labels=["Murah", "Sedang", "Mahal"], duplicates="drop")
            ff = f.groupby("Freight", observed=True)["review_score"].mean().reset_index()
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x="Freight", y="review_score", data=ff, color=COLOR_MAIN, ax=ax)
            ax.set_ylim(1, 5)
            ax.set_ylabel("Rata-rata Skor")
            ax.set_title("Biaya Freight vs Review Score", fontsize=14, pad=12)
            st.pyplot(fig)

        # ---- Faktor 6: seller
        if "seller_state" in bh.columns:
            s = bh.groupby("seller_state").agg(n=("review_score", "size"), avg=("review_score", "mean")).reset_index()
            s = s[s["n"] >= 10].sort_values("avg")
            if not s.empty:
                st.markdown("##### Faktor 6: Asal seller vs skor ulasan")
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(x="avg", y="seller_state", data=s, color=COLOR_MAIN, ax=ax)
                ax.set_xlim(1, 5)
                ax.set_xlabel("Rata-rata Skor")
                ax.set_ylabel(None)
                ax.set_title("Skor per Seller State (min. 10 pesanan)", fontsize=14, pad=12)
                st.pyplot(fig)

        worst = trend.sort_values("avg").iloc[0]
        late_avg = late.loc[late["Status Kirim"] == "Terlambat", "mean"]
        ontime_avg = late.loc[late["Status Kirim"] == "Tepat waktu", "mean"]
        worst_late = late_month.loc[late_month["month"] == worst["month"], "is_late"]
        txt = (f"**Insight:** Skor terendah terjadi pada {worst['month']} ({worst['avg']:.2f}, n={int(worst['n'])}). ")
        if len(late_avg) and len(ontime_avg):
            txt += (f"Pesanan terlambat rata-rata mendapat skor {late_avg.iloc[0]:.2f}, jauh di bawah pesanan "
                    f"tepat waktu ({ontime_avg.iloc[0]:.2f}). ")
        if len(worst_late):
            txt += f"Pada bulan tersebut {worst_late.iloc[0]:.1f}% pesanan terlambat. "
        txt += ("Keterlambatan pengiriman adalah faktor yang paling konsisten dengan penurunan skor; faktor lain "
                "(freight, asal seller) bersifat pendukung. Karena sampel kecil, simpulan ini bersifat indikatif.")
        st.success(txt)

st.markdown("---")

# =====================================================================
# 5. GEOSPATIAL
# =====================================================================
st.subheader("🌍 Analisis Lanjutan: Top 10 States by Total Revenue")
bystate_df = main_df.groupby("customer_state").agg(
    total_orders=("order_id", "nunique"), total_revenue=("price", "sum")
).reset_index().sort_values("total_revenue", ascending=False).head(10)

if not bystate_df.empty:
    fig, ax = plt.subplots(figsize=(16, 8))
    colors = [COLOR_MAIN if s == "SP" else COLOR_GREY for s in bystate_df["customer_state"]]
    sns.barplot(x="total_revenue", y="customer_state", data=bystate_df, hue="customer_state",
                palette=colors, legend=False, ax=ax)
    ax.set_title("Top 10 Negara Bagian Berdasarkan Total Pendapatan", fontsize=20, pad=15)
    ax.set_ylabel(None)
    ax.set_xlabel("Total Revenue (R$)", fontsize=15)
    st.pyplot(fig)
else:
    st.warning("Data geospatial kosong pada rentang waktu ini.")

st.markdown("---")

# =====================================================================
# 6. RFM
# =====================================================================
st.subheader("🏆 Analisis Lanjutan: Best Customers (RFM)")
rfm_key = "customer_unique_id" if "customer_unique_id" in main_df.columns else "customer_id"
rfm_df = main_df.dropna(subset=[rfm_key]).groupby(rfm_key, as_index=False).agg(
    last_order=("order_purchase_timestamp", "max"),
    frequency=("order_id", "nunique"),
    monetary=("price", "sum"),
)
if not rfm_df.empty:
    rfm_df["recency"] = (main_df["order_purchase_timestamp"].max() - rfm_df["last_order"]).dt.days
    rfm_df["short_id"] = rfm_df[rfm_key].astype(str).str[:8]

    c1, c2, c3 = st.columns(3)
    c1.metric("Average Recency (days)", round(rfm_df["recency"].mean(), 1))
    c2.metric("Average Frequency", round(rfm_df["frequency"].mean(), 2))
    c3.metric("Average Monetary", f"R$ {rfm_df['monetary'].mean():,.2f}")

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(35, 15))
    specs = [("recency", True, "By Recency (days)"),
             ("frequency", False, "By Frequency"),
             ("monetary", False, "By Monetary")]
    for i, (col, asc, title) in enumerate(specs):
        top5 = rfm_df.sort_values(col, ascending=asc).head(5)
        sns.barplot(y=col, x="short_id", data=top5, hue="short_id",
                    palette=[COLOR_MAIN] * 5, legend=False, ax=ax[i])
        ax[i].set_ylabel(None)
        ax[i].set_xlabel("Customer ID", fontsize=30)
        ax[i].set_title(title, fontsize=50)
        ax[i].tick_params(axis="y", labelsize=30)
        ax[i].tick_params(axis="x", labelsize=30, rotation=15)
    st.pyplot(fig)
else:
    st.warning("Data RFM kosong pada rentang waktu ini.")

st.caption("Copyright © Andi Ahmad Zaidun Dhiyaulhaq 2026")
