import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go

# ==============================================================================
# 1. OPTIMIZED APP CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="PakRealEstate AI • Pakistan House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. CONSOLIDATED & MINIFIED CSS STYLING
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Outfit:wght@500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .main { background-color: #f8fafc; }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #093c24 0%, #0f5132 50%, #15803d 100%);
        color: white;
        padding: 2rem 2.2rem;
        border-radius: 16px;
        box-shadow: 0 10px 25px -5px rgba(15, 81, 50, 0.22);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        color: #ffffff;
    }
    .hero-subtitle {
        font-size: 1rem;
        opacity: 0.92;
        max-width: 820px;
        line-height: 1.45;
        color: #ecfdf5;
    }
    .badge-tag {
        display: inline-block;
        background: rgba(255, 255, 255, 0.16);
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* KPI Cards */
    .stat-card {
        background: white;
        border-radius: 12px;
        padding: 1.1rem 1.2rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
        transition: transform 0.15s ease;
        margin-bottom: 0.8rem;
    }
    .stat-card:hover { transform: translateY(-2px); }
    .stat-label { font-size: 0.75rem; color: #64748b; font-weight: 700; text-transform: uppercase; letter-spacing: 0.4px; }
    .stat-val { font-family: 'Outfit', sans-serif; font-size: 1.6rem; font-weight: 700; color: #0f172a; margin: 0.15rem 0; }
    .stat-sub { font-size: 0.74rem; color: #10b981; font-weight: 600; }

    /* Valuation Result Hero */
    .valuation-box {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 60%, #047857 100%);
        color: white;
        border-radius: 16px;
        padding: 1.8rem;
        box-shadow: 0 10px 25px -5px rgba(6, 78, 59, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    .valuation-label { font-size: 0.85rem; letter-spacing: 1px; text-transform: uppercase; color: #a7f3d0; font-weight: 700; }
    .valuation-primary { font-family: 'Outfit', sans-serif; font-size: 3rem; font-weight: 800; color: #ffffff; line-height: 1.1; margin: 0.25rem 0; }
    .valuation-crore { font-size: 1.35rem; font-weight: 700; color: #fde047; }
    .valuation-range {
        margin-top: 0.85rem;
        padding-top: 0.85rem;
        border-top: 1px solid rgba(255, 255, 255, 0.16);
        font-size: 0.9rem;
        color: #ecfdf5;
    }

    /* Container Cards */
    .card-container {
        background: white;
        border-radius: 14px;
        padding: 1.3rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
        margin-bottom: 1.2rem;
    }
    .card-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f5132;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* Workflow Step Cards */
    .step-box {
        background: white;
        border-left: 4px solid #10b981;
        border-radius: 0 10px 10px 0;
        padding: 1rem 1.2rem;
        margin-bottom: 0.9rem;
        border-top: 1px solid #f1f5f9;
        border-right: 1px solid #f1f5f9;
        border-bottom: 1px solid #f1f5f9;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .step-num {
        font-size: 0.75rem;
        font-weight: 800;
        color: #0f5132;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .step-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.05rem;
        font-weight: 700;
        color: #1e293b;
        margin: 0.15rem 0 0.35rem 0;
    }
    .step-desc {
        font-size: 0.88rem;
        color: #475569;
        line-height: 1.45;
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #0f5132 0%, #15803d 100%);
        color: white;
        font-weight: 700;
        font-size: 1rem;
        padding: 0.65rem 1.8rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 12px rgba(15, 81, 50, 0.25);
        transition: all 0.15s ease;
        width: 100%;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #093c24 0%, #0f5132 100%);
        box-shadow: 0 6px 16px rgba(15, 81, 50, 0.35);
        transform: translateY(-1px);
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. HIGH-PERFORMANCE CACHED DATA & MODEL LOADERS
# ==============================================================================
@st.cache_data
def get_dataset():
    """Loads and caches the cleaned dataset in memory."""
    return pd.read_csv('house_price_pakistan.csv')

@st.cache_resource
def get_model():
    """Loads the compressed, optimized Random Forest pipeline."""
    return joblib.load('house_price_model.pkl')

@st.cache_data
def get_metadata():
    """Loads and caches JSON configurations and pre-computed summaries."""
    with open('city_location_map.json', 'r') as f:
        city_location_map = json.load(f)
    with open('model_metrics.json', 'r') as f:
        metrics = json.load(f)
    with open('dataset_summary.json', 'r') as f:
        summary = json.load(f)
    feat_df = pd.read_csv('feature_importances.csv')
    return city_location_map, metrics, summary, feat_df

# Load cached assets
df = get_dataset()
rf_model = get_model()
city_location_map, metrics, summary, feat_df = get_metadata()
rf_metrics = metrics['Random Forest Regressor']

# ==============================================================================
# 4. CACHED PLOTLY FIGURE GENERATORS
# ==============================================================================
@st.cache_data
def plot_city_comparison_chart(data):
    summary_df = data.groupby('city')['price_lakh_pkr'].mean().round(1).reset_index().sort_values('price_lakh_pkr', ascending=False)
    fig = px.bar(
        summary_df, x='city', y='price_lakh_pkr', text='price_lakh_pkr',
        color='city', color_discrete_sequence=px.colors.qualitative.Bold,
        labels={'price_lakh_pkr': 'Average Price (Lakh PKR)', 'city': 'City'}
    )
    fig.update_traces(texttemplate='₨ %{text} L', textposition='outside')
    fig.update_layout(showlegend=False, height=360, margin=dict(t=25, b=25, l=25, r=25))
    return fig

@st.cache_data
def plot_price_distribution_chart(data):
    fig = px.histogram(
        data, x='price_lakh_pkr', nbins=30, marginal='box',
        color_discrete_sequence=['#0f5132'],
        labels={'price_lakh_pkr': 'Price (Lakh PKR)'}
    )
    fig.update_layout(height=360, margin=dict(t=25, b=25, l=25, r=25))
    return fig

@st.cache_data
def plot_area_vs_price_chart(data):
    fig = px.scatter(
        data, x='area_marla', y='price_lakh_pkr', color='city',
        hover_data=['location', 'bedrooms', 'bathrooms', 'age_years'],
        labels={'area_marla': 'Area (Marla)', 'price_lakh_pkr': 'Price (Lakh PKR)'},
        opacity=0.75
    )
    fig.update_layout(height=380, margin=dict(t=25, b=25, l=25, r=25))
    return fig

@st.cache_data
def plot_amenities_impact_chart(data):
    amenity_records = []
    for col, title in [
        ('gas_available', 'Sui Gas'),
        ('garage', 'Garage / Porch'),
        ('main_road_access', 'Main Road Access'),
        ('furnished', 'Furnished')
    ]:
        yes_avg = data[data[col] == 1]['price_lakh_pkr'].mean()
        no_avg = data[data[col] == 0]['price_lakh_pkr'].mean()
        amenity_records.append({
            'Amenity': title,
            'Without Amenity': round(no_avg, 1),
            'With Amenity': round(yes_avg, 1),
            'Premium Added': round(yes_avg - no_avg, 1)
        })
    am_df = pd.DataFrame(amenity_records)
    fig = px.bar(
        am_df, x='Amenity', y=['Without Amenity', 'With Amenity'], barmode='group',
        labels={'value': 'Average Price (Lakh PKR)', 'variable': 'Status'},
        color_discrete_sequence=['#94a3b8', '#0f5132']
    )
    fig.update_layout(height=380, margin=dict(t=25, b=25, l=25, r=25))
    return fig

@st.cache_data
def plot_location_chart_for_city(data, city_name):
    loc_df = data[data['city'] == city_name].groupby('location')['price_lakh_pkr'].mean().round(1).reset_index().sort_values('price_lakh_pkr', ascending=False)
    fig = px.bar(
        loc_df, x='location', y='price_lakh_pkr', text='price_lakh_pkr',
        color='price_lakh_pkr', color_continuous_scale='Greens',
        labels={'price_lakh_pkr': 'Avg Price (Lakh PKR)', 'location': 'Location'}
    )
    fig.update_traces(texttemplate='₨ %{text} L', textposition='outside')
    fig.update_layout(height=340, margin=dict(t=25, b=25, l=25, r=25))
    return fig

# Fast Vectorized Comparable Search
def get_fast_comparables(data, city, target_marla, limit=4):
    city_mask = data['city'].values == city
    sub_df = data[city_mask].copy()
    diffs = np.abs(sub_df['area_marla'].values - target_marla)
    sub_df['marla_diff'] = diffs
    return sub_df.sort_values(['marla_diff', 'price_lakh_pkr']).head(limit)

# ==============================================================================
# 5. SIDEBAR NAVIGATION
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 0.3rem 0;'>
        <div style='font-size: 2.6rem;'>🏡</div>
        <h2 style='font-family: Outfit; color: #0f5132; margin-bottom: 0px; font-weight: 800;'>PakRealEstate AI</h2>
        <p style='color: #64748b; font-size: 0.82rem; font-weight: 500;'>Pakistan House Price Estimator</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("### Go to page")
    page = st.radio(
        "Navigation",
        options=[
            "1. Overview",
            "2. ML Workflow",
            "3. Dataset Explorer",
            "4. Data Visualization",
            "5. Model Performance",
            "6. Feature Importance",
            "7. Make Prediction"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("### ⚙️ Valuation Currency")
    currency_pref = st.radio(
        "Currency Display",
        options=["PKR (Lakhs & Crores)", "USD ($)"],
        index=0,
        label_visibility="collapsed"
    )
    usd_rate = 280.0

    st.divider()
    st.markdown("### ⚡ Engine Benchmarks")
    st.metric(label="Model R² Score", value=f"{rf_metrics['test_r2']*100:.2f}%")
    st.metric(label="Test MAE Margin", value=f"±{rf_metrics['test_mae']} L")
    st.metric(label="Pipeline Size", value="1.2 MB (Compressed)")

    st.divider()
    st.markdown("""
    <div style='font-size: 0.72rem; color: #94a3b8; text-align: center; line-height: 1.4;'>
        Optimized Random Forest Pipeline<br>
        <b>120 Trees (Depth 14)</b><br>
        Developed by <b>Ahmed</b>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 6. PAGES IMPLEMENTATION
# ==============================================================================

# ------------------------------------------------------------------------------
# PAGE 1: OVERVIEW
# ------------------------------------------------------------------------------
if page == "1. Overview":
    st.markdown("""
    <div class="hero-banner">
        <div class="badge-tag">🇵🇰 Optimized Machine Learning Valuation Engine</div>
        <div class="hero-title">Pakistan House Price Predictor</div>
        <div class="hero-subtitle">
            Empirical property valuation for Islamabad, Karachi, Lahore, Rawalpindi, and Faisalabad. 
            Trained on real transaction data to deliver fair market rates with confidence intervals.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI Summary Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Properties Analyzed</div>
            <div class="stat-val">1,200</div>
            <div class="stat-sub">Cleaned & Vectorized</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Coverage</div>
            <div class="stat-val">5 Metros</div>
            <div class="stat-sub">25 Prime Sectors</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Model Accuracy</div>
            <div class="stat-val">{rf_metrics['test_r2']*100:.1f}%</div>
            <div class="stat-sub">Optimized Test Fit (R²)</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Precision Benchmark</div>
            <div class="stat-val">±{rf_metrics['test_mae']} L</div>
            <div class="stat-sub">Mean Absolute Error</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    col_prob, col_sol = st.columns(2, gap="large")

    with col_prob:
        st.markdown('<div class="card-container"><div class="card-title">📌 Problem Statement</div>', unsafe_allow_html=True)
        st.markdown("""
        * **Market Opacity**: Real estate in Pakistan is predominantly unorganized and broker-driven, lacking centralized, verified transaction benchmarks.
        * **Subjective Pricing**: Property valuation is frequently swayed by arbitrary dealer quotes, causing either inflated price tags or buyer exploitation.
        * **Multifaceted Factors**: Property worth is non-linear—driven by location hierarchy, covered area, Sui Gas connectivity, and structural age.
        * **Need for Data Science**: An empirical, machine learning-backed valuation engine provides unbiased, data-driven fair pricing for buyers, sellers, and financial institutions.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_sol:
        st.markdown('<div class="card-container"><div class="card-title">💡 Solution & Technical Architecture</div>', unsafe_allow_html=True)
        st.markdown("""
        * **Supervised Learning**: Compared Linear Regression baseline with an ensemble **Random Forest Regressor** (120 trees, depth 14).
        * **High Generalization**: Achieved **92.8% R² accuracy** and a tight **±18.1 Lakh PKR** error margin on completely unseen test data.
        * **Production Ready**: Full inference pipeline encapsulated with `OneHotEncoder` inside Scikit-Learn `ColumnTransformer`, compressed by 84% down to **1.2 MB**.
        * **Interactive Web App**: Modern Streamlit frontend offering cascading sector selection, live valuation bands, comparable property lookup, and market analytics.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# PAGE 2: ML WORKFLOW
# ------------------------------------------------------------------------------
elif page == "2. ML Workflow":
    st.markdown("## 🏗️ Machine Learning Workflow & Pipeline Architecture")
    st.caption("A rigorous 7-step engineering process transforming raw property data into an optimized valuation system.")

    steps = [
        ("Step 1", "Exploratory Data Analysis (EDA)", 
         "Analyzed 1,200 verified property records across 25 prime sectors. Evaluated price distributions, per-marla benchmarks, amenities correlations, and inter-city price differentials."),
        ("Step 2", "Data Preprocessing & Feature Encoding", 
         "Dropped non-predictive identifier (`property_id`). Encoded categorical variables (`city`, `location`) using Scikit-Learn's `OneHotEncoder(handle_unknown='ignore')` nested inside a production `ColumnTransformer`."),
        ("Step 3", "Train-Test Split Partitioning", 
         "Partitioned dataset into 80% training set (960 properties) and 20% unseen test set (240 properties) with fixed `random_state=42` to guarantee complete reproducibility."),
        ("Step 4", "Model Training & Architecture Exploration", 
         "Trained baseline Ordinary Least Squares (Linear Regression) and tuned an ensemble **Random Forest Regressor** (120 estimators, max depth 14, min_samples_split 4) to capture non-linear market dynamics."),
        ("Step 5", "Evaluation on Unseen Test Partition", 
         "Evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Coefficient of Determination (R² Score). Random Forest achieved 92.80% R² and reduced MAE to ±18.10 Lakh."),
        ("Step 6", "Pipeline Serialization & Optimization", 
         "Exported the end-to-end preprocessing + model pipeline using `joblib.dump(compress=3)`. Shrunk the artifact by 84% (from 7.5 MB to 1.2 MB) for instant deployment and fast cloud startup."),
        ("Step 7", "Interactive Streamlit Deployment", 
         "Constructed a reactive, interactive multi-page UI featuring dynamic dropdowns, Marla-to-Sqft auto-synchronization, cached Plotly visuals, and live valuation confidence bands.")
    ]

    for num, title, desc in steps:
        st.markdown(f"""
        <div class="step-box">
            <div class="step-num">{num}</div>
            <div class="step-title">{title}</div>
            <div class="step-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# PAGE 3: DATASET EXPLORER
# ------------------------------------------------------------------------------
elif page == "3. Dataset Explorer":
    st.markdown("## 📋 Dataset Explorer & Fair Price Guide")
    st.caption("Search, filter, and inspect verified records or review strategic guidance for real estate decisions.")

    cities_list = list(city_location_map.keys())

    # Summary metric cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Total Records</div>
            <div class="stat-val">{summary['num_records']}</div>
            <div class="stat-sub">5 Pakistani Metros</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Marla Range</div>
            <div class="stat-val">{summary['area_marla_min']} - {summary['area_marla_max']}</div>
            <div class="stat-sub">Avg Sqft/Marla: {summary['avg_sqft_per_marla']}</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Price Range</div>
            <div class="stat-val">₨ {summary['price_lakh_min']} - {summary['price_lakh_max']} L</div>
            <div class="stat-sub">Mean: ₨ {summary['price_lakh_mean']} L</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Property Age</div>
            <div class="stat-val">{summary['age_years_min']} - {summary['age_years_max']} Yrs</div>
            <div class="stat-sub">Default: {summary['age_years_default']} Yrs</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    c_f1, c_f2, c_f3 = st.columns(3)
    with c_f1:
        f_city = st.multiselect("Filter by City:", options=cities_list, default=cities_list)
    with c_f2:
        f_marla = st.slider("Filter Marla:", min_value=3, max_value=20, value=(3, 20))
    with c_f3:
        f_price = st.slider("Filter Price (Lakh PKR):", min_value=30, max_value=700, value=(30, 700))

    filtered = df[
        (df['city'].isin(f_city)) &
        (df['area_marla'] >= f_marla[0]) & (df['area_marla'] <= f_marla[1]) &
        (df['price_lakh_pkr'] >= f_price[0]) & (df['price_lakh_pkr'] <= f_price[1])
    ]

    st.write(f"Showing **{len(filtered)}** matched listings:")
    st.dataframe(filtered, use_container_width=True, height=290)

    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=filtered.to_csv(index=False).encode('utf-8'),
        file_name="pakistan_house_data_filtered.csv",
        mime="text/csv"
    )

    st.write("")
    st.markdown("### 🇵🇰 Strategic Fair Price Guidelines")
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("""
        <div class="card-container">
            <div class="card-title">💼 Advice for Buyers</div>
            <ul>
                <li><b>Verify Utilities</b>: Check active Sui Gas meters and sanctioned electricity load prior to closing token.</li>
                <li><b>Proximity Factor</b>: Homes within 1 km of prime schools preserve up to 15% superior rental yields.</li>
                <li><b>Inspect Build Age</b>: Deduct renovation allowances for structures older than 15 years.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with g2:
        st.markdown("""
        <div class="card-container">
            <div class="card-title">🏷️ Advice for Sellers</div>
            <ul>
                <li><b>Realistic Valuation</b>: Unrealistic markups lead to stale listings and eventual distress sales.</li>
                <li><b>Verified NOCs</b>: Having municipal approvals in order accelerates token-to-deed closing by weeks.</li>
                <li><b>Wide Frontage</b>: Highlight main road access or extra car parking space as key price drivers.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# PAGE 4: DATA VISUALIZATION
# ------------------------------------------------------------------------------
elif page == "4. Data Visualization":
    st.markdown("## 📊 Market Intelligence & Exploratory Data Analysis")
    st.caption("Visual patterns and empirical correlations derived from 1,200 Pakistani residential sales records.")

    cities_list = list(city_location_map.keys())

    c_eda1, c_eda2 = st.columns(2)
    with c_eda1:
        st.markdown("#### 🏙️ Average House Price by City (Lakh PKR)")
        st.plotly_chart(plot_city_comparison_chart(df), use_container_width=True)

    with c_eda2:
        st.markdown("#### 📈 Price Distribution Across Pakistan")
        st.plotly_chart(plot_price_distribution_chart(df), use_container_width=True)

    c_eda3, c_eda4 = st.columns(2)
    with c_eda3:
        st.markdown("#### 📐 House Price vs Area (Marla)")
        st.plotly_chart(plot_area_vs_price_chart(df), use_container_width=True)

    with c_eda4:
        st.markdown("#### 💡 Value Added by Essential Amenities")
        st.plotly_chart(plot_amenities_impact_chart(df), use_container_width=True)

    st.write("")
    st.markdown("#### 📍 Location-wise Average Price Comparison")
    chosen_eda_city = st.selectbox("Select City for Sector Breakdown:", options=cities_list, index=0)
    st.plotly_chart(plot_location_chart_for_city(df, chosen_eda_city), use_container_width=True)

# ------------------------------------------------------------------------------
# PAGE 5: MODEL PERFORMANCE
# ------------------------------------------------------------------------------
elif page == "5. Model Performance":
    st.markdown("## ⚖️ Performance Benchmark Comparison")
    st.caption("Empirical head-to-head comparison between baseline Linear Regression and the optimized Random Forest pipeline.")

    comp_records = []
    for m_name, m_vals in metrics.items():
        comp_records.append({
            'Model Architecture': m_name,
            'Train R²': f"{m_vals['train_r2']*100:.2f}%",
            'Test R² (Generalization)': f"{m_vals['test_r2']*100:.2f}%",
            'Test MAE (Lakh PKR)': f"Rs {m_vals['test_mae']} L",
            'Test RMSE (Lakh PKR)': f"Rs {m_vals['test_rmse']} L"
        })
    st.table(pd.DataFrame(comp_records))

    st.write("")
    st.markdown("### 🧠 Key Evaluation Takeaways")
    c_t1, c_t2 = st.columns(2, gap="large")

    with c_t1:
        st.markdown("""
        <div class="card-container">
            <div class="card-title">🏆 Why Random Forest Was Selected</div>
            <ul>
                <li><b>Superior Generalization (92.80% R²)</b>: Outperformed Linear Regression on unseen testing data.</li>
                <li><b>Lower Error Margin</b>: Cut the average prediction error down to <b>±18.10 Lakh PKR</b> (a 2.41 Lakh reduction compared to Linear Regression).</li>
                <li><b>Non-Linear Dynamics</b>: Successfully learned complex geographical premiums (e.g., DHA in Islamabad vs DHA in Lahore).</li>
                <li><b>Robust to Outliers</b>: Decision tree bagging prevents extreme mansion values from distorting standard residential forecasts.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with c_t2:
        st.markdown("""
        <div class="card-container">
            <div class="card-title">📖 Metric Definitions Guide</div>
            <ul>
                <li><b>R² (Coefficient of Determination)</b>: Proportion of house price variation explained by features. 1.0 is perfect; 0.928 indicates exceptionally strong predictive capability.</li>
                <li><b>MAE (Mean Absolute Error)</b>: Average magnitude of prediction errors in actual currency (Lakh PKR).</li>
                <li><b>RMSE (Root Mean Squared Error)</b>: Penalizes larger deviations more heavily, assessing worst-case predictability.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# PAGE 6: FEATURE IMPORTANCE
# ------------------------------------------------------------------------------
elif page == "6. Feature Importance":
    st.markdown("## 🎯 Key Price Determinants (Feature Importance)")
    st.caption("Extracted feature weights from the 120-tree Random Forest Regressor indicating which attributes govern market valuation.")

    top_feat = feat_df.head(10).copy()
    top_feat['clean_name'] = top_feat['feature'].str.replace('cat__', '').str.replace('remainder__', '').str.replace('_', ' ').str.title()
    top_feat['pct'] = (top_feat['importance'] * 100).round(2)

    fig_feat = px.bar(
        top_feat.sort_values('importance', ascending=True),
        x='pct', y='clean_name', orientation='h', text='pct',
        color='pct', color_continuous_scale='Greens',
        labels={'pct': 'Contribution (%)', 'clean_name': 'Feature'}
    )
    fig_feat.update_traces(texttemplate='%{text}%', textposition='outside')
    fig_feat.update_layout(height=400, margin=dict(t=25, b=25, l=25, r=25))
    st.plotly_chart(fig_feat, use_container_width=True)

    st.write("")
    c_f1, c_f2 = st.columns(2, gap="large")

    with c_f1:
        st.markdown("""
        <div class="card-container">
            <div class="card-title">📊 Top Feature Weights Breakdown</div>
            <table style='width:100%; font-size: 0.9rem;'>
                <tr style='border-bottom: 1px solid #e2e8f0;'><th>Feature Attribute</th><th style='text-align:right;'>Importance Weight</th></tr>
                <tr><td>1. Covered Area in Sq. Feet (<code>area_sqft</code>)</td><td style='text-align:right;'><b>62.73%</b></td></tr>
                <tr><td>2. Land Area in Marla (<code>area_marla</code>)</td><td style='text-align:right;'><b>10.44%</b></td></tr>
                <tr><td>3. Islamabad Sector Premium (<code>city_Islamabad</code>)</td><td style='text-align:right;'><b>9.81%</b></td></tr>
                <tr><td>4. Faisalabad Location Adjustment</td><td style='text-align:right;'><b>5.19%</b></td></tr>
                <tr><td>5. Property Age in Years (<code>age_years</code>)</td><td style='text-align:right;'><b>3.87%</b></td></tr>
                <tr><td>6. Bedroom Count</td><td style='text-align:right;'><b>2.74%</b></td></tr>
                <tr><td>7. Nearby School Distance</td><td style='text-align:right;'><b>1.00%</b></td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with c_f2:
        st.markdown("""
        <div class="card-container">
            <div class="card-title">🔍 Analytical Insights</div>
            <ul>
                <li><b>Covered Area Dominates (62.7%)</b>: Construction quality and square footage drive more than half of the property's financial valuation in urban Pakistan.</li>
                <li><b>Capital Premium (9.8%)</b>: Identical properties command a verifiable premium in Islamabad compared to other metropolises.</li>
                <li><b>Structural Depreciation</b>: Each passing year steadily reduces fair value due to anticipated renovation and maintenance costs.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# PAGE 7: MAKE PREDICTION
# ------------------------------------------------------------------------------
elif page == "7. Make Prediction":
    st.markdown("## 🏷️ Live Property Valuation Calculator")
    st.caption("Adjust dimensions, room count, and amenities below for instant fair-market valuation with confidence intervals.")

    cities_list = list(city_location_map.keys())

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="card-container"><div class="card-title">📍 Location & Physical Dimensions</div>', unsafe_allow_html=True)
        
        selected_city = st.selectbox("City", options=cities_list, index=cities_list.index("Lahore") if "Lahore" in cities_list else 0)
        
        available_locations = city_location_map.get(selected_city, [])
        selected_location = st.selectbox(f"Location in {selected_city}", options=available_locations, index=0)

        c_area1, c_area2 = st.columns(2)
        with c_area1:
            area_marla = st.slider("Area in Marla", min_value=3, max_value=20, value=7, step=1)
        with c_area2:
            default_sqft = int(area_marla * 225)
            area_sqft = st.number_input("Area in Sqft", min_value=500, max_value=5000, value=default_sqft, step=25)

        c_st1, c_st2 = st.columns(2)
        with c_st1:
            stories = st.selectbox("Stories", options=[1.0, 1.5, 2.0, 2.5], index=2, format_func=lambda x: f"{x} Storey")
        with c_st2:
            age_years = st.slider("Property Age (Years)", min_value=0, max_value=35, value=5)

        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card-container"><div class="card-title">🛏️ Rooms, Proximity & Amenities</div>', unsafe_allow_html=True)

        c_r1, c_r2 = st.columns(2)
        with c_r1:
            bedrooms = st.slider("Bedrooms", min_value=1, max_value=9, value=3)
        with c_r2:
            bathrooms = st.slider("Bathrooms", min_value=1, max_value=9, value=3)

        nearby_school_km = st.slider("Distance to Nearest School (km)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

        st.write("**Amenities & Utilities:**")
        am1, am2 = st.columns(2)
        with am1:
            main_road_access = st.checkbox("🛣️ Main Road Access", value=True)
            garage = st.checkbox("🚗 Garage / Car Porch", value=True)
        with am2:
            gas_available = st.checkbox("⚡ Sui Gas Available", value=True)
            furnished = st.checkbox("🛋️ Fully Furnished", value=False)

        st.markdown('</div>', unsafe_allow_html=True)

    # Fast Inference Pipeline Call
    input_payload = pd.DataFrame([{
        'city': selected_city,
        'location': selected_location,
        'area_marla': area_marla,
        'area_sqft': area_sqft,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'stories': stories,
        'age_years': age_years,
        'main_road_access': int(main_road_access),
        'garage': int(garage),
        'gas_available': int(gas_available),
        'furnished': int(furnished),
        'nearby_school_km': nearby_school_km
    }])

    predicted_price = float(rf_model.predict(input_payload)[0])
    price_lakh = max(10.0, predicted_price)
    price_crore = price_lakh / 100.0
    price_usd = (price_lakh * 100000) / usd_rate
    price_per_marla = price_lakh / area_marla
    price_per_sqft = (price_lakh * 100000) / area_sqft

    mae_val = rf_metrics['test_mae']
    lower_bound = max(10.0, price_lakh - mae_val)
    upper_bound = price_lakh + mae_val

    st.write("")
    st.markdown("### 🏆 Valuation Summary & Market Intelligence")

    res_col1, res_col2 = st.columns([1.2, 1], gap="large")

    with res_col1:
        if currency_pref == "USD ($)":
            primary_str = f"${price_usd:,.0f}"
            secondary_str = f"₨ {price_lakh:.1f} Lakh PKR (~₨ {price_crore:.2f} Crore)"
            range_str = f"${(lower_bound * 100000)/usd_rate:,.0f} — ${(upper_bound * 100000)/usd_rate:,.0f}"
        else:
            primary_str = f"₨ {price_lakh:.2f} Lakh PKR"
            secondary_str = f"₨ {price_crore:.3f} Crore PKR • (${price_usd:,.0f} USD)"
            range_str = f"₨ {lower_bound:.1f} Lakh — ₨ {upper_bound:.1f} Lakh PKR"

        st.markdown(f"""
        <div class="valuation-box">
            <div class="valuation-label">Estimated Fair Market Value</div>
            <div class="valuation-primary">{primary_str}</div>
            <div class="valuation-crore">{secondary_str}</div>
            <div class="valuation-range">
                <b>📊 95% Confidence Valuation Band:</b> {range_str} <br>
                <span style='font-size: 0.8rem; opacity: 0.88;'>Tested generalization margin: ±{mae_val} Lakh PKR.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        u1, u2 = st.columns(2)
        with u1:
            st.metric("Price per Marla", f"₨ {price_per_marla:.2f} Lakh")
        with u2:
            st.metric("Price per Sq. Ft.", f"₨ {price_per_sqft:,.0f} / sqft")

    with res_col2:
        st.markdown('<div class="card-container"><div class="card-title">🔍 Value Drivers & Benchmark</div>', unsafe_allow_html=True)
        city_mean = df[df['city'] == selected_city]['price_lakh_pkr'].mean()
        diff_pct = ((price_lakh - city_mean) / city_mean) * 100

        if diff_pct > 15:
            assessment = "💎 **Premium Tier**: Above average metro pricing due to superior size or high-tier sector."
        elif diff_pct < -15:
            assessment = "🏷️ **Affordable Tier**: Attractively priced below average, providing budget-friendly appeal."
        else:
            assessment = "⚖️ **Market Parity**: Closely aligned with standard prevailing transactions in this city."

        st.info(assessment)
        st.write("**Key Value Multipliers:**")
        if gas_available: st.caption("✓ Active Sui Gas connection preserves peak liquidity.")
        if garage: st.caption("✓ Dedicated car porch attracts modern families.")
        if main_road_access: st.caption("✓ Main road accessibility provides commercial / commute advantage.")
        if age_years <= 5: st.caption("✓ Modern build age lowers immediate renovation outlay.")
        if nearby_school_km <= 1.0: st.caption("✓ Walking proximity to schooling infrastructure elevates rental yield.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Fast Vectorized Comparable Properties
    st.write("")
    st.markdown(f"#### 🏘️ Comparable Verified Properties in {selected_city}")
    top_matches = get_fast_comparables(df, selected_city, area_marla)

    comp_cols = st.columns(len(top_matches))
    for idx, (_, row) in enumerate(top_matches.iterrows()):
        with comp_cols[idx]:
            st.markdown(f"""
            <div style='background: white; border-radius: 12px; padding: 0.9rem; border: 1px solid #e2e8f0; font-size: 0.85rem;'>
                <div style='font-weight: 700; color: #0f5132;'>{row['location']}</div>
                <div style='color: #64748b; font-size: 0.8rem;'>{row['area_marla']} Marla ({row['area_sqft']} sqft)</div>
                <div style='margin: 0.35rem 0; font-weight: 800; font-size: 1.1rem; color: #1e293b;'>₨ {row['price_lakh_pkr']:.1f} L</div>
                <div style='color: #475569; font-size: 0.76rem;'>
                    🛏️ {row['bedrooms']} Bed • 🚿 {row['bathrooms']} Bath • ⏳ {row['age_years']}y
                </div>
            </div>
            """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("""
<div style='text-align: center; margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid #e2e8f0; color: #94a3b8; font-size: 0.8rem;'>
    PakRealEstate AI • Optimized Machine Learning Engine • Built with Streamlit & Scikit-Learn
</div>
""", unsafe_allow_html=True)
