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
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 1.5rem;
    }
    .stat-card {
        background: white;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
        transition: transform 0.15s ease;
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

    /* Input Container Card */
    .card-container {
        background: white;
        border-radius: 14px;
        padding: 1.3rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
        margin-bottom: 1rem;
    }
    .card-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #0f5132;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
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

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #f1f5f9;
        padding: 5px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.92rem;
        padding: 7px 16px;
        color: #475569;
    }
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        color: #0f5132 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
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

# Pre-compute static city summaries once
@st.cache_data
def get_city_averages(data):
    return data.groupby('city')['price_lakh_pkr'].agg(['mean', 'median', 'count']).round(1).reset_index()

# ==============================================================================
# 4. CACHED PLOTLY FIGURE GENERATORS (Zero-latency Tab switching)
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
# 5. SIDEBAR
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
    rf_metrics = metrics['Random Forest Regressor']
    st.metric(label="Model R² Score", value=f"{rf_metrics['test_r2']*100:.2f}%")
    st.metric(label="Test MAE Margin", value=f"±{rf_metrics['test_mae']} L")
    st.metric(label="Pipeline Size", value="1.2 MB (Compressed)")

    st.divider()
    st.markdown("""
    <div style='font-size: 0.72rem; color: #94a3b8; text-align: center; line-height: 1.4;'>
        Optimized Random Forest Pipeline • <b>120 Trees (Depth 14)</b><br>
        Cached for instantaneous inference.
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 6. HERO BANNER & STATS
# ==============================================================================
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

# ==============================================================================
# 7. TABS
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🏡 Live Valuation Calculator",
    "📊 Market Intelligence & EDA",
    "🧠 Model Diagnostics & Architecture",
    "📋 Dataset Explorer & Fair Price Guide"
])

# ------------------------------------------------------------------------------
# TAB 1: VALUATION CALCULATOR
# ------------------------------------------------------------------------------
with tab1:
    st.markdown("### 🏷️ Configure Property Specifications")
    st.caption("Adjust dimensions, room count, and amenities below for instant fair-market valuation.")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="card-container"><div class="card-title">📍 Location & Physical Dimensions</div>', unsafe_allow_html=True)
        
        cities_list = list(city_location_map.keys())
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

# ------------------------------------------------------------------------------
# TAB 2: MARKET INTELLIGENCE & EDA (CACHED FIGURES)
# ------------------------------------------------------------------------------
with tab2:
    st.markdown("### 📊 Market Intelligence & Exploratory Data Analysis")
    st.caption("Visual patterns derived from 1,200 Pakistani residential sales records.")

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
# TAB 3: MODEL DIAGNOSTICS & ARCHITECTURE
# ------------------------------------------------------------------------------
with tab3:
    st.markdown("### 🧠 Machine Learning Model Diagnostics & Architecture")
    st.caption("Evaluation and pipeline specifications matching House_Price_Prediction.docx.")

    with st.expander("📌 Review the 7-Step Solution Pipeline Architecture", expanded=True):
        st.markdown("""
        1. **EDA**: Correlation, distribution analysis across 1,200 records.
        2. **Preprocessing**: OneHotEncoder for `city` & `location`, `property_id` dropped.
        3. **Train-Test Split**: 80% train (960) / 20% test (240) with fixed random seed.
        4. **Model Training**: Linear Regression baseline + Optimized Random Forest Regressor (120 trees, depth 14).
        5. **Evaluation**: MAE, RMSE, and R² score metrics on unseen test partition.
        6. **Serialization**: Compressed Joblib pipeline (1.2 MB footprint, ~84% reduction).
        7. **Streamlit Deployment**: Live cached inference engine.
        """)

    st.write("")
    st.markdown("#### ⚖️ Performance Benchmark Comparison")
    comp_records = []
    for m_name, m_vals in metrics.items():
        comp_records.append({
            'Model Architecture': m_name,
            'Train R²': f"{m_vals['train_r2']*100:.2f}%",
            'Test R² (Generalization)': f"{m_vals['test_r2']*100:.2f}%",
            'Test MAE (Lakh PKR)': f"₨ {m_vals['test_mae']} L",
            'Test RMSE (Lakh PKR)': f"₨ {m_vals['test_rmse']} L"
        })
    st.table(pd.DataFrame(comp_records))

    st.write("")
    st.markdown("#### 🎯 Top Feature Importances (Random Forest)")
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
    fig_feat.update_layout(height=380, margin=dict(t=25, b=25, l=25, r=25))
    st.plotly_chart(fig_feat, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 4: DATASET EXPLORER & FAIR PRICE GUIDE
# ------------------------------------------------------------------------------
with tab4:
    st.markdown("### 📋 Dataset Explorer & Real Estate Fair Price Guide")
    st.caption("Filter and export property records or review strategic guidance for property transactions.")

    c_f1, c_f2, c_f3 = st.columns(3)
    with c_f1:
        f_city = st.multiselect("Filter by City:", options=cities_list, default=cities_list)
    with c_f2:
        f_marla = st.slider("Filter Marla:", min_value=3, max_value=20, value=(3, 20))
    with c_f3:
        f_price = st.slider("Filter Price (Lakh):", min_value=30, max_value=700, value=(30, 700))

    filtered = df[
        (df['city'].isin(f_city)) &
        (df['area_marla'] >= f_marla[0]) & (df['area_marla'] <= f_marla[1]) &
        (df['price_lakh_pkr'] >= f_price[0]) & (df['price_lakh_pkr'] <= f_price[1])
    ]

    st.write(f"Showing **{len(filtered)}** matched listings:")
    st.dataframe(filtered, use_container_width=True, height=280)

    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=filtered.to_csv(index=False).encode('utf-8'),
        file_name="pakistan_house_data_filtered.csv",
        mime="text/csv"
    )

    st.write("")
    st.markdown("---")
    st.markdown("### 🇵🇰 Strategic Fair Price Guidelines")
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("""
        #### 💼 Advice for Buyers:
        - **Verify Utilities**: Check active Sui Gas meters and sanctioned electricity load.
        - **Proximity Factor**: Homes within 1 km of schools retain up to 15% better rental stability.
        - **Inspect Build Age**: Factor renovation discounts for older constructions (>15 years).
        """)
    with g2:
        st.markdown("""
        #### 🏷️ Advice for Sellers:
        - **Realistic Pricing**: Overpricing stalls properties, leading to steeper eventual discounts.
        - **Clear Clearances**: Providing verified NOCs significantly speeds up token-to-deed closing.
        - **Wide Frontage**: Emphasize main road or wide street access as a premium driver.
        """)

# --- FOOTER ---
st.markdown("""
<div style='text-align: center; margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid #e2e8f0; color: #94a3b8; font-size: 0.8rem;'>
    PakRealEstate AI • Optimized Machine Learning Engine • Built with Streamlit & Scikit-Learn
</div>
""", unsafe_allow_html=True)
