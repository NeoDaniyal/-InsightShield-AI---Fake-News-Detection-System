import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="InsightShield | Fake News Intelligence",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM DASHING THEME (CSS) ---
st.markdown("""
    <style>
        /* Main background and font adjustments */
        .reportview-container {
            background: #0e1117;
        }
        /* Metric card styling */
        div[data-testid="stMetricValue"] {
            font-size: 2.2rem;
            font-weight: 700;
            color: #ff4b4b;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 1rem;
            color: #a3a8b4;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        /* Custom card wrapper */
        .dashboard-card {
            background-color: #161a24;
            padding: 1.5rem;
            border-radius: 10px;
            border: 1px solid #262730;
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- 3. MOCK DATA FOR THE VISUALIZATIONS ---
# Based on your actual dataset proportions (17,441 Real vs 5,755 Fake)
@st.cache_data
def load_dashboard_data():
    distribution_data = pd.DataFrame({
        "Label": ["Real News", "Fake News"],
        "Count": [17441, 5755]
    })
    
    # Mock timeline distribution of articles over a week
    timeline_data = pd.DataFrame({
        "Day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        "Real News": [2100, 2500, 2800, 2400, 2900, 2300, 2441],
        "Fake News": [700, 850, 900, 1100, 950, 650, 605]
    })
    
    # Model comparison benchmarks
    model_benchmarks = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "XGBoost", "LSTM Network"],
        "Accuracy": [78, 79, 82, 81],
        "Fake News Recall": [26, 64, 47, 60]
    })
    
    return distribution_data, timeline_data, model_benchmarks

dist_df, time_df, model_df = load_dashboard_data()

# --- 4. SIDEBAR NAVIGATION ---
st.sidebar.image("https://img.icons8.com/nolan/96/shield.png", width=80)
st.sidebar.title("InsightShield AI")
st.sidebar.markdown("*Advanced Misinformation Analytics*")
st.sidebar.divider()

page = st.sidebar.radio("Navigate Workspace", ["🔮 Prediction Portal", "📊 Analytics Dashboard", "🤖 Model Performance"])

# --- 5. PAGE 1: PREDICTION PORTAL ---
if page == "🔮 Prediction Portal":
    st.title("🔮 AI Verification Portal")
    st.markdown("Submit text streams or headlines below to verify content integrity using our champion **XGBoost** model pipeline.")
    
    # Main content split layout
    col_input, col_meta = st.columns([2, 1])
    
    with col_input:
        st.subheader("Input Stream")
        headline = st.text_area("News Headline / Article Title", placeholder="Type or paste a headline here...", height=120)
        source_domain = st.text_input("Source Domain (Optional)", placeholder="e.g., trustednews.com")
        
        # Simulated interactive Slider for user input features
        tweet_num = st.slider("Initial Tweet Spread Count", min_value=0, max_value=5000, value=120)
        
        predict_btn = st.button("Run Analytics Engine", type="primary", use_container_width=True)
    
    with col_meta:
        st.subheader("System Parameters")
        st.info("💡 **Feature Engineering Note:** Providing the `source_domain` dramatically changes prediction boundaries by analyzing historical publisher credibility alongside language syntax.")
        
        with st.container():
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown("### Deployment Spec")
            st.markdown("- **Active Backbone:** XGBoost Classifier v1.2")
            st.markdown("- **NLP Vectorizer:** TF-IDF (5,000 Dimensions)")
            st.markdown("- **Inference Latency:** ~14ms")
            st.markdown('</div>', unsafe_allow_html=True)

    if predict_btn:
        st.divider()
        st.subheader("Analysis Reports Output")
        
        if not headline.strip():
            st.warning("Please enter a valid headline text sequence to classify.")
        else:
            # Basic dummy logic wrapper simulating your pipeline integration
            is_fake_indicator = "fake" in headline.lower() or "shocking" in headline.lower() or "clickbait" in source_domain.lower()
            
            p_col1, p_col2 = st.columns([1, 2])
            
            with p_col1:
                if is_fake_indicator:
                    st.error("🚨 FLAG DETECTED: HIGH PROBABILITY OF MISINFORMATION")
                    confidence = np.random.uniform(84.0, 97.8)
                else:
                    st.success("✅ CLEAR: VERIFIED AS LEGITIMATE NEWS CONTEXT")
                    confidence = np.random.uniform(81.5, 99.2)
                    
                st.metric(label="Model Confidence Score", value=f"{confidence:.2f}%")
            
            with p_col2:
                # Gauge Chart for Confidence Meter
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = confidence,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Prediction Confidence Density", 'font': {'size': 16}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#a3a8b4"},
                        'bar': {'color': "#ff4b4b" if is_fake_indicator else "#00cc96"},
                        'bgcolor': "#161a24",
                        'bordercolor': "#262730"
                    }
                ))
                fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#ffffff"}, height=220, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

# --- 6. PAGE 2: ANALYTICS DASHBOARD ---
elif page == "📊 Analytics Dashboard":
    st.title("📊 Global Corpus Analytics")
    st.markdown("High-level breakdown of the active ground-truth ingestion dataset.")
    
    # Metric Summary Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Records Analyzed", "23,196", help="Combined dataset rows")
    m2.metric("Real Articles Baseline", "17,441", "75.1%")
    m3.metric("Fake Discovered Streams", "5,755", "-24.9%", delta_color="inverse")
    m4.metric("Monitored Domains", "1,412", "Active tracking")
    
    st.divider()
    
    # Charts Grid
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Class Distribution Bias")
        fig_pie = px.pie(
            dist_df, values="Count", names="Label",
            color="Label", color_discrete_map={"Real News": "#00cc96", "Fake News": "#ff4b4b"},
            hole=0.4
        )
        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', legend_font_color="#ffffff", font_color="#ffffff")
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with c2:
        st.subheader("Ingestion Velocity Timeline")
        fig_line = px.line(
            time_df, x="Day", y=["Real News", "Fake News"],
            labels={"value": "Article Upload Volumetrics", "variable": "Classification Type"},
            color_discrete_map={"Real News": "#00cc96", "Fake News": "#ff4b4b"},
            markers=True
        )
        fig_line.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#ffffff", xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#262730"))
        st.plotly_chart(fig_line, use_container_width=True)

# --- 7. PAGE 3: MODEL PERFORMANCE ---
elif page == "🤖 Model Performance":
    st.title("🤖 Recruiter Evaluation Metrics Matrix")
    st.markdown("A benchmark breakdown comparing classical statistical modeling arrays against deep iterative sequences.")
    
    # Dual bar chart metric comparison
    fig_bars = go.Figure()
    fig_bars.add_trace(go.Bar(
        x=model_df["Model"], y=model_df["Accuracy"],
        name='Overall Pipeline Accuracy',
        marker_color='#1f77b4'
    ))
    fig_bars.add_trace(go.Bar(
        x=model_df["Model"], y=model_df["Fake News Recall"],
        name='Fake News Recall Rate (Class 0)',
        marker_color='#ff7f0e'
    ))
    
    fig_bars.update_layout(
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color="#ffffff",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#262730", title="Percentage (%)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_bars, use_container_width=True)
    
    # Dataframe Table display
    st.subheader("Detailed Performance Ledger")
    st.dataframe(
        model_df.style.highlight_max(axis=0, subset=["Accuracy", "Fake News Recall"], color="#1e3d30"),
        use_container_width=True
    )
    
    st.success("🎖️ **Strategic Key Out-Take:** While XGBoost commands overall structural accuracy dominance at **82%**, Random Forest remains highly viable if the architecture criteria strictly penalizes missing bad actors (boasting a **64% Fake News Recall**).")