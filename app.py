"""
Streamlit Dashboard for Swipe Session Intelligence
Interactive dashboard for exploring dating app behavioral analytics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Swipe Session Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Custom CSS
st.markdown("""
    <style>
    /* Main styling */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    /* KPI Card styling */
    .kpi-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        transition: transform 0.3s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.15);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: white;
    }
    
    .sidebar .sidebar-content {
        background: white;
    }
    
    /* Ensure sidebar widgets (Location/Day) are visible on white */
    [data-testid="stSidebar"] * {
        color: #2d3748 !important;
    }
    [data-testid="stSidebar"] label {
        font-weight: 600;
    }
    [data-testid="stSidebar"] [role="combobox"] {
        background: #f7f9fc !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] [role="listbox"] {
        background: #ffffff !important;
        color: #2d3748 !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Insight cards */
    .insight-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Section headers */
    h2 {
        color: #667eea;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    
    h3 {
        color: #764ba2;
        margin-top: 1.5rem;
    }
    
    /* Professional colors */
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --accent-color: #f093fb;
        --success-color: #4facfe;
        --text-color: #2d3748;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(data_file='swipe_session_data.csv'):
    """Load session data."""
    try:
        df = pd.read_csv(data_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except FileNotFoundError:
        st.error(f"❌ Data file '{data_file}' not found. Please run `python generate_data.py` first.")
        st.stop()

@st.cache_data
def load_results(results_file='analysis_results.pkl'):
    """Load analysis results."""
    try:
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        return results
    except FileNotFoundError:
        st.warning(f"⚠️ Results file '{results_file}' not found. Running analysis...")
        return None

def calculate_realtime_kpis(df_filtered):
    """Calculate KPIs from filtered data."""
    total_swipes = df_filtered['swipes'].sum()
    total_likes = df_filtered['likes'].sum()
    total_matches = df_filtered['matches'].sum()
    total_messages = df_filtered['messages_sent'].sum()
    
    kpis = {
        'avg_swipes': df_filtered['swipes'].mean(),
        'match_rate': (total_matches / total_likes * 100) if total_likes > 0 else 0,
        'message_rate': (total_messages / total_matches * 100) if total_matches > 0 else 0,
        'avg_session_length': df_filtered['session_length_minutes'].mean(),
        'total_sessions': len(df_filtered),
        'total_users': df_filtered['user_id'].nunique(),
        'swipe_to_like_rate': (total_likes / total_swipes * 100) if total_swipes > 0 else 0
    }
    return kpis

def create_professional_chart_style(fig):
    """Apply professional styling to charts."""
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif", size=12, color="#2d3748"),
        title_font=dict(size=18, color="#667eea", family="Arial, sans-serif"),
        hovermode='closest',
        legend=dict(
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1
        )
    )
    return fig

def main():
    """Main dashboard function."""
    # Professional Header
    st.markdown("""
        <div class="main-header">
            <h1>📊 Swipe Session Intelligence</h1>
            <p>Advanced Behavioral Analytics Dashboard</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    results = load_results()
    
    # Sidebar with professional styling
    with st.sidebar:
        st.markdown("### 🔍 Filter Options")
        st.markdown("---")
        
        # Location filter
        locations = ['All'] + sorted(df['location'].unique().tolist())
        selected_location = st.selectbox(
            "📍 Location",
            locations,
            help="Filter data by user location"
        )
        
        # Day of week filter
        days = ['All'] + sorted(df['day_of_week'].unique().tolist())
        selected_day = st.selectbox(
            "📅 Day of Week",
            days,
            help="Filter data by day of week"
        )
        
        st.markdown("---")
        
        # Data summary in sidebar
        st.markdown("### 📈 Data Summary")
        st.metric("Total Users", f"{df['user_id'].nunique():,}")
        st.metric("Total Sessions", f"{len(df):,}")
        date_range = f"{df['timestamp'].min().strftime('%b %d, %Y')} to {df['timestamp'].max().strftime('%b %d, %Y')}"
        st.caption(f"📅 {date_range}")
    
    # Apply filters
    df_filtered = df.copy()
    if selected_location != 'All':
        df_filtered = df_filtered[df_filtered['location'] == selected_location]
    if selected_day != 'All':
        df_filtered = df_filtered[df_filtered['day_of_week'] == selected_day]
    
    # Calculate filtered KPIs
    kpis_filtered = calculate_realtime_kpis(df_filtered)
    
    # Main tabs with professional labels
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview", 
        "👥 Segmentation", 
        "🔮 Predictive Analytics", 
        "💡 Recommendations"
    ])
    
    # Tab 1: Overview KPIs
    with tab1:
        st.markdown("## Key Performance Indicators")
        st.markdown("---")
        
        # KPI Cards in a professional grid
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                label="Match Rate",
                value=f"{kpis_filtered['match_rate']:.2f}%",
                help="Percentage of likes that result in matches"
            )
        
        with col2:
            st.metric(
                label="Message Rate",
                value=f"{kpis_filtered['message_rate']:.2f}%",
                help="Percentage of matches that lead to messages"
            )
        
        with col3:
            st.metric(
                label="Avg Session",
                value=f"{kpis_filtered['avg_session_length']:.1f}m",
                help="Average session duration in minutes"
            )
        
        with col4:
            st.metric(
                label="Avg Swipes",
                value=f"{kpis_filtered['avg_swipes']:.1f}",
                help="Average swipes per session"
            )
        
        with col5:
            st.metric(
                label="Total Sessions",
                value=f"{kpis_filtered['total_sessions']:,}",
                help="Total number of sessions in filtered data"
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Charts in a professional layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Conversion Funnel")
            total_swipes = df_filtered['swipes'].sum()
            total_likes = df_filtered['likes'].sum()
            total_matches = df_filtered['matches'].sum()
            total_messages = df_filtered['messages_sent'].sum()
            
            funnel_data = pd.DataFrame({
                'Stage': ['Swipes', 'Likes', 'Matches', 'Messages'],
                'Count': [total_swipes, total_likes, total_matches, total_messages]
            })
            
            fig_funnel = go.Figure(go.Funnel(
                y=funnel_data['Stage'],
                x=funnel_data['Count'],
                textposition="inside",
                textinfo="value+percent initial",
                marker={"color": ["#667eea", "#764ba2", "#f093fb", "#4facfe"]},
                connector={"line": {"color": "#667eea", "dash": "dot", "width": 3}}
            ))
            
            fig_funnel = create_professional_chart_style(fig_funnel)
            fig_funnel.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_funnel, use_container_width=True)
            
            # Conversion rates
            st.markdown("#### Conversion Rates")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                swipe_to_like = (total_likes / total_swipes * 100) if total_swipes > 0 else 0
                st.metric("Swipe → Like", f"{swipe_to_like:.1f}%")
            with col_b:
                like_to_match = kpis_filtered['match_rate']
                st.metric("Like → Match", f"{like_to_match:.1f}%")
            with col_c:
                match_to_msg = kpis_filtered['message_rate']
                st.metric("Match → Message", f"{match_to_msg:.1f}%")
        
        with col2:
            st.markdown("### Daily Activity Trends")
            daily_metrics = df_filtered.groupby(df_filtered['timestamp'].dt.date).agg({
                'swipes': 'sum',
                'matches': 'sum',
                'messages_sent': 'sum'
            }).reset_index()
            daily_metrics['timestamp'] = pd.to_datetime(daily_metrics['timestamp'])
            
            fig_time = px.line(
                daily_metrics,
                x='timestamp',
                y=['swipes', 'matches', 'messages_sent'],
                labels={'value': 'Count', 'timestamp': 'Date', 'variable': 'Metric'},
                color_discrete_map={
                    'swipes': '#667eea',
                    'matches': '#764ba2',
                    'messages_sent': '#f093fb'
                }
            )
            fig_time = create_professional_chart_style(fig_time)
            st.plotly_chart(fig_time, use_container_width=True)
            
            # Activity summary
            st.markdown("#### Activity Summary")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Avg Daily Swipes", f"{daily_metrics['swipes'].mean():.0f}")
            with col_b:
                st.metric("Avg Daily Matches", f"{daily_metrics['matches'].mean():.0f}")
            with col_c:
                st.metric("Avg Daily Messages", f"{daily_metrics['messages_sent'].mean():.0f}")
        
        st.markdown("---")
        
        # Additional charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Swipes Distribution")
            fig_swipes = px.histogram(
                df_filtered,
                x='swipes',
                nbins=30,
                labels={'swipes': 'Number of Swipes', 'count': 'Number of Sessions'},
                color_discrete_sequence=['#667eea']
            )
            fig_swipes = create_professional_chart_style(fig_swipes)
            st.plotly_chart(fig_swipes, use_container_width=True)
        
        with col2:
            st.markdown("### Session Length Distribution")
            fig_session = px.histogram(
                df_filtered,
                x='session_length_minutes',
                nbins=30,
                labels={'session_length_minutes': 'Session Length (minutes)', 'count': 'Number of Sessions'},
                color_discrete_sequence=['#764ba2']
            )
            fig_session = create_professional_chart_style(fig_session)
            st.plotly_chart(fig_session, use_container_width=True)
    
    # Tab 2: User Segments
    with tab2:
        st.markdown("## User Behavioral Segmentation")
        st.markdown("---")
        
        if results and 'clusters' in results and results['clusters'] is not None:
            clusters_df = results['clusters']
            cluster_names = results.get('cluster_names', {})
            
            # Segment overview
            st.markdown("### Segment Overview")
            segment_counts = clusters_df['cluster_name'].value_counts().reset_index()
            segment_counts.columns = ['Segment', 'User Count']
            segment_counts['Percentage'] = (segment_counts['User Count'] / segment_counts['User Count'].sum() * 100).round(1)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig_segments = px.bar(
                    segment_counts,
                    x='Segment',
                    y='User Count',
                    text='User Count',
                    labels={'User Count': 'Number of Users', 'Segment': 'User Segment'},
                    color='User Count',
                    color_continuous_scale='Viridis'
                )
                fig_segments = create_professional_chart_style(fig_segments)
                fig_segments.update_traces(texttemplate='%{text:,}', textposition='outside')
                st.plotly_chart(fig_segments, use_container_width=True)
            
            with col2:
                st.markdown("#### Distribution")
                for _, row in segment_counts.iterrows():
                    st.metric(row['Segment'], f"{row['User Count']:,}", f"{row['Percentage']}%")
            
            st.markdown("---")
            
            # Segment characteristics table
            st.markdown("### Segment Characteristics")
            segment_stats = clusters_df.groupby('cluster_name').agg({
                'total_swipes': 'mean',
                'match_rate': 'mean',
                'message_rate': 'mean',
                'num_sessions': 'mean',
                'swipe_to_match_efficiency': 'mean'
            }).round(3)
            
            segment_stats.columns = ['Avg Swipes', 'Match Rate', 'Message Rate', 'Avg Sessions', 'Swipe Efficiency']
            segment_stats_display = segment_stats.style.background_gradient(subset=['Match Rate', 'Message Rate'], cmap='YlOrRd')
            st.dataframe(segment_stats_display, use_container_width=True, height=200)
            
            # Behavioral comparison chart
            st.markdown("### Behavioral Comparison")
            comparison_cols = ['total_swipes', 'match_rate', 'message_rate', 'num_sessions']
            comparison_df = clusters_df.groupby('cluster_name')[comparison_cols].mean().reset_index()
            
            fig_comparison = px.bar(
                comparison_df,
                x='cluster_name',
                y=comparison_cols,
                barmode='group',
                labels={'value': 'Metric Value', 'cluster_name': 'Segment', 'variable': 'Metric'},
                color_discrete_sequence=['#667eea', '#764ba2', '#f093fb', '#4facfe']
            )
            fig_comparison = create_professional_chart_style(fig_comparison)
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Segment descriptions
            st.markdown("### Segment Descriptions")
            segment_descriptions = {
                "Serial Swiper": "High-volume users who swipe frequently but with lower match efficiency. Focus on improving quality over quantity.",
                "Balanced Engager": "Moderate activity with consistent engagement across all metrics. Ideal user segment.",
                "High-Quality Matcher": "Selective users achieving high match rates with fewer swipes. Quality-focused approach.",
                "Casual User": "Low overall engagement with infrequent app usage. Opportunity for re-engagement campaigns."
            }
            
            cols = st.columns(2)
            for idx, (segment, desc) in enumerate(segment_descriptions.items()):
                if segment in clusters_df['cluster_name'].values:
                    with cols[idx % 2]:
                        with st.container():
                            st.markdown(f"**{segment}**")
                            st.caption(desc)
        else:
            st.warning("⚠️ User segment data not available. Please run `python analyze_swipe_sessions.py` first.")
    
    # Tab 3: Predictive Model
    with tab3:
        st.markdown("## Predictive Model Insights")
        st.markdown("---")
        
        if results and 'feature_importance' in results and results['feature_importance'] is not None:
            importance_df = results['feature_importance']
            
            st.markdown("### Feature Importance Analysis")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig_importance = px.bar(
                    importance_df.head(10),
                    x='importance',
                    y='feature',
                    orientation='h',
                    labels={'importance': 'Importance Score', 'feature': 'Feature'},
                    color='importance',
                    color_continuous_scale='Blues'
                )
                fig_importance = create_professional_chart_style(fig_importance)
                fig_importance.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_importance, use_container_width=True)
            
            with col2:
                st.markdown("#### Top Predictors")
                top_features = importance_df.head(5)
                for idx, row in top_features.iterrows():
                    with st.container():
                        st.markdown(f"**{row['feature']}**")
                        max_imp = float(top_features.iloc[0]['importance']) if float(top_features.iloc[0]['importance']) != 0 else 1.0
                        ratio = float(row['importance']) / max_imp
                        ratio = max(0.0, min(1.0, ratio))
                        st.progress(int(ratio * 100))
                        st.caption(f"Score: {float(row['importance']):.4f}")
                
                st.markdown("---")
                st.markdown("#### Model Performance")
                st.info("""
                **Model Type**: XGBoost / Logistic Regression
                
                **Primary Drivers**:
                - Session behavior patterns
                - User historical metrics
                - Contextual factors
                """)
            
            # Model insights card
            st.markdown("---")
            st.markdown("### Model Insights")
            
            insight_cols = st.columns(3)
            with insight_cols[0]:
                st.markdown("#### 🎯 Session Behavior")
                st.write("Swipe patterns and session length are strong predictors of match success.")
            with insight_cols[1]:
                st.markdown("#### 📊 User History")
                st.write("Past engagement metrics provide context for current session performance.")
            with insight_cols[2]:
                st.markdown("#### 🌍 Contextual Factors")
                st.write("Day of week and app version influence user engagement patterns.")
        else:
            st.warning("⚠️ Model results not available. Please run `python analyze_swipe_sessions.py` first.")
    
    # Tab 4: Recommendations
    with tab4:
        st.markdown("## Actionable Insights & Recommendations")
        st.markdown("---")
        
        # Top Insights Section
        if results and 'insights' in results and len(results['insights']) > 0:
            insights = results['insights']
            
            st.markdown("### Key Insights from Analysis")
            
            for i, insight in enumerate(insights, 1):
                with st.expander(f"💡 Insight #{i}: {insight['insight']}", expanded=(i==1)):
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.markdown("**📊 Evidence**")
                        st.info(insight['evidence'])
                    with col2:
                        st.markdown("**🎯 Recommended Action**")
                        st.success(insight['action'])
        else:
            st.warning("⚠️ Insights not available. Please run `python analyze_swipe_sessions.py` first.")
        
        st.markdown("---")
        
        # Real-time recommendations
        st.markdown("### Real-Time Data-Driven Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🕐 Peak Activity Analysis")
            if len(df_filtered) > 0:
                hourly_activity = df_filtered.groupby(df_filtered['timestamp'].dt.hour).agg({
                    'matches': 'sum',
                    'messages_sent': 'sum',
                    'swipes': 'sum'
                }).reset_index()
                hourly_activity.columns = ['Hour', 'Matches', 'Messages', 'Swipes']
                
                if len(hourly_activity) > 0:
                    peak_hour_row = hourly_activity.loc[hourly_activity['Matches'].idxmax()]
                    peak_hour = int(peak_hour_row['Hour'])
                    
                    fig_hourly = px.line(
                        hourly_activity,
                        x='Hour',
                        y=['Matches', 'Messages'],
                        labels={'value': 'Count', 'Hour': 'Hour of Day', 'variable': 'Metric'},
                        color_discrete_map={'Matches': '#667eea', 'Messages': '#764ba2'}
                    )
                    fig_hourly = create_professional_chart_style(fig_hourly)
                    fig_hourly.update_layout(paper_bgcolor='white', plot_bgcolor='white')
                    st.plotly_chart(fig_hourly, use_container_width=True)
                    
                    st.success(f"**Peak Hour**: {peak_hour}:00")
                    st.caption(f"💡 Schedule push notifications at {peak_hour}:00 for maximum engagement")
        
        with col2:
            st.markdown("#### 📅 Day-of-Week Performance")
            if len(df_filtered) > 0:
                day_performance = df_filtered.groupby('day_of_week').agg({
                    'matches': 'mean',
                    'messages_sent': 'mean',
                    'swipes': 'mean'
                }).reset_index()
                day_performance = day_performance.sort_values('matches', ascending=False)
                
                if len(day_performance) > 0:
                    best_day = day_performance.iloc[0]
                    
                    fig_days = px.bar(
                        day_performance,
                        x='day_of_week',
                        y=['matches', 'messages_sent'],
                        barmode='group',
                        labels={'value': 'Average Count', 'day_of_week': 'Day of Week', 'variable': 'Metric'},
                        color_discrete_map={'matches': '#667eea', 'messages_sent': '#764ba2'}
                    )
                    fig_days = create_professional_chart_style(fig_days)
                    fig_days.update_layout(paper_bgcolor='white', plot_bgcolor='white')
                    st.plotly_chart(fig_days, use_container_width=True)
                    
                    st.success(f"**Best Performing Day**: {best_day['day_of_week']}")
                    st.caption(f"💡 Optimize content and features for {best_day['day_of_week']} users")
        
        # Location-specific recommendation
        if selected_location != 'All' and len(df_filtered) > 0:
            st.markdown("---")
            st.markdown("#### 📍 Location-Specific Insights")
            location_data = df_filtered[df_filtered['location'] == selected_location]
            if len(location_data) > 0:
                total_likes = location_data['likes'].sum()
                total_matches = location_data['matches'].sum()
                total_messages = location_data['messages_sent'].sum()
                match_rate = (total_matches / total_likes * 100) if total_likes > 0 else 0
                message_rate = (total_messages / total_matches * 100) if total_matches > 0 else 0
                
                loc_cols = st.columns(4)
                with loc_cols[0]:
                    st.metric("Match Rate", f"{match_rate:.2f}%", 
                             delta=f"{(match_rate - kpis_filtered['match_rate']):.2f}% vs avg" if match_rate else None)
                with loc_cols[1]:
                    st.metric("Message Rate", f"{message_rate:.2f}%",
                             delta=f"{(message_rate - kpis_filtered['message_rate']):.2f}% vs avg" if message_rate else None)
                with loc_cols[2]:
                    st.metric("Total Sessions", f"{len(location_data):,}")
                with loc_cols[3]:
                    st.metric("Unique Users", f"{location_data['user_id'].nunique():,}")
                
                performance_status = "above" if match_rate > kpis_filtered['match_rate'] else "below"
                st.info(f"💡 **{selected_location}** shows {performance_status} average performance. Consider targeted engagement campaigns.")
        
        # Quick wins
        if len(df_filtered) > 0:
            st.markdown("---")
            st.markdown("### 📈 Quick Wins & Opportunities")
            
            quick_wins = []
            
            # Check session length
            avg_session = df_filtered['session_length_minutes'].mean()
            if avg_session < 10:
                quick_wins.append(("Session Duration", "Short sessions detected - implement features to increase engagement duration", "🕐"))
            
            # Check message rate
            if kpis_filtered['message_rate'] < 40:
                quick_wins.append(("Message Initiation", "Low message rate - consider match prompts or conversation starters", "💬"))
            
            # Check swipe efficiency
            total_swipes = df_filtered['swipes'].sum()
            total_matches = df_filtered['matches'].sum()
            swipe_efficiency = (total_matches / total_swipes * 100) if total_swipes > 0 else 0
            if swipe_efficiency < 2:
                quick_wins.append(("Swipe Quality", "Low swipe efficiency - show quality indicators to improve decision-making", "🎯"))
            
            if quick_wins:
                for icon, title, description in quick_wins:
                    with st.container():
                        st.markdown(f"**{icon} {title}**")
                        st.caption(description)
                        st.markdown("---")
            else:
                st.success("✨ Great performance across all metrics! Keep up the momentum.")

if __name__ == "__main__":
    main()
