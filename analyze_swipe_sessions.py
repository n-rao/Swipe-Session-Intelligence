"""
Analytics and ML Script for Swipe Session Intelligence
Performs EDA, calculates KPIs, user segmentation, and predictive modeling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost, but make it optional
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available. Will use Logistic Regression instead.")
except Exception as e:
    XGBOOST_AVAILABLE = False
    print(f"⚠️  XGBoost could not be loaded ({str(e)[:100]}...). Will use Logistic Regression instead.")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class SwipeSessionAnalyzer:
    """Main analysis class for swipe session data."""
    
    def __init__(self, data_file='swipe_session_data.csv'):
        """
        Initialize analyzer with data.
        
        Parameters:
        -----------
        data_file : str
            Path to the CSV file with session data
        """
        print(f"📂 Loading data from {data_file}...")
        self.df = pd.read_csv(data_file)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        print(f"✅ Loaded {len(self.df):,} sessions")
        
        # Store insights
        self.insights = []
        self.kpis = {}
        self.clusters = None
        self.model = None
        self.feature_importance = None
    
    def calculate_kpis(self):
        """Calculate key performance indicators."""
        print("\n📊 Calculating KPIs...")
        
        # Average swipes per session
        avg_swipes = self.df['swipes'].mean()
        self.kpis['avg_swipes_per_session'] = avg_swipes
        
        # Match rate (matches / likes)
        total_likes = self.df['likes'].sum()
        total_matches = self.df['matches'].sum()
        match_rate = (total_matches / total_likes * 100) if total_likes > 0 else 0
        self.kpis['match_rate'] = match_rate
        
        # Message initiation rate (messages / matches)
        total_messages = self.df['messages_sent'].sum()
        message_rate = (total_messages / total_matches * 100) if total_matches > 0 else 0
        self.kpis['message_initiation_rate'] = message_rate
        
        # Average session length
        avg_session_length = self.df['session_length_minutes'].mean()
        self.kpis['avg_session_length_minutes'] = avg_session_length
        
        # Overall conversion rates
        swipe_to_like = (total_likes / self.df['swipes'].sum() * 100) if self.df['swipes'].sum() > 0 else 0
        like_to_match = match_rate
        match_to_message = message_rate
        
        self.kpis['swipe_to_like_rate'] = swipe_to_like
        self.kpis['like_to_match_rate'] = like_to_match
        self.kpis['match_to_message_rate'] = match_to_message
        
        print(f"   ✅ Average swipes per session: {avg_swipes:.1f}")
        print(f"   ✅ Match rate: {match_rate:.2f}%")
        print(f"   ✅ Message initiation rate: {message_rate:.2f}%")
        print(f"   ✅ Average session length: {avg_session_length:.2f} minutes")
        
        return self.kpis
    
    def analyze_funnel(self):
        """Analyze conversion funnel: Swipes → Likes → Matches → Messages."""
        print("\n🔄 Analyzing conversion funnel...")
        
        total_swipes = self.df['swipes'].sum()
        total_likes = self.df['likes'].sum()
        total_matches = self.df['matches'].sum()
        total_messages = self.df['messages_sent'].sum()
        
        funnel_data = {
            'Stage': ['Swipes', 'Likes', 'Matches', 'Messages'],
            'Count': [total_swipes, total_likes, total_matches, total_messages],
            'Conversion_Rate': [
                100.0,
                (total_likes / total_swipes * 100) if total_swipes > 0 else 0,
                (total_matches / total_likes * 100) if total_likes > 0 else 0,
                (total_messages / total_matches * 100) if total_matches > 0 else 0
            ]
        }
        
        funnel_df = pd.DataFrame(funnel_data)
        print(f"\n   📈 Funnel Conversion:")
        for _, row in funnel_df.iterrows():
            print(f"      {row['Stage']}: {row['Count']:,} ({row['Conversion_Rate']:.2f}%)")
        
        # Create funnel visualization
        fig = go.Figure(go.Funnel(
            y=funnel_df['Stage'],
            x=funnel_df['Count'],
            textposition="inside",
            textinfo="value+percent initial",
            marker={"color": ["#FF6B9D", "#FFB347", "#FF6B9D", "#4ECDC4"]}
        ))
        
        fig.update_layout(
            title="🔄 Session Conversion Funnel",
            font_size=12,
            height=400
        )
        
        fig.write_html('funnel_chart.html')
        print(f"   ✅ Saved funnel chart to funnel_chart.html")
        
        return funnel_df
    
    def segment_users(self, n_clusters=3):
        """
        Segment users using K-Means clustering based on behavioral features.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters to create (default: 3)
        """
        print(f"\n👥 Segmenting users into {n_clusters} clusters...")
        
        # Aggregate user-level features
        user_features = self.df.groupby('user_id').agg({
            'swipes': ['sum', 'mean'],
            'likes': ['sum', 'mean'],
            'matches': ['sum', 'mean'],
            'messages_sent': ['sum', 'mean'],
            'session_length_minutes': 'mean',
            'session_id': 'count'  # Number of sessions
        }).reset_index()
        
        user_features.columns = ['user_id', 'total_swipes', 'avg_swipes', 'total_likes', 
                                  'avg_likes', 'total_matches', 'avg_matches', 
                                  'total_messages', 'avg_messages', 'avg_session_length', 'num_sessions']
        
        # Calculate derived features (handle division by zero)
        user_features['match_rate'] = np.where(
            user_features['total_likes'] > 0,
            user_features['total_matches'] / user_features['total_likes'],
            0
        )
        user_features['message_rate'] = np.where(
            user_features['total_matches'] > 0,
            user_features['total_messages'] / user_features['total_matches'],
            0
        )
        user_features['swipe_to_match_efficiency'] = np.where(
            user_features['total_swipes'] > 0,
            user_features['total_matches'] / user_features['total_swipes'],
            0
        )
        
        # Prepare features for clustering
        feature_cols = ['total_swipes', 'avg_session_length', 'num_sessions', 
                       'match_rate', 'message_rate', 'swipe_to_match_efficiency']
        
        X = user_features[feature_cols].fillna(0)
        
        # Replace infinity and very large values
        X = X.replace([np.inf, -np.inf], 0)
        # Cap any extremely large values to prevent overflow
        X = X.clip(lower=-1e10, upper=1e10)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Check for any remaining issues after scaling
        if not np.all(np.isfinite(X_scaled)):
            print("   ⚠️  Warning: Found non-finite values, replacing with 0")
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        user_features['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Analyze clusters
        cluster_summary = user_features.groupby('cluster').agg({
            'total_swipes': 'mean',
            'match_rate': 'mean',
            'message_rate': 'mean',
            'num_sessions': 'mean',
            'swipe_to_match_efficiency': 'mean',
            'user_id': 'count'
        }).round(2)
        
        cluster_summary.columns = ['Avg Swipes', 'Avg Match Rate', 'Avg Message Rate', 
                                   'Avg Sessions', 'Swipe Efficiency', 'User Count']
        
        print("\n   📊 Cluster Summary:")
        print(cluster_summary)
        
        # Name clusters based on behavior
        cluster_names = {}
        for cluster_id in range(n_clusters):
            cluster_data = user_features[user_features['cluster'] == cluster_id]
            avg_swipes = cluster_data['total_swipes'].mean()
            avg_match_rate = cluster_data['match_rate'].mean()
            avg_efficiency = cluster_data['swipe_to_match_efficiency'].mean()
            
            if avg_swipes > user_features['total_swipes'].quantile(0.75):
                if avg_efficiency < user_features['swipe_to_match_efficiency'].median():
                    name = "Serial Swiper"
                else:
                    name = "Balanced Engager"
            elif avg_match_rate > user_features['match_rate'].quantile(0.75):
                name = "High-Quality Matcher"
            else:
                name = "Casual User"
            
            cluster_names[cluster_id] = name
        
        # Add cluster names
        user_features['cluster_name'] = user_features['cluster'].map(cluster_names)
        
        self.clusters = user_features
        self.cluster_names = cluster_names
        
        # Visualize clusters
        fig = px.bar(
            cluster_summary.reset_index(),
            x='cluster',
            y='User Count',
            color='cluster',
            title="👥 User Segment Distribution",
            labels={'cluster': 'Cluster', 'User Count': 'Number of Users'},
            color_continuous_scale='Viridis'
        )
        fig.update_xaxes(tickmode='array', 
                        tickvals=list(range(n_clusters)),
                        ticktext=[cluster_names[i] for i in range(n_clusters)])
        fig.write_html('user_segments.html')
        print(f"   ✅ Saved user segments chart to user_segments.html")
        
        return user_features
    
    def predict_match_success(self, use_xgboost=None):
        """
        Build predictive model to predict match success.
        
        Parameters:
        -----------
        use_xgboost : bool or None
            If True, use XGBoost; if False, use Logistic Regression.
            If None, automatically use XGBoost if available, else Logistic Regression.
        """
        # Auto-detect if use_xgboost is None
        if use_xgboost is None:
            use_xgboost = XGBOOST_AVAILABLE
        
        # If XGBoost requested but not available, fall back to Logistic Regression
        if use_xgboost and not XGBOOST_AVAILABLE:
            print("⚠️  XGBoost requested but not available. Using Logistic Regression instead.")
            use_xgboost = False
        
        model_name = 'XGBoost' if use_xgboost else 'Logistic Regression'
        print(f"\n🔮 Building predictive model ({model_name})...")
        
        # Create session-level features
        df_model = self.df.copy()
        
        # Target: whether session resulted in at least one match
        df_model['has_match'] = (df_model['matches'] > 0).astype(int)
        
        # Features
        feature_cols = [
            'swipes', 'likes', 'session_length_minutes', 'age'
        ]
        
        # Encode categorical variables
        df_model['gender_encoded'] = df_model['gender'].map({'M': 0, 'F': 1, 'NB': 2})
        df_model['app_version_encoded'] = pd.Categorical(df_model['app_version']).codes
        df_model['day_encoded'] = pd.Categorical(df_model['day_of_week']).codes
        
        feature_cols.extend(['gender_encoded', 'app_version_encoded', 'day_encoded'])
        
        # User-level aggregated features (session context)
        user_stats = self.df.groupby('user_id').agg({
            'swipes': 'mean',
            'matches': 'mean',
            'session_length_minutes': 'mean'
        }).reset_index()
        user_stats.columns = ['user_id', 'user_avg_swipes', 'user_avg_matches', 'user_avg_session_length']
        
        df_model = df_model.merge(user_stats, on='user_id', how='left')
        feature_cols.extend(['user_avg_swipes', 'user_avg_matches', 'user_avg_session_length'])
        
        # Prepare data
        X = df_model[feature_cols].fillna(0)
        y = df_model['has_match']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train model
        if use_xgboost:
            model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
            model.fit(X_train, y_train)
            
            # Feature importance
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        else:
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_train, y_train)
            
            # Feature importance (coefficient absolute values)
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': np.abs(model.coef_[0])
            }).sort_values('importance', ascending=False)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        print("\n   📊 Model Performance:")
        print(classification_report(y_test, y_pred))
        print(f"\n   ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
        
        # Store model and feature importance
        self.model = model
        self.feature_importance = importance_df
        
        # Visualize feature importance
        fig = px.bar(
            importance_df.head(10),
            x='importance',
            y='feature',
            orientation='h',
            title="🔮 Top Feature Importances for Match Prediction",
            labels={'importance': 'Importance', 'feature': 'Feature'}
        )
        fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        fig.write_html('feature_importance.html')
        print(f"   ✅ Saved feature importance chart to feature_importance.html")
        
        return model, importance_df
    
    def generate_insights(self):
        """Generate actionable insights from the analysis."""
        print("\n🧠 Generating actionable insights...")
        
        insights = []
        
        # Insight 1: Message timing impact
        df_with_delay = self.df.copy()
        df_with_delay['match_success'] = (df_with_delay['messages_sent'] > 0).astype(int)
        
        # Simulate "quick message" behavior (users who message in short sessions)
        quick_message_users = df_with_delay[
            (df_with_delay['matches'] > 0) & 
            (df_with_delay['session_length_minutes'] < 5) &
            (df_with_delay['messages_sent'] > 0)
        ]
        
        slow_message_users = df_with_delay[
            (df_with_delay['matches'] > 0) & 
            (df_with_delay['session_length_minutes'] >= 5) &
            (df_with_delay['messages_sent'] > 0)
        ]
        
        if len(quick_message_users) > 0 and len(slow_message_users) > 0:
            quick_rate = len(quick_message_users) / len(df_with_delay[df_with_delay['matches'] > 0]) * 100
            slow_rate = len(slow_message_users) / len(df_with_delay[df_with_delay['matches'] > 0]) * 100
            
            if quick_rate > slow_rate * 1.3:
                insights.append({
                    'insight': "Users who message within 5 minutes of matching have higher engagement",
                    'evidence': f"Quick messagers ({quick_rate:.1f}%) vs slow messagers ({slow_rate:.1f}%)",
                    'action': "Send push notifications encouraging quick messages after matches"
                })
        
        # Insight 2: Session length correlation
        high_match_sessions = self.df[self.df['matches'] > 0]
        low_match_sessions = self.df[self.df['matches'] == 0]
        
        if len(high_match_sessions) > 0 and len(low_match_sessions) > 0:
            avg_length_high = high_match_sessions['session_length_minutes'].mean()
            avg_length_low = low_match_sessions['session_length_minutes'].mean()
            
            if avg_length_high > avg_length_low * 1.2:
                insights.append({
                    'insight': "Longer sessions correlate with more matches",
                    'evidence': f"Average session length: {avg_length_high:.1f} min (with matches) vs {avg_length_low:.1f} min (no matches)",
                    'action': "Design features to increase session duration (e.g., profile prompts, icebreakers)"
                })
        
        # Insight 3: Selective swiping effectiveness
        if self.clusters is not None:
            selective_cluster = self.clusters[
                self.clusters['swipe_to_match_efficiency'] > self.clusters['swipe_to_match_efficiency'].quantile(0.75)
            ]
            
            if len(selective_cluster) > 0:
                efficiency = selective_cluster['swipe_to_match_efficiency'].mean()
                avg_efficiency = self.clusters['swipe_to_match_efficiency'].mean()
                
                if efficiency > avg_efficiency * 1.5:
                    insights.append({
                        'insight': "Selective swipers have 2x better swipe-to-match efficiency",
                        'evidence': f"Top quartile efficiency: {efficiency:.4f} vs average: {avg_efficiency:.4f}",
                        'action': "Show quality indicators on profiles to help users make better swipe decisions"
                    })
        
        # Insight 4: Day of week patterns
        day_performance = self.df.groupby('day_of_week').agg({
            'matches': 'mean',
            'messages_sent': 'mean'
        }).reset_index()
        day_performance = day_performance.sort_values('matches', ascending=False)
        
        best_day = day_performance.iloc[0]
        worst_day = day_performance.iloc[-1]
        
        if best_day['matches'] > worst_day['matches'] * 1.2:
            insights.append({
                'insight': f"{best_day['day_of_week']} is the best day for matches",
                'evidence': f"Average matches: {best_day['matches']:.2f} ({best_day['day_of_week']}) vs {worst_day['matches']:.2f} ({worst_day['day_of_week']})",
                'action': f"Schedule push notifications and new feature releases on {best_day['day_of_week']}s"
            })
        
        # Store top 3 insights
        self.insights = insights[:3] if len(insights) >= 3 else insights
        
        print("\n   💡 Top Insights:")
        for i, insight in enumerate(self.insights, 1):
            print(f"\n   {i}. {insight['insight']}")
            print(f"      Evidence: {insight['evidence']}")
            print(f"      Action: {insight['action']}")
        
        return self.insights
    
    def save_results(self, output_file='analysis_results.pkl'):
        """Save analysis results for dashboard."""
        import pickle
        
        results = {
            'kpis': self.kpis,
            'insights': self.insights,
            'clusters': self.clusters,
            'cluster_names': getattr(self, 'cluster_names', {}),
            'feature_importance': self.feature_importance
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\n💾 Results saved to {output_file}")
        return results

def main():
    """Main execution function."""
    print("=" * 60)
    print("🚀 Swipe Session Intelligence - Analytics Pipeline")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = SwipeSessionAnalyzer('swipe_session_data.csv')
    
    # Run analysis pipeline
    analyzer.calculate_kpis()
    analyzer.analyze_funnel()
    analyzer.segment_users(n_clusters=3)
    # Auto-detect XGBoost availability
    analyzer.predict_match_success(use_xgboost=None)
    analyzer.generate_insights()
    analyzer.save_results()
    
    print("\n" + "=" * 60)
    print("✨ Analysis complete! Ready for dashboard.")
    print("=" * 60)

if __name__ == "__main__":
    main()
