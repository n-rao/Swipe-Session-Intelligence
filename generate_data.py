"""
Data Generation Script for Swipe Session Intelligence
Simulates realistic user behavioral analytics for a dating app.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_synthetic_data(n_users=10000, days=30, output_file='swipe_session_data.csv'):
    """
    Generate synthetic event-level data for dating app behavioral analytics.
    
    Parameters:
    -----------
    n_users : int
        Number of unique users to generate (default: 10000)
    days : int
        Number of days to simulate (default: 30)
    output_file : str
        Output CSV filename (default: 'swipe_session_data.csv')
    
    Returns:
    --------
    pd.DataFrame
        Generated dataset with event-level behavioral data
    """
    np.random.seed(42)
    random.seed(42)
    
    # User attributes
    user_ids = list(range(1, n_users + 1))
    genders = ['M', 'F', 'NB']
    app_versions = ['2.1.0', '2.2.0', '2.3.0']
    locations = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 
                 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
    
    # Generate user base with attributes
    user_profiles = pd.DataFrame({
        'user_id': user_ids,
        'gender': np.random.choice(genders, n_users, p=[0.45, 0.45, 0.10]),
        'age': np.random.normal(28, 8, n_users).astype(int).clip(18, 65),
        'location': np.random.choice(locations, n_users),
        'app_version': np.random.choice(app_versions, n_users, p=[0.2, 0.3, 0.5]),
        # User engagement profile (affects behavior)
        'engagement_level': np.random.choice(['low', 'medium', 'high'], n_users, p=[0.3, 0.5, 0.2])
    })
    
    # Date range
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(days)]
    
    # Generate sessions
    sessions = []
    session_id_counter = 1
    
    for date in dates:
        day_of_week = date.strftime('%A')
        
        # Activity varies by day (weekends more active)
        if day_of_week in ['Saturday', 'Sunday']:
            active_users = np.random.choice(user_ids, size=int(n_users * 0.35), replace=False)
        else:
            active_users = np.random.choice(user_ids, size=int(n_users * 0.25), replace=False)
        
        for user_id in active_users:
            user_profile = user_profiles[user_profiles['user_id'] == user_id].iloc[0]
            engagement = user_profile['engagement_level']
            
            # Engagement level affects session frequency and behavior
            if engagement == 'high':
                num_sessions = np.random.poisson(3)  # High engagers use app more
            elif engagement == 'medium':
                num_sessions = np.random.poisson(1.5)
            else:
                num_sessions = np.random.poisson(0.7)
            
            for _ in range(min(num_sessions, 5)):  # Cap at 5 sessions per day
                # Session time (peaks in evening)
                hour = np.clip(np.random.normal(20, 3), 6, 23)
                minute = np.random.randint(0, 60)
                timestamp = date.replace(hour=int(hour), minute=minute, second=np.random.randint(0, 60))
                
                # Session length varies by engagement
                if engagement == 'high':
                    base_length = np.random.normal(15, 5)
                elif engagement == 'medium':
                    base_length = np.random.normal(8, 3)
                else:
                    base_length = np.random.normal(4, 2)
                
                session_length = max(1, base_length)  # Minimum 1 minute
                
                # Swipes (core activity)
                if engagement == 'high':
                    swipes = np.random.poisson(50)
                elif engagement == 'medium':
                    swipes = np.random.poisson(25)
                else:
                    swipes = np.random.poisson(10)
                
                swipes = max(1, swipes)  # At least 1 swipe
                
                # Likes (subset of swipes, with realistic ratio)
                # More swipes → higher absolute likes, but ratio decreases slightly
                like_ratio = 0.15 + (np.random.random() - 0.5) * 0.1  # 10-20% of swipes
                likes = max(0, int(swipes * like_ratio))
                
                # Matches (mutual likes - realistic correlation)
                # Match rate depends on swipe volume and quality
                if swipes > 40:
                    match_rate = 0.08  # Serial swipers have lower match rate
                elif swipes > 20:
                    match_rate = 0.12  # Balanced users
                else:
                    match_rate = 0.15  # Selective swipers
                
                # Add engagement modifier
                match_rate *= (1.2 if engagement == 'high' else 1.0 if engagement == 'medium' else 0.9)
                
                matches = max(0, int(likes * match_rate))
                
                # Messages (only if there's a match)
                if matches > 0:
                    # Higher engagement → more likely to message
                    message_rate = 0.4 + (0.3 if engagement == 'high' else 0.1 if engagement == 'medium' else 0.0)
                    messages_sent = np.random.binomial(matches, message_rate)
                else:
                    messages_sent = 0
                
                # Correlation: short sessions → fewer matches (less thoughtful swiping)
                if session_length < 3:
                    matches = int(matches * 0.6)
                    messages_sent = int(messages_sent * 0.7)
                
                # Add some randomness for realism
                matches = max(0, matches + np.random.randint(-1, 2))
                messages_sent = max(0, min(messages_sent, matches))
                
                session = {
                    'user_id': user_id,
                    'session_id': session_id_counter,
                    'timestamp': timestamp,
                    'swipes': swipes,
                    'likes': likes,
                    'matches': matches,
                    'messages_sent': messages_sent,
                    'session_length_minutes': round(session_length, 2),
                    'app_version': user_profile['app_version'],
                    'gender': user_profile['gender'],
                    'age': user_profile['age'],
                    'location': user_profile['location'],
                    'day_of_week': day_of_week
                }
                
                sessions.append(session)
                session_id_counter += 1
    
    # Create DataFrame
    df = pd.DataFrame(sessions)
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"✅ Generated {len(df):,} sessions for {df['user_id'].nunique():,} users")
    print(f"✅ Data saved to {output_file}")
    print(f"\n📊 Summary Statistics:")
    print(f"   - Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   - Total swipes: {df['swipes'].sum():,}")
    print(f"   - Total matches: {df['matches'].sum():,}")
    print(f"   - Match rate: {(df['matches'].sum() / df['likes'].sum() * 100):.2f}%")
    print(f"   - Message initiation rate: {(df['messages_sent'].sum() / df['matches'].sum() * 100):.2f}%")
    
    return df

if __name__ == "__main__":
    print("🚀 Generating synthetic swipe session data...")
    df = generate_synthetic_data(n_users=10000, days=30)
    print("\n✨ Data generation complete!")
