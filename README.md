# 💘 Swipe Session Intelligence

A comprehensive end-to-end project simulating user behavioral analytics for a dating app, inspired by Bumble. This project demonstrates product data science workflows including data simulation, exploratory analysis, KPI calculation, user segmentation, predictive modeling, and interactive dashboard creation.

## 🎯 Project Overview

This project simulates a dating app's user behavioral analytics system with:
- **Synthetic data generation** for ~10,000 users over 30 days
- **Exploratory data analysis** with conversion funnel tracking
- **KPI calculation** (match rates, message rates, session metrics)
- **User segmentation** using K-Means clustering
- **Predictive modeling** (XGBoost/Logistic Regression) for match prediction
- **Interactive Streamlit dashboard** with visualizations and insights

## 📋 Features

### Data Simulation
- Event-level data with realistic correlations
- User attributes (gender, age, location, app version)
- Behavioral patterns (swipes → likes → matches → messages)
- Day-of-week and engagement level variations

### Analytics & ML
- **Conversion Funnel Analysis**: Track user journey from swipes to messages
- **KPI Dashboard**: Real-time metrics calculation
- **User Segmentation**: Identify behavioral clusters (Serial Swiper, Balanced Engager, High-Quality Matcher, Casual User)
- **Predictive Modeling**: Predict match success using XGBoost
- **Feature Importance**: Identify key drivers of engagement

### Interactive Dashboard
- 📊 Overview KPIs with real-time filtering
- 👥 User segment visualization and analysis
- 🔮 Predictive model insights and feature importance
- 🧠 Actionable recommendations and insights

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. **Clone the repository** (or navigate to the project directory):
   ```bash
   cd Swipe-Session-Intelligence
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

#### Step 1: Generate Synthetic Data
```bash
python generate_data.py
```
This will create `swipe_session_data.csv` with ~10,000 users and 30 days of session data.

#### Step 2: Run Analytics & ML Pipeline
```bash
python analyze_swipe_sessions.py
```
This will:
- Calculate KPIs
- Perform funnel analysis
- Create user segments
- Train predictive models
- Generate insights
- Save results to `analysis_results.pkl`

#### Step 3: Launch Dashboard
```bash
streamlit run app.py
```
The dashboard will open in your default web browser at `http://localhost:8501`

## 📁 Project Structure

```
Swipe-Session-Intelligence/
├── generate_data.py              # Synthetic data generation
├── analyze_swipe_sessions.py     # Analytics & ML pipeline
├── app.py                        # Streamlit dashboard
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── swipe_session_data.csv        # Generated data (after Step 1)
└── analysis_results.pkl          # Analysis results (after Step 2)
```

## 📊 Data Schema

The generated dataset includes the following columns:

| Column | Description |
|--------|-------------|
| `user_id` | Unique user identifier |
| `session_id` | Unique session identifier |
| `timestamp` | Session timestamp |
| `swipes` | Number of swipes in session |
| `likes` | Number of likes given |
| `matches` | Number of matches obtained |
| `messages_sent` | Number of messages sent |
| `session_length_minutes` | Session duration in minutes |
| `app_version` | App version used |
| `gender` | User gender (M/F/NB) |
| `age` | User age |
| `location` | User location (city) |
| `day_of_week` | Day of week when session occurred |

## 🔍 Key Metrics

### KPIs Calculated
- **Average Swipes per Session**: Average number of swipes across all sessions
- **Match Rate**: Percentage of likes that result in matches
- **Message Initiation Rate**: Percentage of matches that lead to messages
- **Average Session Length**: Mean session duration in minutes

### User Segments
1. **Serial Swiper**: High swipe volume, lower match efficiency
2. **Balanced Engager**: Moderate activity with good engagement
3. **High-Quality Matcher**: Selective swiping with high match rates
4. **Casual User**: Low engagement and infrequent usage

## 🧠 Actionable Insights

The system generates insights such as:
- Users who message within 5 minutes of matching have higher engagement
- Longer sessions correlate with more matches
- Selective swipers have 2x better swipe-to-match efficiency
- Day-of-week patterns for optimal engagement

## 🎨 Dashboard Features

### Overview KPIs Tab
- Real-time KPI cards
- Conversion funnel visualization
- Daily activity trends
- Swipes distribution

### User Segments Tab
- Segment distribution charts
- Behavioral comparison metrics
- Segment characteristic tables

### Predictive Model Tab
- Feature importance rankings
- Top predictors visualization
- Model performance insights

### Recommendations Tab
- Top 3 actionable insights
- Data-driven recommendations
- Filter-specific suggestions

### Filters
- **Location**: Filter by city
- **Day of Week**: Filter by specific day

## 🔬 Product Data Science Mapping

Each component of this project maps to real-world product data science functions:

### `generate_data.py` → **Event Tracking & Data Collection**
- Simulates event-level tracking system (similar to Amplitude, Mixpanel)
- Models user behavior patterns and correlations
- Represents data engineering and ETL pipelines

### `analyze_swipe_sessions.py` → **Experimentation & Analysis**
- **KPIs Calculation**: Product metrics monitoring (similar to dashboards in Looker, Tableau)
- **Funnel Analysis**: Conversion optimization (A/B testing framework)
- **Clustering**: User segmentation for personalization (cohort analysis)
- **Predictive Modeling**: Churn prediction, engagement forecasting
- **Insights Generation**: Product recommendations engine

### `app.py` → **Business Intelligence & Reporting**
- Interactive dashboards for stakeholders
- Real-time metrics tracking
- Self-service analytics (similar to Mode, Redash)
- Executive reporting and decision-making tools

## 🛠️ Technology Stack

- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, xgboost
- **Visualization**: plotly, seaborn, matplotlib
- **Dashboard**: streamlit
- **Data Analysis**: Statistical analysis, clustering, predictive modeling

## 📈 Use Cases

This project demonstrates:
1. **Product Analytics**: Understanding user behavior and engagement
2. **Data Science Pipeline**: End-to-end ML workflow
3. **Business Intelligence**: Executive dashboards and reporting
4. **A/B Testing Framework**: Hypothesis generation and validation
5. **User Segmentation**: Personalization and targeting
6. **Predictive Analytics**: Forecasting and risk assessment

## 🎓 Learning Outcomes

After exploring this project, you'll understand:
- How to simulate realistic user behavioral data
- Product metrics calculation and monitoring
- User segmentation techniques
- Building predictive models for product features
- Creating interactive dashboards for stakeholders
- Translating data insights into actionable recommendations

## 📝 Notes

- The data is synthetic and designed for demonstration purposes
- Model performance may vary based on generated data patterns
- Adjust parameters in `generate_data.py` to create different scenarios
- Customize clustering and modeling parameters in `analyze_swipe_sessions.py`

## 🤝 Contributing

This is an educational project. Feel free to:
- Modify data generation parameters
- Experiment with different ML models
- Add new visualizations to the dashboard
- Extend the analysis with additional metrics

## 📄 License

This project is for educational purposes.

---

**Built with 💘 for Product Data Science**