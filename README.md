README.md — Customer Experience Dashboard

Project Overview:
- The Customer Experience Dashboard is a Streamlit-based analytics application that visualizes end-to-end delivery, cost, and customer feedback data.
- It helps identify key trends in delivery performance, logistics efficiency, customer satisfaction, and overall sentiment.
The goal is to support data-driven decisions and highlight at-risk customers who may require intervention to prevent churn.

Objectives:
- Analyze customer experience data from multiple CSV sources (orders, costs, feedback, performance, logistics).
- Create an interactive dashboard that provides actionable insights in real time.
- Identify at-risk customers using quantitative and sentiment-based indicators.
- Support data-driven recommendations for improving customer satisfaction and delivery performance.
- Establish a clear, reproducible data preparation process using Google Colab.

Features:
- Interactive Filters :— Filter by customer segment, order priority, and product category.
- Dynamic KPIs :— View metrics such as average rating, delivery delays, and on-time performance.
- Delivery Analysis :— Understand delay distributions and carrier consistency.
- Customer Feedback :— Visualize issue categories and customer recommendation trends.
- Cost Insights :— Compare delivery costs versus order values for operational efficiency.
- Sentiment Analysis :— Perform text-based sentiment scoring using NLTK’s VADER analyzer.
- At-Risk Detection :— Identify customers likely to churn and suggest targeted interventions.
- Colab Data Preparation :— Merge and clean multiple datasets before dashboard integration.

Tech Stack:
- Frontend/UI: Streamlit (v1.50.0)
- Data Handling: Pandas (v2.3.3), NumPy (v2.3.4)
- Visualization: Plotly (v6.3.1), Matplotlib (v3.9.2)
- Sentiment Analysis: NLTK (v3.9.2)
- Environment: Python 3.11+
- Data Preprocessing: Google Colab

Project Structure:
Customer Experience Dashboard/
│
├── app.py                          # Main Streamlit application
├── data_preparation.ipynb          # Google Colab notebook for data merging and cleaning
├── merged_customer_experience.csv  # Final combined dataset
├── requirements.txt                # Dependencies for reproducibility
├── README.md                       # Project documentation
└── innovation_brief.pdf            # Final summary and recommendations


Setup Instructions:

1) Create a virtual environment

   python -m venv venv
   
   Activate it:
    Windows → venv\Scripts\activate
    macOS/Linux → source venv/bin/activate

2) Install dependencies
   pip install -r requirements.txt

3) Run the dashboard
   streamlit run app.py

Dataset Description:
- The merged dataset consolidates multiple operational data files including:
  - orders.csv — Order details and customer information.
  - delivery_performance.csv — Delivery times, service ratings, and performance metrics.
  - customer_feedback.csv — Text-based customer feedback and recommendations.
  - routes_distance.csv — Travel distances, route complexity, and toll data.
  - cost_breakdown.csv — Logistics, transport, and operational cost breakdowns.
  - warehouse_inventory.csv and vehicle_fleet.csv — Supply chain and fleet context.
- All datasets were merged using Order_ID as the primary key.
- The merged file was generated using data_preparation.ipynb in Google Colab for consistency and quality assurance.

At-Risk Customer Detection Logic:
- A risk score was computed using four weighted indicators:
   -Low customer rating (< 3)
   -Delivery delay greater than 2 days
   -Negative sentiment (VADER compound < -0.05)
   -“Would Recommend” marked as No
- Customers scoring 2 or more on these indicators were flagged as at-risk.
- These insights help teams proactively engage and recover dissatisfied customers.

Future Enhancements:
- Incorporate time-series trend analysis for delivery KPIs.
- Develop a machine learning model for predictive churn risk estimation.
- Integrate with Power BI or Tableau for advanced reporting.
- Automate weekly performance summary reports using Streamlit scheduling or cron jobs.

AI and Predictive Analytics Integration

The latest version extends the dashboard with machine learning capabilities to move from descriptive to predictive insights.

Key Additions:
- AI Sentiment Model: A Logistic Regression model trained on feedback text using TF-IDF features and VADER-based pseudo-labels provides more reliable sentiment classification into Positive, Neutral, and Negative categories.
- AI Risk Prediction: An optional pre-trained model (risk_model.pkl) predicts customer churn probability using delivery, cost, and sentiment metrics, complementing the rule-based approach.
- Impact: These AI components enable early detection of dissatisfied or high-risk customers, supporting proactive retention strategies and data-driven decision-making.

Author
Kartik Trakroo
B.Tech Computer Science & Engineering
Manipal University Jaipur
