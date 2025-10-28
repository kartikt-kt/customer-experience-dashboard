import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import joblib

# ML libs for in-app sentiment model
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Experience Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure VADER lexicon is available
nltk.download("vader_lexicon", quiet=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("merged_customer_experience.csv")
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    # --- Apply readable mappings ---
    customer_segment_map = {
        0: "Enterprise",
        1: "Individual",
        2: "SMB"
    }

    priority_map = {
        0: "Economy",
        1: "Express",
        2: "Standard"
    }

    product_category_map = {
        0: "Books",
        1: "Electronics",
        2: "Fashion",
        3: "Food & Beverage",
        4: "Healthcare",
        5: "Home Goods",
        6: "Industrial"
    }

    df["customer_segment"] = df["customer_segment"].map(customer_segment_map)
    df["priority"] = df["priority"].map(priority_map)
    df["product_category"] = df["product_category"].map(product_category_map)

    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load 'merged_customer_experience.csv'. Make sure file exists. Error: {e}")
    st.stop()

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("🔍 Filters")

def checkbox_multiselect(column_name, label):
    """Simplified multiselect without 'Select All'."""
    if column_name not in df.columns:
        return None
    unique_vals = sorted(df[column_name].dropna().astype(str).unique())
    with st.sidebar.expander(label, expanded=True):
        selected = []
        for v in unique_vals:
            if st.checkbox(v, value=True, key=f"{column_name}_{v}"):
                selected.append(v)
        # If none selected, default to all to avoid empty filtered frame
        if len(selected) == 0:
            selected = unique_vals.copy()
    return selected

selected_segments = checkbox_multiselect("customer_segment", "Customer Segment")
selected_priorities = checkbox_multiselect("priority", "Priority")
selected_categories = checkbox_multiselect("product_category", "Product Category")

# If any of these columns are missing, set default to all rows
if selected_segments is None:
    selected_segments = df["customer_segment"].astype(str).unique().tolist() if "customer_segment" in df.columns else []
if selected_priorities is None:
    selected_priorities = df["priority"].astype(str).unique().tolist() if "priority" in df.columns else []
if selected_categories is None:
    selected_categories = df["product_category"].astype(str).unique().tolist() if "product_category" in df.columns else []

# ---------------- APPLY FILTERS ----------------
filtered = df.copy()
if "customer_segment" in df.columns:
    filtered = filtered[filtered["customer_segment"].astype(str).isin(selected_segments)]
if "priority" in df.columns:
    filtered = filtered[filtered["priority"].astype(str).isin(selected_priorities)]
if "product_category" in df.columns:
    filtered = filtered[filtered["product_category"].astype(str).isin(selected_categories)]

# ---------------- HEADER & KPIs ----------------
st.title("📊 Customer Experience Dashboard")
st.markdown("Gain insights into delivery performance, cost efficiency, and customer satisfaction.")

def safe_mean(col):
    return filtered[col].mean() if col in filtered.columns else np.nan

def safe_median(col):
    return filtered[col].median() if col in filtered.columns else np.nan

avg_rating = safe_mean("customer_rating")
median_delay = safe_median("delivery_delay_days")
late_pct = ((filtered["delivery_delay_days"] > 0).mean() * 100) if "delivery_delay_days" in filtered.columns else np.nan
total_orders = len(filtered)

col1, col2, col3, col4 = st.columns(4)
col1.metric("⭐ Average Rating", f"{avg_rating:.2f}" if not np.isnan(avg_rating) else "N/A")
col2.metric("⏱️ Median Delay (days)", f"{median_delay:.1f}" if not np.isnan(median_delay) else "N/A")
col3.metric("🚚 % Late Deliveries", f"{late_pct:.1f}%" if not np.isnan(late_pct) else "N/A")
col4.metric("📦 Total Orders", f"{total_orders}")

st.markdown("---")

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Delivery Analysis",
    "💬 Customer Feedback",
    "💰 Cost Insights",
    "❤️ Sentiment Analysis",
    "⚠️ At-Risk Customers",
])

# ---------------- TAB 1: Delivery Analysis ----------------
with tab1:
    st.subheader("Delivery Performance Overview")
    if "delivery_delay_days" in filtered.columns:
        fig1 = px.histogram(filtered, x="delivery_delay_days", nbins=20,
                            title="Distribution of Delivery Delays (Days)", color_discrete_sequence=["#4C78A8"])
        st.plotly_chart(fig1, use_container_width=True)
        st.info("Interpretation: A tall bar near 0 means most deliveries are on time. Bars to the right indicate delays.")
    else:
        st.info("No delivery delay data available.")

    st.subheader("Delivery Performance by Carrier")
    if "carrier" in filtered.columns and "delivery_delay_days" in filtered.columns:
        fig2 = px.box(filtered, x="carrier", y="delivery_delay_days", color="carrier", title="Delivery Delay by Carrier")
        st.plotly_chart(fig2, use_container_width=True)
        st.info("Shorter boxes mean consistent delivery times; wide boxes or outliers indicate inconsistency.")

# ---------------- TAB 2: Customer Feedback ----------------
with tab2:
    st.subheader("Customer Ratings by Issue Category")
    if "issue_category" in filtered.columns and "customer_rating" in filtered.columns:
        fig3 = px.box(filtered, x="issue_category", y="customer_rating", color="issue_category",
                      title="Customer Ratings by Issue Category")
        st.plotly_chart(fig3, use_container_width=True)
        st.info("Higher medians and tighter boxes indicate better handling of those issues.")
    else:
        st.info("Issue category or rating data not available.")

    st.subheader("Recommendation Trends")
    if "would_recommend" in filtered.columns:
        fig4 = px.histogram(filtered, x="would_recommend", color="would_recommend", title="Would Customers Recommend Us?")
        st.plotly_chart(fig4, use_container_width=True)
        st.info("A higher count of 'Yes' reflects customer loyalty and satisfaction.")
    else:
        st.info("Recommendation data not present.")

# ---------------- TAB 3: Cost Insights ----------------
with tab3:
    st.subheader("Cost and Value Insights")
    if {"order_value_inr", "delivery_cost_inr", "product_category"}.issubset(filtered.columns):
        avg_costs = filtered.groupby("product_category")[["order_value_inr", "delivery_cost_inr"]].mean().reset_index()
        fig5 = px.bar(avg_costs, x="product_category", y=["order_value_inr", "delivery_cost_inr"],
                      barmode="group", title="Average Order Value vs Delivery Cost by Product Category",
                      labels={"value": "Average (INR)", "product_category": "Product Category"})
        st.plotly_chart(fig5, use_container_width=True)

        filtered["estimated_profit"] = filtered["order_value_inr"] - filtered["delivery_cost_inr"]
        avg_profit = filtered["estimated_profit"].mean()
        st.metric("💰 Estimated Average Profit per Order", f"₹{avg_profit:,.2f}")
        st.info("If delivery cost approaches order value, investigate logistics or pricing for that category.")
    else:
        st.info("Order value or delivery cost data not available.")

# ---------------- TAB 4: Sentiment Analysis ----------------
with tab4:
    st.subheader("Sentiment Analysis (VADER baseline + AI model)")

    if "feedback_text" in filtered.columns:
        sia = SentimentIntensityAnalyzer()

        filtered["vader_score"] = filtered["feedback_text"].astype(str).apply(lambda x: sia.polarity_scores(x)["compound"])
        filtered["vader_label"] = filtered["vader_score"].apply(lambda v: "Positive" if v > 0.05 else ("Negative" if v < -0.05 else "Neutral"))

        st.markdown("**VADER Sentiment (baseline)**")
        vader_counts = filtered["vader_label"].value_counts().reset_index()
        vader_counts.columns = ["Sentiment", "Count"]
        fig_vader = px.pie(vader_counts, names="Sentiment", values="Count",
                           title="VADER Sentiment Distribution",
                           color="Sentiment",
                           color_discrete_map={"Positive": "#4CAF50", "Neutral": "#FFEB3B", "Negative": "#F44336"})
        st.plotly_chart(fig_vader, use_container_width=True)
        st.info("Interpretation: VADER provides a quick lexical sentiment estimate; used as baseline and for model labels.")

        st.markdown("**AI Sentiment Model (Logistic Regression trained on VADER labels)**")
        feedback_mask = filtered["feedback_text"].notna() & (filtered["feedback_text"].astype(str).str.strip() != "")
        feedback_texts = filtered.loc[feedback_mask, "feedback_text"].astype(str)

        if len(feedback_texts) >= 10:
            pseudo_labels = (filtered.loc[feedback_mask, "vader_score"] > 0).astype(int)

            pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
                ("clf", LogisticRegression(max_iter=1000))
            ])

            X_train, X_test, y_train, y_test = train_test_split(feedback_texts, pseudo_labels, test_size=0.2, random_state=42)
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
            acc = accuracy_score(y_test, preds)
            st.success(f"AI sentiment model trained (pseudo-labels). Accuracy (on held-out): {acc*100:.2f}%")

            all_preds = pipeline.predict(filtered.loc[feedback_mask, "feedback_text"].astype(str))
            ai_series = pd.Series(index=filtered.index, dtype=object)
            ai_series.loc[feedback_mask] = np.where(all_preds == 1, "Positive", "Negative")
            ai_series.fillna("Neutral", inplace=True)
            filtered["ai_sentiment"] = ai_series

            ai_counts = filtered["ai_sentiment"].value_counts().reset_index()
            ai_counts.columns = ["Sentiment", "Count"]
            fig_ai = px.pie(ai_counts, names="Sentiment", values="Count", title="AI Sentiment Distribution",
                            color_discrete_map={"Positive": "#4CAF50", "Neutral": "#FFEB3B", "Negative": "#F44336"})
            st.plotly_chart(fig_ai, use_container_width=True)
            st.info("AI model provides a model-driven sentiment classification complementing VADER.")
        else:
            st.warning("Not enough feedback rows to train AI model. Showing only VADER baseline.")
    else:
        st.info("No feedback_text column present in the dataset.")

# ---------------- TAB 5: At-Risk Customers ----------------
with tab5:
    st.subheader("At-Risk Customers & Suggested Interventions")

    if "vader_score" in filtered.columns:
        filtered["sentiment_score"] = filtered["vader_score"]
    else:
        if "feedback_text" in filtered.columns:
            sia = SentimentIntensityAnalyzer()
            filtered["sentiment_score"] = filtered["feedback_text"].astype(str).apply(lambda x: sia.polarity_scores(x)["compound"])
        else:
            filtered["sentiment_score"] = 0.0

    required = {"customer_rating", "delivery_delay_days", "feedback_text", "would_recommend"}
    if required.issubset(filtered.columns):
        filtered["risk_score"] = (
            (filtered["customer_rating"].astype(float) < 3).astype(int) +
            (filtered["delivery_delay_days"].astype(float) > 2).astype(int) +
            (filtered["sentiment_score"].astype(float) < -0.05).astype(int) +
            (filtered["would_recommend"].astype(str).str.lower() == "no").astype(int)
        )
        filtered["at_risk_rule"] = (filtered["risk_score"] >= 2).astype(int)
        st.metric("🚨 Rule-based At-Risk Customers", int(filtered["at_risk_rule"].sum()))

        at_risk_df = filtered[filtered["at_risk_rule"] == 1]
        if not at_risk_df.empty:
            st.dataframe(at_risk_df[["order_id", "customer_segment", "product_category", "customer_rating",
                                     "delivery_delay_days", "sentiment_score", "would_recommend"]].head(15),
                         use_container_width=True)
            st.info("Interpretation: These customers show multiple negative indicators (low rating, delays, negative sentiment). Prioritize outreach.")
            st.markdown("### Suggested Interventions")
            st.markdown("""
            - Follow-up calls / personalized outreach  
            - Priority redelivery or refunds for delayed/failed orders  
            - Route cases to senior customer support for escalation  
            - Loyalty offers / coupons to retain customers  
            """)
        else:
            st.success("No high-risk customers detected in the selected filters.")
    else:
        st.warning("Required columns missing to calculate at-risk customers.")

    st.markdown("### AI Model Prediction (optional: load trained model)")

    model_path = "risk_model.pkl"
    if os.path.exists(model_path):
        try:
            risk_model = joblib.load(model_path)
            live = filtered.copy()

            for col in ["customer_segment", "priority", "product_category"]:
                if col in live.columns:
                    live[col] = live[col].astype("category").cat.codes
                else:
                    live[col] = 0

            feature_cols = ["customer_rating", "delivery_delay_days", "order_value_inr", "delivery_cost_inr",
                            "sentiment_score", "customer_segment", "priority", "product_category"]
            for c in feature_cols:
                if c not in live.columns:
                    live[c] = 0

            X_live = live[feature_cols].fillna(0)
            probs = risk_model.predict_proba(X_live)[:, 1]
            live["ai_risk_prob"] = probs
            live["ai_at_risk"] = (live["ai_risk_prob"] > 0.5).astype(int)

            st.metric("🤖 AI-predicted At-Risk Customers", int(live["ai_at_risk"].sum()))
            st.dataframe(live[["order_id", "customer_segment", "product_category", "ai_risk_prob", "ai_at_risk"]]
                         .sort_values("ai_risk_prob", ascending=False).head(15), use_container_width=True)
            st.info("AI model outputs probability of being at-risk. Combine with rule-based flags for prioritization.")
        except Exception as e:
            st.warning(f"Could not load or run risk_model.pkl: {e}")
    else:
        st.info("Optional: place trained 'risk_model.pkl' in app folder to enable AI risk predictions.")

st.markdown("---")
st.caption("Built by Kartik Trakroo using Streamlit + Plotly + NLTK | AI-Augmented Customer Experience Dashboard")
