import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Load models
performance_model = joblib.load('supplier_performance_model.pkl')
risk_model = joblib.load('risk_model.pkl')  # Load the trained risk model

# Load data
supplier_data = pd.read_csv('supplier_data.csv')
feedback_data = pd.read_csv('feedback_data_with_sentiment.csv')  # Load the feedback data with sentiment

# Ensure correct data types
supplier_data['Delivery_Time'] = pd.to_numeric(supplier_data['Delivery_Time'], errors='coerce')
supplier_data['Quality_Rating'] = pd.to_numeric(supplier_data['Quality_Rating'], errors='coerce')
supplier_data['Quantity_Delivered'] = pd.to_numeric(supplier_data['Quantity_Delivered'], errors='coerce')
supplier_data['Historical_Performance_Score'] = pd.to_numeric(supplier_data['Historical_Performance_Score'], errors='coerce')

# Generate Risk Level Labels
def assign_risk_level(row):
    if row['Historical_Performance_Score'] < 33:
        return 'High'
    elif row['Historical_Performance_Score'] < 66:
        return 'Medium'
    else:
        return 'Low'

supplier_data['Risk_Level'] = supplier_data.apply(assign_risk_level, axis=1)

# Streamlit app
st.title('PPAP Process Insights and Predictions')

# Global Supplier Filter
supplier_list = ['ALL'] + list(supplier_data['Supplier_ID'].unique())
selected_supplier = st.sidebar.selectbox('Select Supplier ID:', options=supplier_list)

# Function to filter data based on selected supplier
def filter_data(data, supplier_id):
    if supplier_id == 'ALL':
        return data
    else:
        return data[data['Supplier_ID'] == supplier_id]

# Filter data based on selected supplier
filtered_supplier_data = filter_data(supplier_data, selected_supplier)
filtered_feedback_data = filter_data(feedback_data, selected_supplier)

# Predict supplier performance
st.sidebar.header('Input Parameters')
delivery_time = st.sidebar.slider('Delivery Time (Days)', 1, 30, 15)
quality_rating = st.sidebar.slider('Quality Rating', 1, 10, 5)
quantity_delivered = st.sidebar.number_input('Quantity Delivered', 1, 1000, 500)
historical_performance = st.sidebar.slider('Historical Performance Score', 0.0, 100.0, 50.0)

input_data = pd.DataFrame({
    'Delivery_Time': [delivery_time],
    'Quality_Rating': [quality_rating],
    'Quantity_Delivered': [quantity_delivered],
    'Historical_Performance_Score': [historical_performance]
})
performance_prediction = performance_model.predict(input_data)
st.write(f'Predicted Approval Status: {performance_prediction[0]}')

# Predict risk level for the input supplier
risk_prediction = risk_model.predict(input_data)
st.write(f'Predicted Risk Level: {risk_prediction[0]}')

# Visualizations
st.subheader('Historical Supplier Performance')

# Interactive Bar Chart with filtering
supplier_metrics = filtered_supplier_data.groupby('Supplier_ID').agg({
    'Delivery_Time': 'mean',
    'Quality_Rating': 'mean',
    'Quantity_Delivered': 'mean'
}).reset_index()
bar_chart = px.bar(
    supplier_metrics,
    x='Supplier_ID',
    y=['Delivery_Time', 'Quality_Rating', 'Quantity_Delivered'],
    title='Average Supplier Performance Metrics'
)
st.plotly_chart(bar_chart)

# Additional Interactive Charts
st.subheader('Additional Supplier Performance Analysis')

# Line Chart for Performance Trends
line_chart = px.line(
    filtered_supplier_data,
    x='Date',
    y=['Delivery_Time', 'Quality_Rating', 'Quantity_Delivered'],
    title='Supplier Performance Trends Over Time'
)
st.plotly_chart(line_chart)

# Missed Deliveries vs Quality Rating Chart
st.subheader('Missed Deliveries vs Quality Rating')

# Calculate missed deliveries
delivery_threshold = 20  # Example threshold for delivery time
filtered_supplier_data['Missed_Deliveries'] = filtered_supplier_data['Delivery_Time'] > delivery_threshold

# Group by supplier and date to count missed deliveries
missed_deliveries = filtered_supplier_data.groupby(['Date']).agg({
    'Missed_Deliveries': 'sum',
    'Quality_Rating': 'mean'
}).reset_index()

# Create combined bar and line chart
fig = go.Figure()

# Add bar chart for missed deliveries
fig.add_trace(go.Bar(
    x=missed_deliveries['Date'],
    y=missed_deliveries['Missed_Deliveries'],
    name='Missed Deliveries',
    marker_color='red',
    yaxis='y1'
))

# Add line chart for quality rating
fig.add_trace(go.Scatter(
    x=missed_deliveries['Date'],
    y=missed_deliveries['Quality_Rating'],
    name='Quality Rating',
    mode='lines+markers',
    line=dict(color='blue'),
    yaxis='y2'
))

# Update layout for dual y-axes
fig.update_layout(
    title='Missed Deliveries and Quality Rating Over Time',
    xaxis=dict(title='Date'),
    yaxis=dict(
        title='Missed Deliveries',
        titlefont=dict(color='red'),
        tickfont=dict(color='red')
    ),
    yaxis2=dict(
        title='Quality Rating',
        titlefont=dict(color='blue'),
        tickfont=dict(color='blue'),
        overlaying='y',
        side='right'
    ),
    legend=dict(x=0.1, y=1.1, orientation='h')
)

st.plotly_chart(fig)

# Display Risk Levels in a Table with filtering
st.subheader('Supplier Risk Levels')
risk_levels = filtered_supplier_data[['Supplier_ID', 'Risk_Level']].drop_duplicates()
st.dataframe(risk_levels)

# Sentiment Analysis of Supplier Feedback and Audit Reports
st.subheader('Supplier Feedback and Audit Report Analysis')

# Display sentiment analysis results
st.write("Sentiment Analysis of Supplier Feedback and Audit Reports:")
filtered_feedback_data['Sentiment_Score'] = filtered_feedback_data['Sentiment'].apply(lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral')
sentiment_summary = filtered_feedback_data.groupby('Sentiment_Score').size().reset_index(name='Count')
sentiment_chart = px.pie(sentiment_summary, names='Sentiment_Score', values='Count', title='Sentiment Distribution')
st.plotly_chart(sentiment_chart)

# Display the sentiment data with feedback and audit reports
st.write("Detailed Sentiment Analysis:")
st.dataframe(filtered_feedback_data[['Supplier_ID', 'Feedback', 'Audit_Report', 'Sentiment', 'Sentiment_Score']])
