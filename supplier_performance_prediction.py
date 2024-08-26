import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pickle
from fpdf import FPDF
import base64
from io import BytesIO

# Ford's blue color
FORD_BLUE = "#003399"

# Load the models and scalers from the pickle files
with open('vocf_model_risk.pkl', 'rb') as f:
    model_risk, scaler_risk = pickle.load(f)

with open('vocf_model_failure.pkl', 'rb') as f:
    model_failure, scaler_failure = pickle.load(f)

# Streamlit App
st.set_page_config(page_title="VoCF Dashboard", layout="wide")

# Add a logo to the dashboard
st.image("logo.png", width=200)

st.markdown(f"<h1 style='color: {FORD_BLUE};'>VoCF Insights and Risk Mitigation Dashboard</h1>", unsafe_allow_html=True)

# Upload new CSV file
st.sidebar.header('Upload New Data (Summary Format)')
st.sidebar.markdown(f"""
    <style>
    .stFileUpload label {{
        background-color: {FORD_BLUE};
        color: white;
        border-radius: 5px;
        padding: 10px;
    }}
    .stFileUpload label:hover {{
        background-color: #002266;
    }}
    </style>
""", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    new_data = pd.read_csv(uploaded_file)

    # Preprocess the new data
    new_data['UNITS_REQUIRING_REFLASH'] = pd.to_numeric(new_data['UNITS_REQUIRING_REFLASH'], errors='coerce')
    new_data['UNITS_REFLASHED'] = pd.to_numeric(new_data['UNITS_REFLASHED'], errors='coerce')
    new_data['EXPECTED_REFLASH_DURATION_PER_UNIT_IN_MINS'] = pd.to_numeric(new_data['EXPECTED_REFLASH_DURATION_PER_UNIT_IN_MINS'], errors='coerce')
    new_data['Percentage_Updated'] = (new_data['UNITS_REFLASHED'] / new_data['UNITS_REQUIRING_REFLASH']) * 100
    new_data['ALERT_APPROVAL_DATE'] = pd.to_datetime(new_data['ALERT_APPROVAL_DATE'], errors='coerce')
    
    if 'COMPLETED_DATE' in new_data.columns:
        new_data['COMPLETED_DATE'] = pd.to_datetime(new_data['COMPLETED_DATE'], errors='coerce')
    else:
        new_data['COMPLETED_DATE'] = pd.NaT
    
    new_data['COMPLETED_DATE'] = new_data['COMPLETED_DATE'].fillna(pd.Timestamp.today())
    new_data['ALERT_APPROVAL_DATE'] = new_data['ALERT_APPROVAL_DATE'].fillna(pd.Timestamp.today())
    new_data['Time_to_Completion'] = (new_data['COMPLETED_DATE'] - new_data['ALERT_APPROVAL_DATE']).dt.days
    new_data['OTA_CAPABILITY'] = new_data['OTA_CAPABILITY'].fillna('Unknown')
    new_data['SW_AVAILABILITY_DATE'] = new_data['SW_AVAILABILITY_DATE'].fillna(pd.Timestamp.today())

    new_data['Alert_Counts'] = new_data.groupby('PROGRAM_CODE')['PROGRAM_CODE'].transform('size')
    new_data['Average_Reflash_Duration'] = new_data.groupby('PROGRAM_CODE')['EXPECTED_REFLASH_DURATION_PER_UNIT_IN_MINS'].transform('mean')
    new_data['Number_of_Modules'] = new_data.groupby('PROGRAM_CODE')['MODULE_NAME'].transform('nunique')
    new_data['Risk'] = new_data['Percentage_Updated'].apply(lambda x: 1 if x < 50 else 0)
    new_data['Module_Failure'] = new_data['Alert_Counts'].apply(lambda x: 1 if x > 10 else 0)

    # Select relevant features for the new data
    X_new_risk = new_data[['Percentage_Updated', 'Alert_Counts', 'Time_to_Completion', 'Average_Reflash_Duration', 'Number_of_Modules']]
    X_new_failure = new_data[['Percentage_Updated', 'Alert_Counts', 'Time_to_Completion', 'Average_Reflash_Duration', 'Number_of_Modules']]

    # Standardize the new data
    X_new_risk = scaler_risk.transform(X_new_risk)
    X_new_failure = scaler_failure.transform(X_new_failure)

    # Make predictions on the new data
    new_data['Risk_Prediction'] = model_risk.predict(X_new_risk)
    new_data['Module_Failure_Prediction'] = model_failure.predict(X_new_failure)

    # Display the new data with predictions
    st.markdown(f"<h2 style='color: {FORD_BLUE};'>New Data with Predictions</h2>", unsafe_allow_html=True)
    relevant_columns = ['PROGRAM_CODE', 'MODEL_YEAR', 'MODULE_NAME', 'Percentage_Updated', 'Alert_Counts', 'Time_to_Completion', 'Risk_Prediction', 'Module_Failure_Prediction']
    styled_data = new_data[relevant_columns].style.applymap(lambda x: 'background-color: red' if x == 1 else 'background-color: green', subset=['Risk_Prediction', 'Module_Failure_Prediction'])
    st.dataframe(styled_data)

    # Function to save Plotly figures as images
    def save_plotly_figure(fig, filename):
        fig.write_image(filename)

    # Create and save Plotly figures
    fig1 = px.histogram(new_data, x='Percentage_Updated', title='Distribution of Percentage of VINs Updated', marginal='box', nbins=50, color_discrete_sequence=[FORD_BLUE])
    fig1.update_layout(xaxis_title='Percentage of VINs Updated', yaxis_title='Count', height=500)
    save_plotly_figure(fig1, 'percentage_updated.png')

    alert_counts = new_data.groupby('PROGRAM_CODE')['Alert_Counts'].sum().reset_index()
    fig2 = px.bar(alert_counts, x='PROGRAM_CODE', y='Alert_Counts', title='Distribution of Alert Counts per Program Code', color_discrete_sequence=[FORD_BLUE])
    fig2.update_layout(xaxis_title='Program Code', yaxis_title='Alert Counts', height=500)
    save_plotly_figure(fig2, 'alert_counts.png')

    fig3 = px.histogram(new_data, x='Time_to_Completion', title='Distribution of Time to Completion', marginal='box', nbins=50, color_discrete_sequence=[FORD_BLUE])
    fig3.update_layout(xaxis_title='Time to Completion (days)', yaxis_title='Count', height=500)
    save_plotly_figure(fig3, 'time_to_completion.png')

    fig4 = px.pie(new_data, names='OTA_CAPABILITY', title='OTA Capability Distribution', color_discrete_sequence=[FORD_BLUE])
    fig4.update_layout(height=500)
    save_plotly_figure(fig4, 'ota_capability.png')

    fig5 = px.bar(new_data, x='Risk_Prediction', title='Risk Predictions', color='Risk_Prediction', barmode='group', color_discrete_sequence=[FORD_BLUE])
    fig5.update_layout(xaxis_title='Risk Prediction', yaxis_title='Count', height=500)
    save_plotly_figure(fig5, 'risk_predictions.png')

    resource_data = new_data.groupby('PROGRAM_CODE')['Average_Reflash_Duration'].mean().reset_index()
    fig6 = px.bar(resource_data, x='PROGRAM_CODE', y='Average_Reflash_Duration', title='Average Reflash Duration per Program Code', color_discrete_sequence=[FORD_BLUE])
    fig6.update_layout(xaxis_title='Program Code', yaxis_title='Average Reflash Duration (mins)', height=500)
    save_plotly_figure(fig6, 'average_reflash_duration.png')

    fig7 = px.bar(new_data, x='Module_Failure_Prediction', title='Module Failure Predictions', color='Module_Failure_Prediction', barmode='group', color_discrete_sequence=[FORD_BLUE])
    fig7.update_layout(xaxis_title='Module Failure Prediction', yaxis_title='Count', height=500)
    save_plotly_figure(fig7, 'module_failure_predictions.png')

    fig8 = px.box(new_data, x='Time_to_Completion', title='Time to Completion Box Plot', color_discrete_sequence=[FORD_BLUE])
    fig8.update_layout(xaxis_title='Time to Completion (days)', height=500)
    save_plotly_figure(fig8, 'time_to_completion_box.png')

    new_data['Month'] = new_data['ALERT_APPROVAL_DATE'].dt.to_period('M')
    trend_data = new_data.groupby('Month')['Percentage_Updated'].mean().reset_index()
    trend_data['Month'] = trend_data['Month'].astype(str)
    fig9 = px.line(trend_data, x='Month', y='Percentage_Updated', title='Trend of Software Update Success Rates Over Time', color_discrete_sequence=[FORD_BLUE])
    fig9.update_layout(xaxis_title='Month', yaxis_title='Average Percentage Updated', height=500)
    save_plotly_figure(fig9, 'trend_of_success_rates.png')

    fig10 = px.scatter(new_data, x='Percentage_Updated', y='Alert_Counts', title='Percentage Updated vs. Number of Alerts', color_discrete_sequence=[FORD_BLUE])
    fig10.update_layout(xaxis_title='Percentage of VINs Updated', yaxis_title='Number of Alerts', height=500)
    save_plotly_figure(fig10, 'percentage_vs_alerts.png')

    correlation_matrix = new_data[['Percentage_Updated', 'Alert_Counts', 'Time_to_Completion', 'Average_Reflash_Duration', 'Number_of_Modules']].corr()
    fig11 = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='Viridis'))
    fig11.update_layout(title='Correlation Heatmap', height=500)
    save_plotly_figure(fig11, 'correlation_heatmap.png')

    fig12 = px.box(new_data, x='PROGRAM_CODE', y='Percentage_Updated', title='Percentage Updated per Program Code', color_discrete_sequence=[FORD_BLUE])
    fig12.update_layout(xaxis_title='Program Code', yaxis_title='Percentage Updated', height=500)
    save_plotly_figure(fig12, 'percentage_updated_per_program.png')

    # Function to generate PDF
    def generate_pdf():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, txt="VoCF Insights and Risk Mitigation Dashboard", ln=True, align='C')
        pdf.ln(10)

        pdf.cell(200, 10, txt="Overview:", ln=True)
        pdf.multi_cell(0, 10, txt="This dashboard provides insights and risk mitigation strategies for software updates. It includes predictions for software update risks and module failures, as well as visualizations for trend analysis, root cause analysis, resource allocation, customer satisfaction, and comparison of program codes.")
        pdf.ln(10)

        pdf.cell(200, 10, txt="Predictions:", ln=True)
        pdf.multi_cell(0, 10, txt="The dashboard predicts the risk of software update failures and module failures based on historical data. High-risk predictions are highlighted in red, while low-risk predictions are highlighted in green.")
        pdf.ln(10)

        pdf.cell(200, 10, txt="Insights:", ln=True)
        pdf.multi_cell(0, 10, txt="The dashboard provides various insights, including trend analysis of software update success rates over time, root cause analysis using correlation heatmaps, resource allocation based on average reflash duration per program code, customer satisfaction analysis, and comparison of program codes.")
        pdf.ln(10)

        pdf.cell(200, 10, txt="Visualizations:", ln=True)
        pdf.multi_cell(0, 10, txt="The dashboard includes histograms, bar plots, pie charts, scatter plots, and box plots to visualize the data and insights. Each visualization is accompanied by a description to help interpret the results.")
        
        pdf.add_page()
        pdf.cell(200, 10, txt="Graphs and Visualizations:", ln=True)
        
        pdf.ln(10)
        pdf.cell(200, 10, txt="Distribution of Percentage of VINs Updated:", ln=True)
        pdf.multi_cell(0, 10, txt="Business Insight: This histogram shows the distribution of the percentage of VINs that have been updated, helping to understand how many VINs have been updated and the spread of the update percentages.\nTechnical Insight: Calculation is Percentage_Updated = (UNITS_REFLASHED / UNITS_REQUIRING_REFLASH) * 100.")
        pdf.image('percentage_updated.png', x=10, y=pdf.get_y(), w=190)
        pdf.ln(100)
        
        pdf.cell(200, 10, txt="Distribution of Alert Counts per Program Code:", ln=True)
        pdf.multi_cell(0, 10, txt="Business Insight: This bar plot displays the distribution of alert counts across the new data, grouped by program code, helping to visualize how many alerts have been raised for each program code.\nTechnical Insight: Alert_Counts is the number of alerts per PROGRAM_CODE.")
        pdf.image('alert_counts.png', x=10, y=pdf.get_y(), w=190)
        pdf.ln(100)
        
        pdf.cell(200, 10, txt="Distribution of Time to Completion:", ln=True)
        pdf.multi_cell(0, 10, txt="Business Insight: This histogram shows the distribution of the time taken to complete the updates, helping to understand how long the updates are taking to complete.\nTechnical Insight: Calculation is Time_to_Completion = COMPLETED_DATE - ALERT_APPROVAL_DATE.")
        pdf.image('time_to_completion.png', x=10, y=pdf.get_y(), w=190)
        pdf.ln(100)
        
        pdf.cell(200, 10, txt="OTA Capability Distribution:", ln=True)
        pdf.multi_cell(0, 10, txt="Business Insight: This pie chart shows the distribution of OTA capability, helping to understand how many units have OTA capability.\nTechnical Insight: OTA_CAPABILITY is a categorical feature indicating whether the unit has OTA capability.")
        pdf.image('ota_capability.png', x=10, y=pdf.get_y(), w=190)
        pdf.ln(100)
        
        pdf.cell(200, 10, txt="Risk Predictions:", ln=True)
        pdf.multi_cell(0, 10, txt="Business Insight: This bar plot shows the count of high-risk and low-risk predictions, helping to visualize the distribution of risk predictions.\nTechnical Insight: Risk_Prediction is 1 if Percentage_Updated < 50, else 0.")
        pdf.image('risk_predictions.png', x=10, y=pdf.get_y(), w=190)
        pdf.ln(100)
        
        pdf.cell(200, 10, txt="Average Reflash Duration per Program Code:", ln=True)
        pdf.multi_cell(0, 10, txt="Business Insight: This bar plot shows the average reflash duration per program code, helping to optimize resource allocation for software updates.\nTechnical Insight: Average_Reflash_Duration is the mean of EXPECTED_REFLASH_DURATION_PER_UNIT_IN_MINS per PROGRAM_CODE.")
        pdf.image('average_reflash_duration.png', x=10, y=pdf.get_y(), w=190)
        pdf.ln(100)
        
        pdf.cell(200, 10, txt="Module Failure Predictions:", ln=True)
        pdf.multi_cell(0, 10, txt="Business Insight: This bar plot shows the count of predicted module failures, helping to visualize the distribution of module failure predictions.\nTechnical Insight: Module_Failure_Prediction is 1 if Alert_Counts > 10, else 0.")
        pdf.image('module_failure_predictions.png', x=10, y=pdf.get_y(), w=190)
        pdf.ln(100)
        
        pdf.cell(200, 10, txt="Time to Completion Box Plot:", ln=True)
        pdf.multi_cell(0, 10, txt="Business Insight: This box plot shows the spread and outliers in time to completion, helping to visualize the distribution of time to completion values.\nTechnical Insight: Calculation is Time_to_Completion = COMPLETED_DATE - ALERT_APPROVAL_DATE.")
        pdf.image('time_to_completion_box.png', x=10, y=pdf.get_y(), w=190)
        pdf.ln(100)
        
        pdf.cell(200, 10, txt="Trend of Software Update Success Rates Over Time:", ln=True)
        pdf.multi_cell(0, 10, txt="Business Insight: This line plot shows the trend of software update success rates over time, helping to understand how the success rates have changed over time.\nTechnical Insight: Trend data is grouped by month and the average Percentage_Updated is calculated.")
        pdf.image('trend_of_success_rates.png', x=10, y=pdf.get_y(), w=190)
        pdf.ln(100)
        
        pdf.cell(200, 10, txt="Percentage Updated vs. Number of Alerts:", ln=True)
        pdf.multi_cell(0, 10, txt="Business Insight: This scatter plot shows the relationship between the percentage updated and the number of alerts, helping to understand how software update performance affects customer satisfaction.\nTechnical Insight: Scatter plot of Percentage_Updated vs. Alert_Counts.")
        pdf.image('percentage_vs_alerts.png', x=10, y=pdf.get_y(), w=190)
        pdf.ln(100)
        
        pdf.cell(200, 10, txt="Correlation Heatmap:", ln=True)
        pdf.multi_cell(0, 10, txt="Business Insight: This heatmap shows the correlation between different features, helping to identify potential root causes of software update failures.\nTechnical Insight: Correlation matrix of selected features.")
        pdf.image('correlation_heatmap.png', x=10, y=pdf.get_y(), w=190)
        pdf.ln(100)
        
        pdf.cell(200, 10, txt="Percentage Updated per Program Code:", ln=True)
        pdf.multi_cell(0, 10, txt="Business Insight: This box plot shows the distribution of percentage updated per program code, helping to compare the performance of different program codes.\nTechnical Insight: Calculation is Percentage_Updated per PROGRAM_CODE.")
        pdf.image('percentage_updated_per_program.png', x=10, y=pdf.get_y(), w=190)
        pdf.ln(100)

        return pdf.output(dest='S').encode('latin1')

    # Button to download PDF
    pdf_file = generate_pdf()
    b64 = base64.b64encode(pdf_file).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="VoCF_Insights_and_Risk_Mitigation_Dashboard.pdf">Download PDF</a>'
    st.markdown
