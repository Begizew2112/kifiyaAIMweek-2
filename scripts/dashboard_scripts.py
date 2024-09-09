import pandas as pd
import streamlit as st
import sys
import os

# Add the path to your data connection script
sys.path.append(os.path.abspath(r'C:\Users\Yibabe\Desktop\kifiyaAIMweek-2\scripts'))
from task_1_connection import load_data_from_postgres

# Define the query to fetch data from the 'cleaned_data' table
query = "SELECT * FROM cleaned_data"

# Load the data into a DataFrame
df_cleaned = load_data_from_postgres(query)

# Check if data is loaded correctly
if df_cleaned is not None:
    st.write("Data successfully loaded!")
else:
    st.write("Failed to load data.")

# Calculate Throughput (kbps) and add it as a new column
df_cleaned['Throughput (kbps)'] = (df_cleaned['Total DL (Bytes)'] + df_cleaned['Total UL (Bytes)']) / df_cleaned['Dur. (ms)'] * 8 / 1000

# Check the first few rows to ensure throughput is calculated
st.write(df_cleaned[['Throughput (kbps)', 'Total DL (Bytes)', 'Total UL (Bytes)', 'Dur. (ms)']].head())

# KPIs - Example: Average Throughput
avg_throughput = df_cleaned['Throughput (kbps)'].mean()
st.write(f"**Average Throughput**: {avg_throughput:.2f} kbps")

# You can use this calculated column in your visualizations or further analysis.
import plotly.express as px

# Group by 'Handset Type' and calculate the average throughput
avg_throughput_by_handset = df_cleaned.groupby('Handset Type')['Avg Bearer TP DL (kbps)'].mean().reset_index()

# Create the bar chart
fig = px.bar(avg_throughput_by_handset, x='Handset Type', y='Avg Bearer TP DL (kbps)', 
             title='Average Throughput by Handset Type')
st.plotly_chart(fig)
# Create the histogram
fig = px.histogram(df_cleaned, x='Avg RTT DL (ms)', nbins=50, 
                   title='Distribution of Average RTT')
st.plotly_chart(fig)
# Create the scatter plot
fig = px.scatter(df_cleaned, x='Avg RTT DL (ms)', y='Avg Bearer TP DL (kbps)', 
                 title='Average RTT vs. Average Throughput')
st.plotly_chart(fig)
# Count occurrences of each handset type
handset_distribution = df_cleaned['Handset Type'].value_counts().reset_index()
handset_distribution.columns = ['Handset Type', 'Count']

# Create the pie chart
fig = px.pie(handset_distribution, names='Handset Type', values='Count', 
             title='Distribution of Handset Types')
st.plotly_chart(fig)
# Count occurrences of each handset type
handset_distribution = df_cleaned['Handset Type'].value_counts().reset_index()
handset_distribution.columns = ['Handset Type', 'Count']

# Count occurrences of each handset type
handset_distribution = df_cleaned['Handset Type'].value_counts().reset_index()
handset_distribution.columns = ['Handset Type', 'Count']

# Create the pie chart
fig = px.pie(handset_distribution, names='Handset Type', values='Count', 
             title='Distribution of Handset Types')
st.plotly_chart(fig)
# Convert 'Start' to datetime and set as index
df_cleaned['Start'] = pd.to_datetime(df_cleaned['Start'])
df_cleaned.set_index('Start', inplace=True)

# Resample data by day and sum total download bytes
daily_downloads = df_cleaned['Total DL (Bytes)'].resample('D').sum().reset_index()

# Create the line chart
fig = px.line(daily_downloads, x='Start', y='Total DL (Bytes)', 
              title='Total Download Bytes Over Time')
st.plotly_chart(fig)
# Create the box plot
fig = px.box(df_cleaned, x='Handset Type', y='TCP DL Retrans. Vol (Bytes)', 
             title='TCP Retransmissions by Handset Type')
st.plotly_chart(fig)
import plotly.figure_factory as ff

# Compute the correlation matrix
correlation_matrix = df_cleaned[['Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)', 
                         'TCP DL Retrans. Vol (Bytes)', 'Total DL (Bytes)']].corr()

import plotly.express as px

# Create a heatmap
fig = px.imshow(correlation_matrix, text_auto=True, color_continuous_scale='Viridis')

# Set z-axis properties if needed
fig.update_layout(coloraxis_colorbar=dict(
    title='Correlation',
    tickvals=[-1, 0, 1],
    ticktext=['-1', '0', '1']
))

# Show the figure
st.plotly_chart(fig)




