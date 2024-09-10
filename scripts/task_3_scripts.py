import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import sys
import os

# Add the path to your scripts directory
sys.path.append(os.path.abspath(r'C:\Users\Yibabe\Desktop\kifiyaAIMweek-2\scripts'))
from load_data import load_data_from_postgres

def preprocess_data(df):
    # Replace missing values with column mean
    df.fillna(df.mean(), inplace=True)
    # Define metrics for standardization
    metrics = ['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)',
               'Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Average Throughput (kbps)']
    # Standardize the metrics
    scaler = StandardScaler()
    df[metrics] = scaler.fit_transform(df[metrics])
    return df

def compute_top_bottom_frequent(df, column):
    top_10 = df[column].nlargest(10)
    bottom_10 = df[column].nsmallest(10)
    most_frequent = df[column].mode().head(10)
    return top_10, bottom_10, most_frequent

def analyze_distribution(df, column, group_by):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=group_by, y=column, data=df)
    plt.title(f'Distribution of {column} by {group_by}')
    plt.show()

def average_per_group(df, metric, group_by):
    return df.groupby(group_by)[metric].mean().reset_index()

def kmeans_clustering(df, n_clusters=3):
    metrics = ['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)',
               'Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Average Throughput (kbps)']
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[metrics])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df_scaled)
    cluster_description = df.groupby('Cluster').agg({
        'TCP DL Retrans. Vol (Bytes)': ['mean', 'std'],
        'TCP UL Retrans. Vol (Bytes)': ['mean', 'std'],
        'Avg RTT DL (ms)': ['mean', 'std'],
        'Avg RTT UL (ms)': ['mean', 'std'],
        'Average Throughput (kbps)': ['mean', 'std']
    }).reset_index()
    return df, cluster_description

def main():
    # Load data from PostgreSQL
    query = "SELECT * FROM xdr_data;"  # Replace with your actual table name
    data_cleaned = load_data_from_postgres(query)
    
    if data_cleaned is None:
        print("Failed to load data.")
        sys.exit()

    # Data preprocessing
    data_cleaned = preprocess_data(data_cleaned)

    # Compute top, bottom, and most frequent values
    tcp_top, tcp_bottom, tcp_freq = compute_top_bottom_frequent(data_cleaned, 'TCP DL Retrans. Vol (Bytes)')
    rtt_top, rtt_bottom, rtt_freq = compute_top_bottom_frequent(data_cleaned, 'Avg RTT DL (ms)')
    throughput_top, throughput_bottom, throughput_freq = compute_top_bottom_frequent(data_cleaned, 'Average Throughput (kbps)')

    # Analyze distribution of throughput and TCP retransmission per handset type
    analyze_distribution(data_cleaned, 'Average Throughput (kbps)', 'Handset Type')
    avg_tcp_per_handset = average_per_group(data_cleaned, 'TCP DL Retrans. Vol (Bytes)', 'Handset Type')

    # Perform k-means clustering
    clustered_df, cluster_stats = kmeans_clustering(data_cleaned)
    print("Cluster Statistics:\n", cluster_stats)

if __name__ == '__main__':
    main()
