# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.path.append(os.path.abspath(r'C:\Users\Yibabe\Desktop\kifiyaAIMweek-2\scripts'))
from task1_2_connection import load_data_from_postgres

# Define the query to fetch data from the 'cleaned_data' table
query = "SELECT * FROM cleaned_data"

# Load the data into a DataFrame
df_cleaned = load_data_from_postgres(query)

# Function to segment users into deciles based on total session duration
def segment_users_into_deciles(df_cleaned):
    df_cleaned['Total Data (DL + UL)'] = df_cleaned['Total DL (Bytes)'] + df_cleaned['Total UL (Bytes)']
    df_cleaned['Decile'] = pd.qcut(df_cleaned['Dur. (ms)'], 10, labels=False)
    decile_data = df_cleaned.groupby('Decile').agg({'Total Data (DL + UL)': 'sum'}).reset_index()
    
    # Plot the total data per decile
    sns.barplot(x='Decile', y='Total Data (DL + UL)', data=decile_data)
    plt.title('Total Data (DL + UL) per Decile')
    plt.xlabel('Decile')
    plt.ylabel('Total Data (Bytes)')
    plt.show()

# Function to analyze basic metrics (mean, median, etc.)
def analyze_basic_metrics(df_cleaned):
    summary_stats = df_cleaned[['Dur. (ms)', 'Total DL (Bytes)', 'Total UL (Bytes)']].describe()
    print(summary_stats)
    
    # Histograms for basic metrics
    df_cleaned[['Dur. (ms)', 'Total DL (Bytes)', 'Total UL (Bytes)']].hist(bins=20, figsize=(10, 5))
    plt.show()

# Function to compute non-graphical univariate analysis
def non_graphical_univariate_analysis(df_cleaned):
    dispersion_metrics = {
        'Variance': df_cleaned.var(),
        'Standard Deviation': df_cleaned.std(),
        'Skewness': df_cleaned.skew(),
        'Kurtosis': df_cleaned.kurt()
    }
    dispersion_df = pd.DataFrame(dispersion_metrics)
    print("Dispersion Parameters:")
    print(dispersion_df)

# Function to create graphical univariate analysis
def graphical_univariate_analysis(df_cleaned):
    # Box plot for download and upload data
    sns.boxplot(data=df_cleaned[['Total DL (Bytes)', 'Total UL (Bytes)']])
    plt.title('Box Plot of Download and Upload Data')
    plt.show()

    # Violin plot for detailed distribution of Social Media, YouTube, and Netflix data
    sns.violinplot(data=df_cleaned[['Social Media DL (Bytes)', 'YouTube DL (Bytes)', 'Netflix DL (Bytes)']])
    plt.title('Violin Plot for App-Specific Data')
    plt.show()

# Function to conduct bivariate analysis (relationship between total data and each app)
def bivariate_analysis(df_cleaned):
    df_cleaned['Total Data (DL + UL)'] = df_cleaned['Total DL (Bytes)'] + df_cleaned['Total UL (Bytes)']
    
    # Scatter plot for YouTube DL vs Total Data
    sns.scatterplot(x='YouTube DL (Bytes)', y='Total Data (DL + UL)', data=df_cleaned)
    plt.title('Total Data vs YouTube Data')
    plt.show()

# Function to compute correlation matrix for app-specific data
def correlation_analysis(df_cleaned):
    app_columns = ['Social Media DL (Bytes)', 'YouTube DL (Bytes)', 'Netflix DL (Bytes)', 'Google DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']
    corr_matrix = df_cleaned[app_columns].corr()
    
    # Plot correlation matrix using heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix for App-Specific Data')
    plt.show()

# Function to perform PCA and dimensionality reduction
def dimensionality_reduction(df_cleaned):
    # Select relevant columns for PCA
    app_columns = ['Social Media DL (Bytes)', 'YouTube DL (Bytes)', 'Netflix DL (Bytes)', 'Google DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']
    data = df_cleaned[app_columns]
    
    # Standardize the data before PCA
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    
    # Plot the first two principal components
    plt.scatter(pca_result[:, 0], pca_result[:, 1])
    plt.title('PCA of App-Specific Data')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()

    # Explained variance ratio
    print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")

# Main function to call all the analysis steps
def run_analysis():
    # Define the query to fetch data from the 'cleaned_data' table
    query = "SELECT * FROM cleaned_data"

    # Load the data into a DataFrame
    df_cleaned = load_data_from_postgres(query)
    
    print("1. Segmenting Users into Deciles")
    segment_users_into_deciles(df_cleaned)
    
    print("2. Analyzing Basic Metrics")
    analyze_basic_metrics(df_cleaned)
    
    print("3. Non-Graphical Univariate Analysis")
    non_graphical_univariate_analysis(df_cleaned)
    
    print("4. Graphical Univariate Analysis")
    graphical_univariate_analysis(df_cleaned)
    
    print("5. Bivariate Analysis")
    bivariate_analysis(df_cleaned)
    
    print("6. Correlation Analysis")
    correlation_analysis(df_cleaned)
    
    print("7. Dimensionality Reduction")
    dimensionality_reduction(df_cleaned)

# Run the analysis
if __name__ == '__main__':
    run_analysis()
