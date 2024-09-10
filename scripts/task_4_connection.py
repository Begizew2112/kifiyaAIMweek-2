import os
import pymysql
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fetch MySQL database connection parameters from environment variables
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")  # Ensure this is the correct port for MySQL (default: 3306)
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

def load_data_from_mysql(query):
    """
    Connects to the MySQL database and loads data based on the provided SQL query.
    :param query: SQL query to execute.
    :return: DataFrame containing the results of the query.
    """
    try:
        # Establish a connection to the MySQL database
        connection = pymysql.connect(
            host=DB_HOST,
            port=int(DB_PORT),
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )

        # Load data using pandas
        df = pd.read_sql_query(query, connection)

        # Close the database connection
        connection.close()

        return df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def export_data_to_mysql(df):
    """
    Exports the given DataFrame to a MySQL table.
    :param df: DataFrame containing engagement, experience, and satisfaction scores.
    """
    try:
        # Establish a connection to the MySQL database
        connection = pymysql.connect(
            host=DB_HOST,
            port=int(DB_PORT),
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )

        # Create a cursor object
        cursor = connection.cursor()

        # Example SQL query to insert data into the MySQL table
        for index, row in df.iterrows():
            sql_query = """
            INSERT INTO user_scores (MSISDN_Number, engagement_score, experience_score, satisfaction_score)
            VALUES (%s, %s, %s, %s)
            """
            cursor.execute(sql_query, (
                row['MSISDN/Number'], 
                row['engagement_score'], 
                row['experience_score'], 
                row['satisfaction_score']
            ))

        # Commit the changes and close the connection
        connection.commit()
        cursor.close()
        connection.close()

        print("Data exported successfully.")

    except Exception as e:
        print(f"An error occurred during export: {e}")


# Example usage
# query = "SELECT * FROM your_table"  # Modify as per your requirement
# df = load_data_from_mysql(query)

# If df is already calculated from previous tasks:
# export_data_to_mysql(df)
