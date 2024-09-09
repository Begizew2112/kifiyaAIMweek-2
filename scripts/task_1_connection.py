import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fetch database connection parameters from environment variables
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

def load_data_from_postgres(query):
    """
    Connects to the PostgreSQL database and loads data based on the provided SQL query.
    :param query: SQL query to execute.
    :return: DataFrame containing the results of the query.
    """
    try:
        # Create the SQLAlchemy engine
        engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
        
        # Load data using pandas
        df = pd.read_sql_query(query, engine)
        
        return df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
