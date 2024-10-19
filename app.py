import sys
import os

import certifi
ca = certifi.where()

import streamlit as st
import pandas as pd

from dotenv import load_dotenv
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

# Load environment variables
load_dotenv()
mongo_db_url = os.getenv("MONGODB_URL_KEY")

# MongoDB client setup
import pymongo
client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
from networksecurity.constants.training_pipeline import DATA_INGESTION_COLLECTION_NAME
from networksecurity.constants.training_pipeline import DATA_INGESTION_DATABASE_NAME

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

# Streamlit application setup
st.set_page_config(page_title="Network Security Prediction", layout="wide")

# Sidebar for Navigation
st.sidebar.title("Navigation")
page_selection = st.sidebar.selectbox("Select Page", ["Home", "Train Model", "Predict"])

# Home Page
if page_selection == "Home":
    st.title("Welcome to the Network Security Prediction System")
    st.write("Use the sidebar to navigate to training or prediction sections.")

# Train Model Page
if page_selection == "Train Model":
    st.title("Train the Network Security Model")
    
    if st.button("Start Training"):
        try:
            train_pipeline = TrainingPipeline()
            with st.spinner('Training is in progress...'):
                train_pipeline.run_pipeline()
            st.success("Training is successful!")
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            raise NetworkSecurityException(e, sys)

# Predict Page
if page_selection == "Predict":
    st.title("Upload Data for Prediction")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            # Load the uploaded CSV file
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data:", df.head())

            # Load the preprocessor and model
            preprocessor = load_object("final_model/preprocessor.pkl")
            final_model = load_object("final_model/model.pkl")
            network_model = NetworkModel(preprocessor=preprocessor, model=final_model)

            # Make predictions
            y_pred = network_model.predict(df)

            # Add the predictions to the DataFrame
            df['predicted_column'] = y_pred
            st.write("Prediction Results:", df)

            # Provide a download option for the results
            csv_output = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", csv_output, "predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            raise NetworkSecurityException(e, sys)
