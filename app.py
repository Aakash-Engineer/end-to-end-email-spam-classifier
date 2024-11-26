# app.py

import pandas as pd
import streamlit as st
import pickle
from src.model import Preprocessing  # Ensure this import works correctly

# Load the trained model
with open('models/model.pkl', 'rb') as file:
    model = pickle.load(file)

# Set up the page configuration
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add information to the sidebar
with st.sidebar:
    st.title("📊 Project Info")
    st.markdown("""
    **Email Spam Classifier**  
    This application uses a machine learning model to classify emails as **Spam** or **Not Spam**.

    **Features:**
    - Text preprocessing
    - Model prediction
    - User-friendly interface

    **How it works:**
    1. Enter the email content in the text area.
    2. Click the **Classify** button to see the result.

    **About the Developer:**
    - [Aakash](aakash-engineer.github.io/home/)

    """)
    # Include an image or logo if available
    # st.image("images/logo.png", use_column_width=True)

# Add a title and description in the main page
st.title("📧 Email Spam Classifier")
st.write("Enter the content of an email to determine if it's **Spam** or **Not Spam**.")

# Create a text area for email input
email_input = st.text_area("Email Content", height=200)

# When the 'Classify' button is clicked
if st.button("Classify"):
    if email_input:
        # Convert the input text into a pandas Series
        processed_text = pd.Series(email_input)
        
        # Make a prediction
        prediction = model.predict(processed_text)
        
        if prediction[0] == 1:
            st.error("🚫 The email is classified as **Spam**.")
        else:
            st.success("✅ The email is classified as **Not Spam**.")
    else:
        st.warning("Please enter the email content.")

# Custom styling
st.markdown("""
    <style>
        /* Main content style */
        .reportview-container .main .block-container{
            padding-top: 2rem;
            padding-right: 2rem;
            padding-left: 2rem;
            padding-bottom: 2rem;
        }
        /* Sidebar style */
        .sidebar .sidebar-content {
            background-color: #f0f2f6;
            padding: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)