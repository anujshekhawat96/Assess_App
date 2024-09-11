import streamlit as st
# from sklearn.externals import joblib  
import pandas as pd
import plotly.express as px

# Load your machine learning model
# model = joblib.load('models/your_model.pkl')

# Define a function for each page
def show_eda_page():
    st.title('Exploratory Data Analysis')
    st.write('Summary of findings...')
    # Load and display your dataset
    df = pd.read_csv('data/insurance_test.csv')
    # st.write(df.head(10))
    # Conduct and display EDA 
    fig = px.histogram(df, x='age', title='Age Distribution')
    st.plotly_chart(fig)
    st.write(rf'Include your insights from the chart here...','\n', 'add as many charts are you feel are necessary.')



    
    # Generate and display charts
    # Example: st.bar_chart(df['your_column'])

def show_insights_page():
    st.title('Machine Learning Insights')
    # Display insights about the model or the data
    st.write('Insight 1: ...')
    st.write('Insight 2: ...')
    st.write('Insight 3: ...')
    st.write('Add insights as you feel are required.')

def show_test_model_page():
    st.title('Test the Model')
    
    # Input form for test data
    input_data = st.text_input('Enter your input data here:')
    #Include a way to upload a csv file 
    input_data = st.file_uploader('Upload a CSV file', type='csv')
    if st.button('Predict'):
        # Process input_data and predict
        # result = model.predict(processed_input_data)
        st.write('Prediction result: ...')

# Sidebar navigation
st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', ['EDA', 'Insights', 'Test Model'])

if page == 'EDA':
    show_eda_page()
elif page == 'Insights':
    show_insights_page()
elif page == 'Test Model':
    show_test_model_page()
