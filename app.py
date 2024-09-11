import streamlit as st
# from sklearn.externals import joblib  
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
# Load your machine learning model
# model = joblib.load('models/your_model.pkl')

# Define a function for each page
def show_eda_page():
    st.title('Exploratory Data Analysis')
    st.write('Summary of findings...')
    # Load and display your dataset
    df_insurance = pd.read_csv('insurance_test.csv')
    # st.write(df.head(10))
    # Conduct and display EDA 
    fig = px.histogram(df_insurance, x='age', title='Age Distribution')
    st.plotly_chart(fig)
    st.write(rf'Include your insights from the chart here...','\n', 'add as many charts are you feel are necessary.')
    all_eda_methods(df_insurance)
    

    
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


def all_eda_methods(df):


    def dataset_overview(df):
        st.header("Dataset Overview")
        st.write("Number of Rows:", df.shape[0])
        st.write("Number of Columns:", df.shape[1])
       
          # Use the duplicated() function to identify duplicate rows
        row_count = df.shape[0]
 
        column_count = df.shape[1]
        duplicates = df[df.duplicated()]
        duplicate_row_count =  duplicates.shape[0]
 
        missing_value_row_count = df[df.isna().any(axis=1)].shape[0]
 
        table_markdown = f"""
          | Description | Value | 
          |---|---|
          | Number of Rows | {row_count} |
          | Number of Columns | {column_count} |
          | Number of Duplicated Rows | {duplicate_row_count} |
          | Number of Rows with Missing Values | {missing_value_row_count} |
          """
        st.markdown(table_markdown)
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        
 
       
 
       
     
    
        st.header("Summary Statistics")
        st.write(df.describe())
       


    def correlation_heatmap(df):
        st.header("Correlation Heatmap")
        
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        #st.pyplot(fig)
    dataset_overview(df)
    correlation_heatmap(df)
    
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    selected_num_col = st.selectbox("Which numeric column do you want to explore?", numeric_cols)
    st.header(f"{selected_num_col} - Statistics")
     
    col_info = {}
    col_info["Number of Unique Values"] = len(df[selected_num_col].unique())
    col_info["Number of Rows with Missing Values"] = df[selected_num_col].isnull().sum()
    col_info["Number of Rows with 0"] = df[selected_num_col].eq(0).sum()
    col_info["Number of Rows with Negative Values"] = df[selected_num_col].lt(0).sum()
    col_info["Average Value"] = df[selected_num_col].mean()
    col_info["Standard Deviation Value"] = df[selected_num_col].std()
    col_info["Minimum Value"] = df[selected_num_col].min()
    col_info["Maximum Value"] = df[selected_num_col].max()
    col_info["Median Value"] = df[selected_num_col].median()
    info_df = pd.DataFrame(list(col_info.items()), columns=['Description', 'Value'])
    st.dataframe(info_df)

 
        
 

   
    

# Sidebar navigation
st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', ['EDA', 'Insights', 'Test Model'])

if page == 'EDA':
    show_eda_page()
elif page == 'Insights':
    show_insights_page()
elif page == 'Test Model':
    show_test_model_page()

