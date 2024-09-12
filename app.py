import streamlit as st
 
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
# Load your machine learning model
# model = joblib.load('models/your_model.pkl')


##global data read 
df_insurance = pd.read_csv('insurance_test.csv')
# Define a function for each page

def show_eda_page():
    st.title('Exploratory Data Analysis')
    
    # Load and display your dataset
    df_insurance = pd.read_csv('insurance_test.csv')
    # st.write(df.head(10))
    # Conduct and display EDA 
    fig = px.histogram(df_insurance, x='age', title='Age Distribution')
    st.plotly_chart(fig)
    
    ##EDA METHODS
    all_eda_methods(df_insurance)
    

    
    # Generate and display charts
    # Example: st.bar_chart(df['your_column'])

def show_insights_page():
    st.title('Machine Learning Insights')
    # Display insights about the model or the data
    all_insights_methods(df_insurance)
   
  

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
        st.pyplot(fig)
    dataset_overview(df)
    #correlation_heatmap(df)
    
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
    
    cat_cols = df.select_dtypes(include='object')
    cat_cols_names = cat_cols.columns.tolist()
 

    selected_cat_col = st.selectbox("Which text column do you want to explore?", cat_cols_names)
 
    st.header(f"{selected_cat_col}")
     
    # add categorical column stats
    cat_col_info = {}
    cat_col_info["Number of Unique Values"] = len(df[selected_cat_col].unique())
    cat_col_info["Number of Rows with Missing Values"] = df[selected_cat_col].isnull().sum()
    cat_col_info["Number of Empty Rows"] = df[selected_cat_col].eq("").sum()
    cat_col_info["Number of Rows with Only Whitespace"] = len(df[selected_cat_col][df[selected_cat_col].str.isspace()])
    cat_col_info["Number of Rows with Only Lowercases"] = len(df[selected_cat_col][df[selected_cat_col].str.islower()])
    cat_col_info["Number of Rows with Only Uppercases"] = len(df[selected_cat_col][df[selected_cat_col].str.isupper()])
    cat_col_info["Number of Rows with Only Alphabet"] = len(df[selected_cat_col][df[selected_cat_col].str.isalpha()])
    cat_col_info["Number of Rows with Only Digits"] = len(df[selected_cat_col][df[selected_cat_col].str.isdigit()])
    cat_col_info["Mode Value"] = df[selected_cat_col].mode()[0]
 
    cat_info_df = pd.DataFrame(list(cat_col_info.items()), columns=['Description', 'Value'])
    st.dataframe(cat_info_df)

 
    


def all_insights_methods(df):
    # 1. Pie chart for the distribution of smokers vs non-smokers
    st.subheader("Proportion of Smokers vs Non-Smokers (Pie Chart)")
    smoker_counts = df['smoker'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(smoker_counts, labels=smoker_counts.index, autopct='%1.1f%%', startangle=90, colors=["#66b3ff", "#99ff99"])
    ax.set_title('Proportion of Smokers vs Non-Smokers')
    st.pyplot(fig)
    
    # 2. Pie chart for the distribution of gender
    st.subheader("Proportion of Male vs Female (Pie Chart)")
    gender_counts = df['sex'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=["#ff9999", "#66b3ff"])
    ax.set_title('Proportion of Male vs Female')
    st.pyplot(fig)
    

    # 2. Distribution of BMI and its effect on medical charges (Scatter plot)
    st.subheader("BMI vs Medical Charges (Scatter Plot)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='bmi', y='charges', hue='smoker', data=df, ax=ax)
    ax.set_title('BMI vs Medical Charges')
    st.pyplot(fig)

    # 3. Bar chart for average medical charges based on smoker status
    st.subheader("Average Medical Charges by Smoker Status (Bar Chart)")
    smoker_charges = df.groupby('smoker')['charges'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='smoker', y='charges', data=smoker_charges, ax=ax)
    ax.set_title('Average Medical Charges by Smoker Status')
    st.pyplot(fig)

    # 4. Histogram for the distribution of BMI
    st.subheader("Distribution of BMI (Histogram)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['bmi'], bins=30, kde=True, ax=ax)
    ax.set_title('Distribution of BMI')
    st.pyplot(fig)

    # 5. Region-wise analysis of medical charges (Bar Chart)
    st.subheader("Average Medical Charges by Region (Bar Chart)")
    region_charges = df.groupby('region')['charges'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='region', y='charges', data=region_charges, ax=ax)
    ax.set_title('Average Medical Charges by Region')
    st.pyplot(fig)

    # 6. Children and its effect on medical charges (Bar Chart)
    st.subheader("Average Medical Charges by Number of Children (Bar Chart)")
    children_charges = df.groupby('children')['charges'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='children', y='charges', data=children_charges, ax=ax)
    ax.set_title('Average Medical Charges by Number of Children')
    st.pyplot(fig)

    # 7. Medical charges distribution for male vs female (Bar Chart)
    st.subheader("Average Medical Charges by Gender (Bar Chart)")
    gender_charges = df.groupby('sex')['charges'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='sex', y='charges', data=gender_charges, ax=ax)
    ax.set_title('Average Medical Charges by Gender')
    st.pyplot(fig)
    

# Sidebar navigation
st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', ['EDA', 'Insights', 'Test Model'])

if page == 'EDA':
    show_eda_page()
elif page == 'Insights':
    show_insights_page()
elif page == 'Test Model':
    show_test_model_page()

