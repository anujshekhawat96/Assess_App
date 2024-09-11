import os

# Define the directory structure and files
project_structure = {
    '': ['app.py', 'requirements.txt'],
    'data': [],
    'models': [],
    'utils': ['data_processing.py'],
}

# Content for app.py
app_py_content = """import streamlit as st

def main():
    st.title('Streamlit App')

    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Go to', ['Home'])

    if page == 'Home':
        st.write('Welcome to the Streamlit App')

if __name__ == '__main__':
    main()
"""

# Content for requirements.txt
requirements_txt_content = """streamlit
pandas
scikit-learn
"""

# Content for data_processing.py (placeholder content)
data_processing_py_content = """# Add your data processing functions here
"""

# Function to create directories and files
def create_project_structure(base_path, structure):
    for path, files in structure.items():
        dir_path = os.path.join(base_path, path)
        os.makedirs(dir_path, exist_ok=True)
        for file in files:
            file_path = os.path.join(dir_path, file)
            with open(file_path, 'w') as f:
                # Write initial content to files
                if file == 'app.py':
                    f.write(app_py_content)
                elif file == 'requirements.txt':
                    f.write(requirements_txt_content)
                elif file == 'data_processing.py':
                    f.write(data_processing_py_content)

# Use the current directory as the base path
base_path = os.getcwd()

# Create the project structure
create_project_structure(base_path, project_structure)

print("Streamlit project structure created successfully.")
