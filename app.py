import streamlit as st

def main():
    st.title('Streamlit App')

    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Go to', ['Home'])

    if page == 'Home':
        st.write('Welcome to the Streamlit App')

if __name__ == '__main__':
    main()
