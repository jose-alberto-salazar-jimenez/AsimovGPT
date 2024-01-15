import streamlit as st
from dotenv import load_dotenv

def main():
    load_dotenv()
    st.set_page_config(page_title='chat with pdfs', page_icon=':books:')
    st.header('chat with multiple pdfs :books:')
    st.text_input("ask a question about Isaac Asimov's Sci-Fi Universe*:")

    with st.sidebar:
        st.subheader('Your Documents')
        st.file_uploader('Upload pdfs and click on process')
        st.button('Process')



if __name__=='__main__':
    main()