import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
#from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings
#from langchain.vectorstores import faiss
#from langchain_community.vectorstores import FAISS
from langchain.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
#from langchain.llms import openai
#from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI


def get_pdf_text(pdf_docs):
    """Extract text from pdf files, returning a concatenated text.
    """
    text = ''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text = page.extract_text()    
    return text

def get_text_chunks(text_raw):
    """Separate a text into chunks of it (a list) with a given size.
    """
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text_raw)
    return chunks

def get_vectorstore(text_chunks):
    """Embeddings..
    """
    embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    """
    """
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory = memory
    )
    return conversation


def main():
    load_dotenv()
    st.set_page_config(page_title='chat with pdfs', page_icon=':books:')
    st.header('chat with multiple pdfs :books:')
    st.text_input("ask a question about Isaac Asimov's Sci-Fi Universe*:")


    if 'conversation' not in st.session_state:
        st.session_state.conversation = None

    with st.sidebar:
        st.subheader('Your Documents')
        pdf_docs = st.file_uploader('Upload pdfs and click on process', accept_multiple_files=True)
        if st.button('Process'):
            with st.spinner('Processing...'):
                raw_text = get_pdf_text(pdf_docs)


                text_chunks = get_text_chunks(raw_text)
                #st.write(text_chunks)

                vectorstore = get_vectorstore(text_chunks)
                
                #conversation = get_conversation_chain(vectorstore)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                
    #st.session_state.conversation



if __name__=='__main__':
    main()