import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
#from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings
#from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
#from langchain.vectorstores import faiss
#from langchain_community.vectorstores import FAISS
from langchain.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
#from langchain.llms import openai
#from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from html_template import css, bot_template, user_template


def get_pdf_text(pdf_docs):
    """Extract text from pdf files, returning a concatenated text.
    """
    text = ''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()    
    return text


def get_text_chunks(text):
    """Separate a text into chunks of it (a list) with a given size.
    """
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    """Embeddings..
    """
    embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
    #embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-base')
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    """
    """
    llm = ChatOpenAI()
    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory = memory
    )
    return conversation_chain



def handle_userinput(user_question):
    """Saves chat history in session, and displays input/output messages.
    """
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    #for i, message in enumerate(st.session_state.chat_history):
    #    if i % 2 == 0:
    #        st.write(user_template.replace(
    #            "{{MSG}}", message.content), unsafe_allow_html=True)
    #    else:
    #        st.write(bot_template.replace(
    #            "{{MSG}}", message.content), unsafe_allow_html=True)

    human_history = [message for i, message in enumerate(st.session_state.chat_history) if i % 2 == 0][::-1][:10] 
    bot_history = [message for i, message in enumerate(st.session_state.chat_history) if i % 2 == 1][::-1][:10] 

    for h, b in zip(human_history, bot_history):
        st.write(user_template.replace("{{MSG}}", h.content), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", b.content), unsafe_allow_html=True)