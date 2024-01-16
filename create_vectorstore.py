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
#from langchain.vectorstores.mongodb_atlas import  MongoDBAtlasVectorSearch #FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
#from langchain.llms import openai
#from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from html_template import css, bot_template, user_template

from pymongo import MongoClient
from langchain_community.vectorstores.mongodb_atlas import  MongoDBAtlasVectorSearch
from langchain_community.document_loaders import PyPDFLoader


#-----------------------------------------------

# BACK-END PARAMETERS
backend_params = {
    'RECREATE_EMBEDDINGS':False #True, #False,
    'OVERRIDE_VECTORSTORE':False #True, #False
    'DBNAME':'123',
}

#-----------------------------------------------

# PDF FILES LOCATION FOR TEXT EXTRACTION
pdf1 = 'pdf_data/wikipedia_isaac_asimov.pdf'
pdf2 = 'pdf_data/wikipedia_psychohistory_fictional.pdf'
pdf3 = 'pdf_data/'
pdf4 = 'pdf_data/'
pdf5 = 'pdf_data/'
pdf6 = 'pdf_data/'
pdf7 = 'pdf_data/the_complete_asimov.pdf'

pdf_docs = [pdf1]

#-----------------------------------------------


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
    """Creates Embeddings from Text Chunks and save them in a Vetor Store
    """
    if backend_params['RECREATE_EMBEDDINGS']:
        embeddings = OpenAIEmbeddings(openai_api_key='')#, temperature=0.1)
    if backend_params['OVERRIDE_VECTORSTORE']:
        #pass
        #vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        conn_string = ''
        print('here1')
        client = MongoClient(conn_string)
        collection = client['asimovgpt_db']['asimovgpt_embeddings']
        print('here2')
        # Insert the documents in MongoDB Atlas with their embedding
        #docsearch = MongoDBAtlasVectorSearch.from_documents(
        #    docs, embeddings, collection=collection, index_name=index_name
        #)
        vectorstore = MongoDBAtlasVectorSearch.from_texts(text_chunks, embeddings, collection=collection)
        #text_chunks, embeddings, collection=collection#, index_name=index_name)
        print('here3')
    #return vectorstore


def main():

    # get pdf text
    raw_text = get_pdf_text(pdf_docs)

    # get the text chunks
    text_chunks = get_text_chunks(raw_text)

    # create vector store
    vectorstore = get_vectorstore(text_chunks)




if __name__=='__main__':
    main()