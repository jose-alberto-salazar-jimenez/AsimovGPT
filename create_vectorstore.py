from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.mongodb_atlas import  MongoDBAtlasVectorSearch
from pymongo import MongoClient
from os import getenv
from pdf_docs_list import pdf_docs

#-----------------------------------------------
# BACK-END PARAMETERS
EXECUTE_EMBEDDING_TO_VECTORSTORE_PROCESS=False #True
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


def save_vectorstore(text_chunks):
    """Creates Embeddings from Text Chunks and save them in a Vetor Store
    """
    if EXECUTE_EMBEDDING_TO_VECTORSTORE_PROCESS:
        
        # create embeddings using OpenAI service/api
        openai_embeddings = OpenAIEmbeddings(getenv('OPENAI_API_KEY'))
        
        # insert embeddings to mongodb collection (a.k.a vectorstore) 
        mongodb_client = MongoClient(getenv('MONGO_URI'))

        MongoDBAtlasVectorSearch.from_texts(
            texts = text_chunks, 
            embedding = openai_embeddings, 
            collection = mongodb_client['asimovgpt_db']['openai_embedding'],
           index_name='embedding_vector_index'
        )


def main():

    # load .env variables (api keys)
    load_dotenv()

    # get raw text from pdf docs
    text_raw = get_pdf_text(pdf_docs)

    # get the text chunks
    text_chunks = get_text_chunks(text_raw)

    # create vectorstore
    vectorstore = save_vectorstore(text_chunks)


if __name__=='__main__':
    if EXECUTE_EMBEDDING_TO_VECTORSTORE_PROCESS:
        main()
    else:
        print("#####################")
        print("# Note: If you want to run the 'Embedding to Vectorstore' process, you have to change the parameter 'EXECUTE_EMBEDDING_TO_VECTORSTORE_PROCESS' to 'True'...")
        print("# WARNING: Doing this could represent a cost, giving that this uses OpenAI and MongoDB services...")
        print("#####################")