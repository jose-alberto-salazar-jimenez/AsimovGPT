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
from langchain_community.vectorstores.mongodb_atlas import MongoDBAtlasVectorSearch




#def get_vectorstore(text_chunks):
#    """Embeddings..
#    """
#    embeddings = OpenAIEmbeddings()
#    #embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
#    #embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-base')
#    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#    return vectorstore


#def get_conversation_chain(vectorstore):
#    """
#    """
#    llm = ChatOpenAI()
#    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
#    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
#    conversation_chain = ConversationalRetrievalChain.from_llm(
#        llm=llm,
#        retriever=vectorstore.as_retriever(),
#        memory = memory
#    )
#   return conversation_chain



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

 
def main():
    load_dotenv()
    st.set_page_config(page_title='AsimovGPT', page_icon=':robot_face:')
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header(':robot_face: AsimovGPT')
    #user_question = st.text_input(r"$\textsf{\Large Ask something about Isaac Asimov's Sci-Fi Universe*:")
    user_question = st.text_input("Ask something about Isaac Asimov's Sci-Fi Universe*:")#, clear_on_submit=True)

    if user_question:
        handle_userinput(user_question)




    vector_search = MongoDBAtlasVectorSearch.from_connection_string(
        MONGODB_ATLAS_CLUSTER_URI,
    DB_NAME + "." + COLLECTION_NAME,
    OpenAIEmbeddings(disallowed_special=()),
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
)

if __name__=='__main__':
    main()