import streamlit as st
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from html_template import css, bot_template, user_template
from langchain_community.vectorstores.mongodb_atlas import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from dotenv import load_dotenv
from os import getenv


def get_conversation_chain(vectorstore):
    """
    """
    llm = ChatOpenAI(temperature=0.1)
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

    human_history = [message for i, message in enumerate(st.session_state.chat_history) if i % 2 == 0][::-1][:10] 
    bot_history = [message for i, message in enumerate(st.session_state.chat_history) if i % 2 == 1][::-1][:10] 

    for h, b in zip(human_history, bot_history):
        st.write(user_template.replace("{{MSG}}", h.content), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", b.content), unsafe_allow_html=True)

 
def main2():
    load_dotenv()

    st.set_page_config(page_title='AsimovGPT', page_icon=':robot_face:')
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header(':robot_face: AsimovGPT')
    user_question = st.text_input("Ask something about Isaac Asimov's Sci-Fi Universe*:")#, clear_on_submit=True)

    if user_question:
        handle_userinput(user_question)

        vectorstore = MongoDBAtlasVectorSearch.from_connection_string(
            connection_string=getenv('MONGO_URI'),
            namespace='asimovgpt_db'+ "." + 'openai_embedding',
            embedding=OpenAIEmbeddings(disallowed_special=()),
            index_name='openai_embedding_vector_index',
        )
        
        retriever = vectorstore.as_retriever()

        # create conversation chain
        st.session_state.conversation = get_conversation_chain(retriever)



def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        #pdf_docs = st.file_uploader(
        #    "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                #raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                #text_chunks = get_text_chunks(raw_text)

                # create vector store
                #vectorstore = get_vectorstore(text_chunks)
                vectorstore = MongoDBAtlasVectorSearch.from_connection_string(
                    connection_string=getenv('MONGO_URI'),
                    namespace='asimovgpt_db'+ "." + 'openai_embedding',
                    embedding=OpenAIEmbeddings(disallowed_special=()),
                    index_name='openai_embedding_vector_index',
                )
                
                retriever = vectorstore.as_retriever()

                

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)



if __name__=='__main__':
    main()