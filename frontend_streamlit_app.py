import streamlit as st
from langchain_community.vectorstores.mongodb_atlas import MongoDBAtlasVectorSearch
from langchain_openai.embeddings import OpenAIEmbeddings
#from langchain.chat_models import ChatOpenAI
#from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
#from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
#from pymongo import mongo_client

load_dotenv()

st.set_page_config(page_title="AsimovGPT", page_icon="🚀")

#st.title("Asimov's Foundation Universe GPT 🚀", help='You can ask questions related to the Foundation Universe (Robots Galactic Empire and Foundation Series) of Isaac Asimov. He was the "GOAT" of Sci-Fi... trust me, I am AI.')
st.title("Asimov's GPT 🚀", help='You can make queries related to the Foundation Universe (Robots Galactic Empire and Foundation Series) of Isaac Asimov. He was the "GOAT" of Sci-Fi... trust me, I am AI.')

user_query_count = 0 # to count number of queries made.


@st.cache_resource(ttl="1h") # to hold session for 1 hour.
def configure_retriever():
    vectorstore = MongoDBAtlasVectorSearch.from_connection_string(
            connection_string=os.getenv('MONGO_URI'),
            namespace='asimovgpt_db'+ "." + 'openai_embedding',
            embedding=OpenAIEmbeddings(disallowed_special=()),
            index_name='openai_embedding_vector_index',
        ) 
    retriever = vectorstore.as_retriever()

    return retriever


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


#class PrintRetrievalHandler(BaseCallbackHandler):
#    def __init__(self, container):
#        self.status = container.status("**Context Retrieval**")
#
#    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
#        self.status.write(f"**Question:** {query}")
#        self.status.update(label=f"**Context Retrieval:** {query}")
#
#    def on_retriever_end(self, documents, **kwargs):
#        for idx, doc in enumerate(documents):
#            source = os.path.basename(doc.metadata["source"])
#            self.status.write(f"**Document {idx} from {source}**")
#            self.status.markdown(doc.page_content)
#        self.status.update(state="complete")


#openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

#if not openai_api_key:
#    st.info("Please add your OpenAI API key to continue.")
#    st.stop()

openai_api_key = os.getenv('OPENAI_API_KEY')

retriever = configure_retriever()

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    memory_key="chat_history", 
    chat_memory=msgs, 
    return_messages=True)

# Setup LLM and QA chain
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", 
    openai_api_key=openai_api_key, 
    temperature=0.1, #0, 
    streaming=True
)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm, 
    retriever=retriever, 
    memory=memory, 
    verbose=False #True
)

if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("I'm R. Isaac Asimov (the 'R' stands for 'robot')... I can (try to) help you with queries about the Foundation Universe.")
    # R. Giskard Reventlov, R. Daneel Olivaw, R. Isaac Asimov
avatars = {"human": "user", "ai": "assistant"}

for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)
    
if user_query := st.chat_input(placeholder="Ask me anything related to the Foundation Universe."):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        #retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        #response = qa_chain.run(user_query, callbacks=[stream_handler]) #callbacks=[retrieval_handler, stream_handler])
        #response = qa_chain.invoke(user_query, callbacks=[stream_handler])
        response = qa_chain.run(user_query, callbacks=[stream_handler]) # apparently .run is deprecated, and .invoke should be used, but .invoke doesnt work properly.


#user_query_count = len([msg.content for msg in msgs.messages if msg.type=='human'])
user_query_count = len([1 for msg in msgs.messages if msg.type=='human'])

if user_query_count>0:
    st.sidebar.write('You have made ', str(user_query_count), ' queries so far.')

