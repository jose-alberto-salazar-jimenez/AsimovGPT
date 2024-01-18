from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage
from langchain_openai import ChatOpenAI
import streamlit as st


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


user_query_qty = 0 #- to track quantity of queries made

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    #if "messages" in st.session_state:

    #    user_questions_qty = 1+len([msg.content for msg in st.session_state.messages if msg.role=='user'])
    #    st.write('You have asked ', user_questions_qty,' questions.') #-
        #print([msg.content for msg in st.session_state.messages if msg.role=='user'])
    #st.write()
    #st.write('You have made ', user_query_qty,' queries.') #-

if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="assistant", content="How can I help you?")]

for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)
    user_query_qty += 1 #-

if prompt := st.chat_input():
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)
    #user_query_qty += 1

    if not openai_api_key:
        user_query_qty = 0 #-
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
        
    if user_query_qty>5: #-
        message =  'You have made ' + str(user_query_qty) + ' queries without an OpenAI API key. Please add your OpenAI API key to continue.'
        st.info(message) #-
        st.stop() #-

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        llm = ChatOpenAI(openai_api_key=openai_api_key, streaming=True, callbacks=[stream_handler])
        response = llm.invoke(st.session_state.messages)
        st.session_state.messages.append(ChatMessage(role="assistant", content=response.content))

with st.sidebar: #-
    st.write('You have made ', user_query_qty,' queries.') #-
    #if "messages" in st.session_state: #-
        #user_query_qty = len([msg.content for msg in st.session_state.messages if msg.role=='user'])
        #st.write('You have made ', user_query_qty,' queries.') #-
    