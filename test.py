from langchain_community.vectorstores.mongodb_atlas import MongoDBAtlasVectorSearch
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

conn_string = 'mongodb+srv://jasj1991:hvC36CUIq7tq3jZE@cluster-asimovgpt.jbwpf6l.mongodb.net/'
print('here1')
#client = MongoClient(conn_string)
#collection = client['asimovgpt_db']['asimovgpt_embeddings']


vector_search = MongoDBAtlasVectorSearch.from_connection_string(
    conn_string,
    'asimov_db'+ "." + 'asimov_embeddings',
    OpenAIEmbeddings(openai_api_key='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', disallowed_special=()),
    #index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
)

# Perform a similarity search between the embedding of the query and the embeddings of the documents
#query = "What were the compute requirements for training GPT 4"
#results = vector_search.similarity_search(query)

#print(results[0].page_content)
query = "Whos was Asimov"

results = vector_search.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 25},
)
print(query)
print(results)
# Display results
#for result in results:
#    print(result)


llm = ChatOpenAI(openai_api_key='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')
#    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
#    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
conversation_chain = ConversationalRetrievalChain.from_llm(
       llm=llm,
       retriever=vector_search.as_retriever(),
   )
print(conversation_chain)