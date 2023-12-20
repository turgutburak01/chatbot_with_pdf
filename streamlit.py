from dotenv import load_dotenv, find_dotenv
import os
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, Document, StorageContext, ServiceContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
import streamlit as st
from llama_index.llms import OpenAI


_ = load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
qdrant_url = os.getenv('QDRANT_HOST')
qdrant_api_key=os.getenv('QDRANT_API_KEY')
llm = OpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=256)

def save_uploadedfile(uploaded_file):
    with open(os.path.join("data", uploaded_file[0].name), "wb") as file:
        file.write(uploaded_file[0].getbuffer())

def search(query, vector_store):
    documents = SimpleDirectoryReader('data').load_data()
    service_context = ServiceContext.from_defaults(llm=llm, chunk_size=800, chunk_overlap=20)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = GPTVectorStoreIndex.from_documents(documents=documents, service_context=service_context, storage_context=storage_context)
    
    query_engine = index.as_query_engine(streaming=True)
    response = query_engine.query(query)
    
    return response

coll_name='my_data'
client = qdrant_client.QdrantClient(url=qdrant_url, api_key=qdrant_api_key)


def main():
    
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    
    hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)
    
    
    if "messages" not in st.session_state:
        st.session_state.messages = []  
    if "qdrant" not in st.session_state:
        st.session_state.qdrant = None 
    
        
    st.sidebar.title("Chat with multiple PDFs :books:")

    user_question = st.chat_input("Ask a question about your documents:")
    
    for message in st.session_state.messages:
        with st.chat_message(message.get("role")):
            st.write(message.get("content"))
            
    st.sidebar.subheader("Your documents")
    pdf = st.sidebar.file_uploader(
        "Upload your PDFs here and click on 'Process'", accept_multiple_files=True, type="pdf")
    
    process = st.sidebar.button("Process")

    if pdf:
        save_uploadedfile(pdf)     
        

    if user_question:
        vector_store = QdrantVectorStore(client=client, collection_name=coll_name)
        response = search(user_question, vector_store)
        print(response)


if __name__ == '__main__':
    main()