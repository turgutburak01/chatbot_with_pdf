import streamlit as st 
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader 
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
from langchain.vectorstores.qdrant import Qdrant
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient, models

_ = load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
qdrant_url = os.getenv('QDRANT_HOST')
qdrant_api_key=os.getenv('QDRANT_API_KEY')
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=openai_api_key)

template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Keep the answer as concise as possible. At the end of each answer, 
        reply You can ask anything else you are curious about about your document. 
        Answer the question in whatever language it is asked. 
        You should not use a language other than that language."""
rag_prompt = PromptTemplate.from_template(template)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    return text

def get_text_chunks(text, chunk_size=1200, chunk_overlap=150):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "(?<=\. )", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_conversation_chain(vector_store):
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    
    return conversation_chain

def get_qa_chain(vector_store):
    qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type='stuff',
                retriever=vector_store.as_retriever()
                )
    return qa_chain

def get_vector_store(coll_name='my_coll', embeddings=None, chunks=None):
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    client.recreate_collection(
        collection_name=coll_name,
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE))

    qdrant = Qdrant.from_texts(
                    url=qdrant_url,
                    api_key=qdrant_api_key,
                    texts=chunks,
                    embedding=embeddings,
                    collection_name=coll_name)
    
    return qdrant
def main():

    st.set_page_config(page_title="Chat with PDFs",
                       page_icon=":books:")
        
    hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)
    
    embeddings = OpenAIEmbeddings()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []  
    if "chain" not in st.session_state:
        st.session_state.chain = None
        
    st.sidebar.title("Chat with PDFs :books:")

    user_question = st.chat_input("Ask a question about your documents:")
    
    for message in st.session_state.messages:
        with st.chat_message(message.get("role")):
            st.write(message.get("content"))
            
    st.sidebar.subheader("Your documents")
    
    with st.sidebar:
        pdf = st.file_uploader(
        "Upload your PDFs here and click on 'Process'", accept_multiple_files=True, type="pdf")
        process = st.button("Process")
        
        if pdf and process:
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf)
                chunks = get_text_chunks(raw_text)
                qdrant = get_vector_store(embeddings=embeddings, chunks=chunks)
                st.session_state.chain = get_qa_chain(qdrant)
            
    if user_question and st.session_state.chain is not None:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)
        with st.spinner("Thinking..."):
            result = st.session_state.chain(user_question)["result"]
            st.session_state.messages.append({"role": "assistant", "content": result})
        with st.chat_message("assistant"):
            st.write(result)     
        
if __name__ == '__main__':
    main()
    
    