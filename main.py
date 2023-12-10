import streamlit as st 
from streamlit_chat import message
from dotenv import load_dotenv, find_dotenv
import os
from PyPDF2 import PdfReader 

from langchain.chat_models import ChatCohere
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import CohereEmbeddings
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
from langchain.vectorstores import qdrant
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient
from langchain.chains.question_answering import load_qa_chain
from qdrant_client.http.models import VectorParams, Distance
from langchain.schema import (
    SystemMessage, 
    HumanMessage, 
    AIMessage
)
from htmlTemplates import css, bot_template, user_template

def init():
    _ = load_dotenv(find_dotenv())
    # load the Cohere API key from the environment variable
    
    api_key = os.environ["COHERE_API_KEY"]
    if os.getenv('COHERE_API_KEY') is None or os.getenv('COHERE_API_KEY') == '':
        print('COHERE_API_KEY is not set')
        exit(1)
    else:
        print('COHERE_API_KEY is set')

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    return text

def get_text_chunks(text, chunk_size=1000, chunk_overlap=150):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "(?<=\. )", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    client = QdrantClient(
        os.getenv('QDRANT_HOST'),
        api_key=os.getenv('QDRANT_API_KEY')
    )

    # create collection
    #os.environ['QDRANT_COLLECTION'] = 'my_collections'
    coll_name='my_collections'
    
    collection_config = VectorParams(
            size=4096, # 768 for instructor-xl, 1536 for OpenAI
            distance=Distance.COSINE
        )

    client.recreate_collection(
        collection_name=coll_name,
        vectors_config=collection_config
    )
    
    embeddings = CohereEmbeddings()
    vector_store = qdrant.Qdrant(client=client, collection_name=coll_name, embeddings=embeddings)
    vector_store.add_texts(texts=text_chunks)
    return vector_store

def get_conversation_chain(vector_store, temperature=0):
    llm = ChatCohere(temperature=temperature)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
            
def main():
    init()

    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    
    st.write(css, unsafe_allow_html=True)
    
    hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)
    
    
    
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
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                print('raw_text length:', len(raw_text))
                print('raw_text type:', type(raw_text))
                
                #new_docs = [Document(page_content=doc) for doc in raw_text]
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                print('text_chunks length:', len(text_chunks))
                print('text_chunks type:', type(text_chunks))
                
                # create vector store
                vector_store = get_vectorstore(text_chunks)
                
                                
                # Build prompt
                template = """Use the following pieces of context to answer the question at the end.
                If you don't know the answer, just say that you don't know, don't try to make up an answer.
                Use three sentences maximum. Keep the answer as concise as possible. 
                Always say "thanks for asking!" at the end of the answer. 
                {context}
                Question: {question}
                Helpful Answer:"""
                QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
                
                qa_chain = RetrievalQA.from_chain_type(
                    llm=ChatCohere(temperature=0),
                    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
                    retriever=vector_store.as_retriever(),
                    return_source_documents=True
                )

                question = """'The most beautiful thing in the world must be ......'\n
                        can you complete it?"""
                response = qa_chain({"query": question})
                print(response["result"])
                print(response["source_documents"][0])

                #found_docs = vector_store.similarity_search("Yazılım Sürecinde Gereksinim ve Tasarım")
                #print(found_docs)
                
                # create conversation chain
                #st.session_state.conversation = get_conversation_chain(vector_store, temperature=0)
        
if __name__ == '__main__':
    main()
    
    