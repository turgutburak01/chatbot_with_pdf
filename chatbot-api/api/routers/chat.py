import random
from http.client import HTTPException
from typing import Annotated
from fastapi import APIRouter, Depends, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
import uuid
import os
from PyPDF2 import PdfReader 
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores.qdrant import Qdrant
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient, models
import fitz

load_dotenv(find_dotenv())

from helper.auth import verification

router = APIRouter()

def pdf_to_text(file_path):
    with open(file_path, 'rb') as f:
        pdf_reader = PdfReader(f)
        text = ""
        for pageNum in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[pageNum]
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


@router.get("/api/v1/uid")
def get_uid(Verification=Depends(verification)):
    my_uid = uuid.uuid4()
    print("Your id has been created. Your id is:" + str(my_uid))
    return my_uid


@router.post("/api/v1/upload-and-extract-text")
async def upload_and_extract_text(file: UploadFile = File(...), user_uid: str = Depends(get_uid)):

    try:
        user_folder_path = os.path.join("tmp", str(user_uid))
        os.makedirs(user_folder_path, exist_ok=True)
        pdf_path = os.path.join(user_folder_path, file.filename)

        with open(pdf_path, "wb") as pdf_file:
            pdf_file.write(file.file.read())

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")    
    
    text = pdf_to_text(pdf_path)
    chunks = get_text_chunks(text=text)
    return {"filename": file.filename, "chunk_size": len(chunks)}

@router.post("api/v1/text-to-chunks")
async def text_to_chunks(text: str = Depends(upload_and_extract_text)):
    try:
        chunks = get_text_chunks(text=text)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while splitting the text into chunks: {str(e)}")

    return chunks

@router.post(path="/api/v1/upload_file",
    description="An API for uploading PDF files and processing their content for vector database integration.",
    )
def chat(pdf_file: Annotated[UploadFile, File()], Verification=Depends(verification)):
    pdf_filename = pdf_file.filename
    pdf_size = round(len(pdf_file.file.read()) / (1024 ** 2), 2)

    return {"filename": pdf_filename, "size_in_mb": pdf_size}



