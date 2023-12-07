import chainlit as cl
import os 
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.document_loaders import UnstructuredPDFLoader,PyPDFLoader
import time
import json

from langchain.embeddings import HuggingFaceEmbeddings

@cl.on_message
async def main(message: cl.Message):
    response = llm(message.content)
    # Send a response back to the user
    await cl.Message(content=response).send()


def llm(query):
    os.environ["OPENAI_API_KEY"] = "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"
    # "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"    
    persist_directory = 'db'
    embedding = OpenAIEmbeddings()
    # loadResourceDocuments();
    vectordb = Chroma(persist_directory=persist_directory, 
                    embedding_function=embedding)

    retriever = vectordb.as_retriever(search_kwargs={"k": 2})
    qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-3.5-turbo-1106",verbose=True), 
                                    chain_type="stuff", 
                                    retriever=retriever, 
                                    return_source_documents=True)
    
    llm_response = qa_chain(query)
    s= llm_response['result'] +"\n\nSource of the information is: " +json.dumps(llm_response["source_documents"][0].metadata);
    print(llm_response)
    return s


def loadResourceDocuments():
    documents=[]
    loader = DirectoryLoader("/Users/vijay/Desktop/Semester 1/Network Security/Lectures",loader_cls=UnstructuredPowerPointLoader)
    loader2 = PyPDFLoader("/Users/vijay/Desktop/Semester 1/Network Security/TB/Stallingstest.pdf")
    loader3 = PyPDFLoader("/Users/vijay/Desktop/Semester 1/Network Security/TB/StallingsTextBook.pdf")
    time.sleep(60)

    documents.extend(loader3.load_and_split())
    time.sleep(50)
    documents.extend(loader2.load_and_split())
    time.sleep(60)
    documents.extend(loader.load_and_split())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10,separator="\n")
    texts = text_splitter.split_documents(documents)

    vectordb = None
    embedding = OpenAIEmbeddings()

    persist_directory = 'db'

    vectordb = Chroma.from_documents(documents=texts, 
                                    embedding=embedding,
                                    persist_directory=persist_directory)

    vectordb.persist()
