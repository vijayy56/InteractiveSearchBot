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

@cl.on_message
async def main(message: cl.Message):
    response = llm(message.content)
    # Send a response back to the user
    await cl.Message(content=response).send()


def llm(query):
    os.environ["OPENAI_API_KEY"] = "API_KEY"    
    persist_directory = 'db'
    embedding = OpenAIEmbeddings()
    # loadResourceDocuments();
    vectordb = Chroma(persist_directory=persist_directory, 
                    embedding_function=embedding)

    retriever = vectordb.as_retriever(search_kwargs={"k": 2})

    qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-3.5-turbo-0613"), 
                                    chain_type="stuff", 
                                    retriever=retriever, 
                                    return_source_documents=True)
    
    llm_response = qa_chain(query)
    s= llm_response['result']+"\n\nSource of the information is:"+llm_response["source_documents"][0].metadata['source'];
    print(s)
    return s


def loadResourceDocuments():
    documents=[]
    # loader2 = DirectoryLoader("/Users/vijay/Desktop/Semester 1/Network Security/Lectures",loader_cls=UnstructuredPowerPointLoader)
    # loader = DirectoryLoader("/Users/vijay/Desktop/Semester 1/Network Security/TB", glob='**/*.pdf',loader_cls=UnstructuredPDFLoader);
    loader2 = PyPDFLoader("/Users/vijay/Desktop/Semester 1/Network Security/TB/Stallingstest.pdf")
    documents.extend(loader2.load())
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100,separator="\n")
    texts = text_splitter.split_documents(documents)
    vectordb = None
    embedding = OpenAIEmbeddings()
    persist_directory = 'db'

    vectordb = Chroma.from_documents(documents=texts, 
                                    embedding=embedding,
                                    persist_directory=persist_directory)

    vectordb.persist()

# def create_agent_chain():
#     model_name = "gpt-3.5-turbo"
#     llm = ChatOpenAI(model_name=model_name)
#     chain = load_qa_chain(llm, chain_type="stuff")
#     return chain
