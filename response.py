import chainlit as cl
import os 
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.document_loaders import UnstructuredPDFLoader

@cl.on_message
async def main(message: cl.Message):
    response = llm(message.content)
    # Send a response back to the user
    await cl.Message(content=response).send()


def llm(query):
    os.environ["OPENAI_API_KEY"] = "API_KEY"    
    persist_directory = 'db'
    embedding = OpenAIEmbeddings()
    loadResourceDocuments();
    vectordb = Chroma(persist_directory=persist_directory, 
                    embedding_function=embedding)

    retriever = vectordb.as_retriever(search_kwargs={"k": 2})

    qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), 
                                    chain_type="stuff", 
                                    retriever=retriever, 
                                    return_source_documents=True)
    
    llm_response = qa_chain(query)
    s= llm_response['result']+"\n\nSource of the information is:"+llm_response["source_documents"][0].metadata['source'];
    print(s)
    return s


def loadResourceDocuments():
    loader2 = DirectoryLoader("/Users/vijay/Desktop/Semester 1/Network Security/Lectures",loader_cls=UnstructuredPowerPointLoader)
    # loader = DirectoryLoader("/Users/vijay/Desktop/Semester 1/Network Security/TB", glob='**/*.pdf',loader_cls=UnstructuredPDFLoader);
    documents = loader2.load()


    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
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
