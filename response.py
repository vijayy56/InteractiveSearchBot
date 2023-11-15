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


@cl.on_message
async def main(message: cl.Message):
    # Your custom logic goes here...
    response = llm(message.content)
    # Send a response back to the user
    await cl.Message(content=response).send()


def llm(query):
    os.environ["OPENAI_API_KEY"] = "USE_YOUR OWN GPT API KEY"
    #loader = UnstructuredPowerPointLoader("./Docs/lecture 1.pptx",mode="elements")
    #loader2 = DirectoryLoader("/Users/vijay/Desktop/Semester 1/Network Security/Lectures",loader_cls=UnstructuredPowerPointLoader)

    #documents = loader2.load()


    persist_directory = 'db'
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # texts = text_splitter.split_documents(documents)
    # here we are using OpenAI embeddings but in future we will swap out to local embeddings
    embedding = OpenAIEmbeddings()
    #vectordb = None

    # vectordb = Chroma.from_documents(documents=texts, 
    #                                 embedding=embedding,
    #                                 persist_directory=persist_directory)


    #vectordb.persist()

    vectordb = Chroma(persist_directory=persist_directory, 
                    embedding_function=embedding)
    retriever = vectordb.as_retriever()

    retriever = vectordb.as_retriever(search_kwargs={"k": 2})

    qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), 
                                    chain_type="stuff", 
                                    retriever=retriever, 
                                    return_source_documents=True)


    llm_response = qa_chain(query)
    s= llm_response['result']+"\n\nSource of the information is:"+llm_response["source_documents"][0].metadata['source']+llm_response["source_documents"][1].metadata['source'];
    print(llm_response["source_documents"])
    return s
