READ ME FILE------------------
Required Environment:
•
Python Environment:
o
Python 3.10
Adopted Libraries:
•
chainlit: A library for building chat applications.
•
langchain: A library for natural language processing, document retrieval, and related tasks.
•
os: The Python standard library module for interacting with the operating system.
•
Chroma: A library for vectorizing and retrieving documents.
Flow of Execution:
•
Message Handling:
o
The @cl.on_message decorator indicates that the main function is triggered when a new message is received.
•
Main Function (main):
o
The main function is an asynchronous function that processes incoming messages.
o
It calls the llm function to generate a response based on the content of the incoming message.
o
The response is then sent back using cl.Message(content=response).send().
•
LLM Function (llm):
o
Configures the OpenAI API key in the environment.
o
Defines a persistent directory for storing vectorized documents (persist_directory).
o
Initializes an OpenAIEmbeddings instance for document embedding.
o
Calls the loadResourceDocuments function to load and vectorize documents using Chroma.
o
Configures a retriever for the vectorized documents.
o
Uses a RetrievalQA chain with a ChatOpenAI language model to generate a response.
o
The response includes the result and the source of the information.
o
The result is printed, and the response is returned.
•
Load Resource Documents Function (loadResourceDocuments):
o
Loads documents from a PDF file using PyPDFLoader.
o
Splits the documents into text chunks using CharacterTextSplitter.
o
Initializes an OpenAIEmbeddings instance.
o
Creates a Chroma vector database from the documents.
o
Persists the vectorized documents.
Command to run
Command to run the file
$pip install chainlit langchain chromadb openai pypdfloader
$ chainlit run response.py -w
Issue:
OpenAI API key might not be set up correctly. Need to double-check the OpenAI API key in the code. Ensure it is valid and has the necessary permissions.
Feedback:
We need to consider using a configuration file or environment variables for sensitive information like API keys, allowing for easier management.
