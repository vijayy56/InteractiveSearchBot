READ ME FILE------------------
Required Environment:
o Python 3.10


Adopted Libraries:
•chainlit: A library for building chat applications.
•langchain: A library for natural language processing, document retrieval, and related tasks.
•os: The Python standard library module for interacting with the operating system.
•Chroma: A library for vectorizing and retrieving documents.


Flow of Execution:

Message Handling:
oThe @cl.on_message decorator indicates that the main function is triggered when a new message is received.
•Main Function (main):
oThe main function is an asynchronous function that processes incoming messages.
oIt calls the llm function to generate a response based on the content of the incoming message.
oThe response is then sent back using cl.Message(content=response).send().

•LLM Function (llm):
oConfigures the OpenAI API key in the environment.
oDefines a persistent directory for storing vectorized documents (persist_directory).
oInitializes an OpenAIEmbeddings instance for document embedding.
oCalls the loadResourceDocuments function to load and vectorize documents using Chroma.
oConfigures a retriever for the vectorized documents.
oUses a RetrievalQA chain with a ChatOpenAI language model to generate a response.
oThe response includes the result and the source of the information.
oThe result is printed, and the response is returned.

•Load Resource Documents Function (loadResourceDocuments):
oLoads documents from a PDF file using PyPDFLoader.
oSplits the documents into text chunks using CharacterTextSplitter.
oInitializes an OpenAIEmbeddings instance.
oCreates a Chroma vector database from the documents.
oPersists the vectorized documents.

Command to run the file


$pip install chainlit langchain chromadb openai pypdfloader


$ chainlit run response.py -w


Issue:
OpenAI API key might not be set up correctly. Need to double-check the OpenAI API key in the code. Ensure it is valid and has the necessary permissions.


Feedback:
We need to consider using a configuration file or environment variables for sensitive information like API keys, allowing for easier management.
